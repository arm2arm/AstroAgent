"""
CrewAI Workflow for Code and Data Tasks

Pipeline
--------
classify  ─┬── simple_path ──► coding ──► execution ──► review ──► check ─┬─► done
            │                                                     ▲       │
            └── complex_path ──► plan ──► analyse ──► coding ─────┘       │
                                                                          │
                                                      needs_revision ◄────┘
                                                          │
                                                          ▼
                                                     revise_code ──► execution ──► review ──► check …

The @router() on *classify* picks the path.  The @router() on *check_approval*
drives the Coder↔Reviewer feedback loop (up to max_review_iterations rounds).
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from crewai import Crew, Process, Task
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel, Field

from agents import AgentFactory, get_task_template
from config import (
    get_execution_config,
    get_workflow_config,
    get_llm_config,
    get_storage_config,
    get_crewai_embedder_dict,
)
from token_budget import build_budget, estimate_tokens, truncate_text


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class WorkflowState(BaseModel):
    """Shared state threaded through every phase."""

    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = Field(default_factory=datetime.now)

    # User inputs
    research_question: str = ""
    task_complexity: int = -1          # -1 = unknown, use LLM heuristic

    # Agent outputs
    analysis_plan: Optional[str] = None
    statistical_approach: Optional[str] = None
    generated_code: Optional[str] = None
    execution_stdout: str = ""
    execution_stderr: str = ""
    execution_artifacts: List[str] = Field(default_factory=list)
    review_report: Optional[str] = None

    # Revision loop
    iterations: int = 0
    max_iterations: int = 3
    approved: bool = False

    # Status tracking (for UI progress)
    status: str = "initialized"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COMPLEXITY_THRESHOLD = 3   # <= 3 -> simple path


def _extract_code_block(text: str) -> str:
    """Pull the first fenced Python code-block out of LLM output, or
    return the full text if no fences are found."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _sanitize_code(code: str) -> str:
    """Fix common LLM code-generation mistakes before execution.

    Repairs:
    - ``matplotlib.use(...)`` without a preceding ``import matplotlib``
    - ``plt.show()`` calls → replaced with ``plt.savefig`` when missing
    - Ensures ``matplotlib.use('Agg')`` is present when pyplot is imported
    - Detects functions that create plots but are never called, and adds
      calls at the bottom of the script
    - Removes blank lines between every line (LLM sometimes double-spaces)
    """
    lines = code.splitlines()
    out: list[str] = []
    has_import_mpl = False
    has_mpl_use_agg = False
    has_savefig = False
    has_pyplot_import = False
    show_lines: list[int] = []           # indices in *out* of plt.show() lines
    defined_funcs: set[str] = set()      # function names defined in the script
    called_names: set[str] = set()       # names that appear as calls
    plot_funcs: set[str] = set()         # functions that contain plt.* calls

    # --- First pass: collect info ---
    current_func: str | None = None
    for line in lines:
        s = line.strip()
        # Track function definitions
        if s.startswith("def ") and s.endswith(":"):
            fname = s[4:s.index("(")].strip() if "(" in s else None
            if fname:
                defined_funcs.add(fname)
                current_func = fname
        elif s and not s.startswith(" ") and not s.startswith("\t") and not s.startswith("#"):
            # Top-level non-function code
            current_func = None

        # Track plot-related calls inside functions
        if current_func and ("plt." in s or "savefig" in s or "ax." in s):
            plot_funcs.add(current_func)

        # Track savefig
        if "savefig" in s:
            has_savefig = True
        if "import matplotlib.pyplot" in s or "from matplotlib" in s:
            has_pyplot_import = True
        if s in ("import matplotlib", "import matplotlib as mpl"):
            has_import_mpl = True
        if "matplotlib.use(" in s:
            has_mpl_use_agg = True

        # Track function calls (simple heuristic)
        for fn in defined_funcs:
            if f"{fn}(" in s and not s.startswith("def "):
                called_names.add(fn)

    # --- Second pass: build output with fixes ---
    prev_blank = False
    savefig_counter = [0]     # mutable counter for auto-generated filenames

    for line in lines:
        stripped = line.strip()

        # Collapse runs of blank lines to at most one
        if stripped == "":
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False

        # Track whether 'import matplotlib' already appeared
        if stripped in ("import matplotlib", "import matplotlib as mpl"):
            has_import_mpl = True

        # Insert 'import matplotlib' before matplotlib.use(...) if missing
        if not has_import_mpl and stripped.startswith("matplotlib.use("):
            out.append("import matplotlib")
            has_import_mpl = True

        # Replace plt.show() with plt.savefig() if no savefig exists
        if stripped == "plt.show()" or stripped.startswith("plt.show("):
            if not has_savefig:
                # Insert savefig before the show
                indent = line[: len(line) - len(line.lstrip())]
                savefig_counter[0] += 1
                fname = f"output_{savefig_counter[0]}.png" if savefig_counter[0] > 1 else "output.png"
                out.append(f'{indent}plt.savefig("{fname}", dpi=150, bbox_inches="tight")')
                out.append(f'{indent}print("SAVED: {fname}")')
            out.append(line[: len(line) - len(line.lstrip())] + "# plt.show()  # removed for headless execution")
            continue

        out.append(line)

    # --- Post-pass: add missing function calls at the bottom ---
    uncalled_plot_funcs = sorted(plot_funcs - called_names)
    # Also check for any defined-but-uncalled functions that look like
    # they do useful work (heuristic: name contains 'plot', 'save', 'generate', 'create', 'run', 'main')
    action_keywords = {"plot", "save", "generate", "create", "run", "main", "draw", "compute", "calculate", "analyze"}
    uncalled_action_funcs = sorted(
        fn for fn in (defined_funcs - called_names - plot_funcs)
        if any(kw in fn.lower() for kw in action_keywords)
    )
    funcs_to_call = uncalled_plot_funcs + uncalled_action_funcs

    if funcs_to_call:
        out.append("")
        out.append("# --- Auto-added: call defined functions ---")
        out.append('if __name__ == "__main__":')
        for fn in funcs_to_call:
            out.append(f"    {fn}()")

    # --- Ensure Agg backend is set when pyplot is imported ---
    if has_pyplot_import and not has_mpl_use_agg:
        # Insert matplotlib.use('Agg') right before the first pyplot import
        new_out: list[str] = []
        inserted = False
        for line in out:
            s = line.strip()
            if not inserted and ("import matplotlib.pyplot" in s or "from matplotlib import pyplot" in s):
                if not has_import_mpl:
                    new_out.append("import matplotlib")
                new_out.append("matplotlib.use('Agg')")
                inserted = True
            new_out.append(line)
        out = new_out

    return "\n".join(out)


def _discover_artifacts(results_dir: str) -> list[str]:
    """Find image files produced by execution in *results_dir*."""
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.svg", "*.pdf")
    found: list[str] = []
    for pattern in patterns:
        found.extend(glob.glob(os.path.join(results_dir, pattern)))
    return sorted(found)


def _run_code_in_docker_direct(
    code: str,
    image: str,
    results_dir: str,
    libraries: list[str] | None = None,
    timeout: int = 300,
) -> tuple[str, str]:
    """Run *code* inside a Docker container, bind-mounting *results_dir*.

    This is a **direct** fallback that does not depend on the LLM calling
    the CodeInterpreterTool.  It uses the Docker CLI so it works even when
    the ``docker`` Python SDK is not importable.

    Returns (stdout, stderr).
    """
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    container_name = "code-interpreter-direct"
    # Remove stale container (ignore errors)
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Write script to results dir so it's available inside the container
    script_path = os.path.join(results_dir, "_run_script.py")
    with open(script_path, "w") as f:
        f.write(code)

    # Install libraries inside the container and run code
    install_cmds = ""
    if libraries:
        pkgs = " ".join(libraries)
        install_cmds = f"pip install --quiet {pkgs} 2>/dev/null; "

    shell_cmd = f'{install_cmds}python3 /workspace/_run_script.py'

    cmd = [
        "docker", "run",
        "--name", container_name,
        "--rm",
        "-v", f"{results_dir}:/workspace",
        "-w", "/workspace",
        image,
        "sh", "-c", shell_cmd,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return "", f"Execution timed out after {timeout}s"
    except Exception as exc:
        return "", f"Docker direct execution failed: {exc}"
    finally:
        # Clean up temp script
        try:
            os.remove(script_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class AstronomyWorkflow(Flow[WorkflowState]):
    """
    Orchestrates the full task pipeline.

    Simple tasks  (complexity <= 3): Coder -> Executor -> Reviewer (+ loop)
    Complex tasks (complexity > 3):  Planner -> Analyst -> Coder -> Executor -> Reviewer (+ loop)
    """

    def __init__(self, state: WorkflowState):
        # Set instance attributes BEFORE super().__init__() because Flow's
        # __init__ calls dir(self)/getattr which triggers @property accessors.
        self.wf_config = get_workflow_config()
        self.exec_config = get_execution_config()
        self.storage_config = get_storage_config()

        # Agent factory (loads config/agents.yaml)
        self._agent_factory = AgentFactory()

        # Agents (created lazily via @property)
        self._planner = None
        self._analyst = None
        self._coder = None
        self._reviewer = None
        self._summarizer = None

        super().__init__(**state.model_dump())

        # Per-workflow results directory
        self._results_dir = os.path.abspath(
            os.path.join(self.wf_config.results_dir, self.state.workflow_id)
        )
        os.makedirs(self._results_dir, exist_ok=True)

        # Honour workflow-config max iterations
        self.state.max_iterations = self.wf_config.max_review_iterations

    # -- lazy agent accessors -----------------------------------------------

    @property
    def planner(self):
        if self._planner is None:
            self._planner = self._agent_factory.planner()
        return self._planner

    @property
    def analyst(self):
        if self._analyst is None:
            self._analyst = self._agent_factory.analyst()
        return self._analyst

    @property
    def coder(self):
        if self._coder is None:
            self._coder = self._agent_factory.coder()
        return self._coder

    @property
    def reviewer(self):
        if self._reviewer is None:
            self._reviewer = self._agent_factory.reviewer()
        return self._reviewer

    @property
    def summarizer(self):
        if self._summarizer is None:
            self._summarizer = self._agent_factory.summarizer()
        return self._summarizer

    # -- internal helpers ---------------------------------------------------

    def _make_crew(self, agents: list, tasks: list[Task]) -> Crew:
        """Create a Crew instance with CrewAI native memory (SQLite LTM)."""
        crew_kwargs = {
            "agents": agents,
            "tasks": tasks,
            "process": Process.sequential,
            "verbose": self.wf_config.verbose,
        }
        if self.storage_config.enabled:
            try:
                from crewai.memory import LongTermMemory
                from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage

                crew_kwargs["memory"] = True
                crew_kwargs["long_term_memory"] = LongTermMemory(
                    storage=LTMSQLiteStorage(db_path=self.storage_config.db_path)
                )
                embedder = get_crewai_embedder_dict()
                if embedder:
                    crew_kwargs["embedder"] = embedder
            except Exception as exc:
                print(f"   >> Memory init skipped: {exc}")
        return Crew(**crew_kwargs)

    def _apply_token_budget(self, task: Task, agents: list) -> None:
        """Set per-task max_tokens based on prompt length and context window."""
        llm_cfg = get_llm_config()
        prompt = f"{task.description}\n\nExpected Output:\n{task.expected_output}"
        budget = build_budget(
            text=prompt,
            model=llm_cfg.model,
            context_window=llm_cfg.context_window,
            default_budget=llm_cfg.output_budget,
            safety_margin=llm_cfg.safety_margin,
        )
        for agent in agents:
            if getattr(agent, "llm", None) is not None:
                agent.llm.max_tokens = budget.max_output_tokens
        if budget.max_output_tokens < llm_cfg.output_budget:
            print(
                "   >> Token budget capped: "
                f"prompt={budget.prompt_tokens}, "
                f"max_output={budget.max_output_tokens}, "
                f"context_window={llm_cfg.context_window}"
            )

    def _summarize_if_needed(self, text: str, label: str) -> str:
        """Summarize long text to keep prompts within context windows."""
        if not text:
            return text
        llm_cfg = get_llm_config()
        token_count = estimate_tokens(text, model=llm_cfg.model)
        if token_count <= llm_cfg.summary_trigger_tokens:
            return text

        max_prompt_tokens = max(
            256,
            llm_cfg.context_window - llm_cfg.safety_margin - llm_cfg.output_budget,
        )
        if token_count > max_prompt_tokens:
            text = truncate_text(text, max_prompt_tokens, model=llm_cfg.model)

        tmpl = get_task_template("summarization_task")
        tmpl["description"] = tmpl["description"].format(
            label=label,
            target_tokens=llm_cfg.summary_target_tokens,
            content=text,
        )
        task = Task(
            description=tmpl["description"],
            expected_output=tmpl["expected_output"],
            agent=self.summarizer,
        )
        self._apply_token_budget(task, [self.summarizer])
        crew = self._make_crew([self.summarizer], [task])
        result = crew.kickoff()
        return self._get_result_text(result)

    def _get_result_text(self, result: Any) -> str:
        """Extract text from CrewAI result objects safely."""
        if result is None:
            return ""
        if hasattr(result, "raw"):
            return getattr(result, "raw") or ""
        return str(result)

    def _retrieve_memory(self, label: str, query: str) -> str:
        return ""

    def _store_memory(self, kind: str, text: str) -> None:
        return

    # ======================================================================
    # Phase 0 -- classify
    # ======================================================================

    @start()
    def classify_task(self) -> Dict[str, Any]:
        """Decide simple vs complex based on the task_complexity field."""
        print(f"\n>>> Starting workflow {self.state.workflow_id}")
        print(f"   Question : {self.state.research_question}")
        print(f"   Complexity: {self.state.task_complexity}")
        self.state.status = "classifying"
        return {
            "research_question": self.state.research_question,
        }

    @router(classify_task)
    def route_by_complexity(self) -> str:
        c = self.state.task_complexity
        if c < 0:
            # Unknown complexity: default to simple path for short, direct
            # questions; complex for longer research-style questions.
            q = self.state.research_question.lower()
            simple_keywords = [
                "plot", "draw", "chart", "graph", "sine", "sin ",
                "cos ", "cosine", "hello", "print", "calculate",
                "generate a", "create a", "show", "visuali",
            ]
            if any(kw in q for kw in simple_keywords) or len(q.split()) <= 12:
                c = 1
            else:
                c = 5
            self.state.task_complexity = c
            print(f"   >> Auto-detected complexity = {c}")
        if 0 <= c <= COMPLEXITY_THRESHOLD:
            print(f"   >> SIMPLE path (complexity {c} <= {COMPLEXITY_THRESHOLD})")
            return "simple_path"
        print(f"   >> COMPLEX path (complexity {c})")
        return "complex_path"

    # ======================================================================
    # Complex path: Plan -> Analyse -> then join into coding
    # ======================================================================

    @listen("complex_path")
    def planning_phase(self) -> Dict[str, Any]:
        """Phase: create analysis plan."""
        print("\n--- Planning Analysis...")
        self.state.status = "planning"

        memory_context = self._retrieve_memory(
            "planning",
            f"{self.state.research_question}",
        )

        tmpl = get_task_template("planning_task")
        tmpl["description"] = tmpl["description"].format(
            research_question=self.state.research_question,
            memory_context=memory_context,
        )
        task = Task(
            description=tmpl["description"],
            expected_output=tmpl["expected_output"],
            agent=self.planner,
        )

        self._apply_token_budget(task, [self.planner])
        crew = self._make_crew([self.planner], [task])
        result = crew.kickoff()
        self.state.analysis_plan = self._get_result_text(result)
        return {"analysis_plan": self.state.analysis_plan}

    @listen(planning_phase)
    def analysis_phase(self) -> Dict[str, Any]:
        """Phase: design statistical approach."""
        print("\n--- Designing Statistical Analysis...")
        self.state.status = "analyzing"

        memory_context = self._retrieve_memory(
            "analysis",
            f"{self.state.research_question}",
        )

        tmpl = get_task_template("analysis_task")
        tmpl["description"] = tmpl["description"].format(
            analysis_plan=self.state.analysis_plan or "",
            memory_context=memory_context,
        )
        task = Task(
            description=tmpl["description"],
            expected_output=tmpl["expected_output"],
            agent=self.analyst,
        )

        self._apply_token_budget(task, [self.analyst])
        crew = self._make_crew([self.analyst], [task])
        result = crew.kickoff()
        self.state.statistical_approach = self._get_result_text(result)
        return {"statistical_approach": self.state.statistical_approach}

    # ======================================================================
    # Coding phase (shared by both paths)
    # ======================================================================

    @listen("simple_path")
    def coding_phase_simple(self) -> Dict[str, Any]:
        """Entry for the simple path -- goes directly to code generation."""
        return self._coding_phase()

    @listen(analysis_phase)
    def coding_phase_complex(self) -> Dict[str, Any]:
        """Entry for the complex path -- coding after analysis."""
        return self._coding_phase()

    def _coding_phase(self) -> Dict[str, Any]:
        """Generate Python code."""
        print("\n--- Generating Python Code...")
        self.state.status = "coding"

        # Build context from whatever is available (summarize if too large)
        context_parts: list[str] = []
        if self.state.analysis_plan:
            plan = self._summarize_if_needed(self.state.analysis_plan, "analysis plan")
            context_parts.append(f"Analysis Plan:\n{plan}")
        if self.state.statistical_approach:
            methods = self._summarize_if_needed(
                self.state.statistical_approach, "statistical methods"
            )
            context_parts.append(f"Statistical Methods:\n{methods}")

        memory_context = self._retrieve_memory(
            "coding",
            f"{self.state.research_question}",
        )
        if memory_context:
            context_parts.append(memory_context.strip())
        context = "\n\n".join(context_parts) or "(no additional context)"

        # If this is a revision, include the review feedback
        revision_hint = ""
        if self.state.iterations > 0 and self.state.review_report:
            review_text = self._summarize_if_needed(
                self.state.review_report, "review feedback"
            )
            exec_text = self._summarize_if_needed(
                (self.state.execution_stdout or self.state.execution_stderr),
                "execution output",
            )
            revision_hint = (
                f"\n\n--- REVISION ROUND {self.state.iterations} ---\n"
                f"Previous code had these issues:\n{review_text}\n"
                f"Previous execution output:\n{exec_text}\n"
                "Fix ALL issues listed above.\n"
            )

        pre_install = ", ".join(self.exec_config.pre_install)

        tmpl = get_task_template("coding_task")
        tmpl["description"] = tmpl["description"].format(
            research_question=self.state.research_question,
            context=context,
            revision_hint=revision_hint,
            pre_install=pre_install,
        )
        task = Task(
            description=tmpl["description"],
            expected_output=tmpl["expected_output"],
            agent=self.coder,
        )

        self._apply_token_budget(task, [self.coder])
        crew = self._make_crew([self.coder], [task])
        result = crew.kickoff()
        raw_code = _extract_code_block(self._get_result_text(result))
        self.state.generated_code = _sanitize_code(raw_code)
        self._save_code()
        return {"generated_code": self.state.generated_code}

    # ======================================================================
    # Execution phase
    # ======================================================================

    @listen(coding_phase_simple)
    def execution_phase_simple(self) -> Dict[str, Any]:
        return self._execution_phase()

    @listen(coding_phase_complex)
    def execution_phase_complex(self) -> Dict[str, Any]:
        return self._execution_phase()

    def _execution_phase(self) -> Dict[str, Any]:
        """Run the generated code inside a Docker sandbox.

        Strategy — direct-first:
        1. Run the code **directly** in Docker (reliable, no LLM involved).
        2. Record stdout/stderr and discover artifacts.
        3. If the code failed (non-empty stderr with errors), the review
           phase will request a revision — no need for an LLM executor.
        """
        print("\n--- Executing Code in Docker Sandbox...")
        self.state.status = "executing"

        code_text = _sanitize_code(self.state.generated_code or "")
        # Persist the sanitized version back so review/save use the fixed code
        self.state.generated_code = code_text
        if not code_text.strip():
            self.state.execution_stderr = "No code to execute."
            return {
                "execution_stdout": "",
                "execution_stderr": self.state.execution_stderr,
            }

        os.makedirs(self._results_dir, exist_ok=True)

        print(f"   >> Running code directly in Docker ({self.exec_config.image})...")
        stdout, stderr = _run_code_in_docker_direct(
            code=code_text,
            image=self.exec_config.image,
            results_dir=self._results_dir,
            libraries=self.exec_config.pre_install,
            timeout=self.wf_config.max_retries * 120,
        )

        # Classify output
        if stderr and ("error" in stderr.lower() or "traceback" in stderr.lower()):
            self.state.execution_stderr = stderr
            self.state.execution_stdout = stdout
        else:
            # Some programs write informational messages to stderr; combine
            combined = stdout
            if stderr:
                combined = f"{stdout}\n--- stderr ---\n{stderr}" if stdout else stderr
            self.state.execution_stdout = combined
            self.state.execution_stderr = ""

        if stdout:
            print(f"   >> stdout ({len(stdout)} chars)")
        if stderr:
            print(f"   >> stderr ({len(stderr)} chars)")

        # Discover any generated artifact files
        self.state.execution_artifacts = _discover_artifacts(self._results_dir)

        if self.state.execution_artifacts:
            print(f"   >> Found {len(self.state.execution_artifacts)} artifact(s):")
            for a in self.state.execution_artifacts:
                print(f"      - {a}")
        else:
            print("   >> No artifacts found in results directory")

        return {
            "execution_stdout": self.state.execution_stdout,
            "execution_stderr": self.state.execution_stderr,
        }

    # ======================================================================
    # Review phase
    # ======================================================================

    @listen(execution_phase_simple)
    def review_phase_simple(self) -> Dict[str, Any]:
        return self._review_phase()

    @listen(execution_phase_complex)
    def review_phase_complex(self) -> Dict[str, Any]:
        return self._review_phase()

    def _review_phase(self) -> Dict[str, Any]:
        """Review the code AND its execution output."""
        print(f"\n--- Reviewing Code (round {self.state.iterations + 1}/{self.state.max_iterations})...")
        self.state.status = "reviewing"

        exec_summary = self.state.execution_stdout or self.state.execution_stderr or "(no output)"
        exec_summary = self._summarize_if_needed(exec_summary, "execution output")

        # Build artifact summary for the reviewer
        artifact_names = [os.path.basename(a) for a in self.state.execution_artifacts]
        if artifact_names:
            artifact_info = (
                f"\n--- GENERATED FILES ---\n"
                f"The code produced these files: {', '.join(artifact_names)}\n"
            )
        else:
            artifact_info = (
                "\n--- GENERATED FILES ---\n"
                "No output files (images, etc.) were produced.\n"
            )

        memory_context = self._retrieve_memory(
            "review",
            f"{self.state.research_question}",
        )

        tmpl = get_task_template("review_task")
        tmpl["description"] = tmpl["description"].format(
            research_question=self.state.research_question,
            generated_code=self.state.generated_code or "",
            exec_summary=exec_summary,
            artifact_info=artifact_info,
            memory_context=memory_context,
        )
        task = Task(
            description=tmpl["description"],
            expected_output=tmpl["expected_output"],
            agent=self.reviewer,
        )

        self._apply_token_budget(task, [self.reviewer])
        crew = self._make_crew([self.reviewer], [task])
        result = crew.kickoff()
        self.state.review_report = self._get_result_text(result)

        # Parse verdict
        upper = (self.state.review_report or "").upper()
        self.state.approved = "APPROVED" in upper and "NEEDS REVISION" not in upper
        self.state.iterations += 1

        return {"review_report": self.state.review_report}

    # ======================================================================
    # Revision router
    # ======================================================================

    @router(review_phase_simple)
    def check_approval_simple(self) -> str:
        return self._check_approval()

    @router(review_phase_complex)
    def check_approval_complex(self) -> str:
        return self._check_approval()

    def _check_approval(self) -> str:
        if self.state.approved:
            print("   >> Code APPROVED")
            return "done"
        if self.state.iterations >= self.state.max_iterations:
            print(f"   >> Max iterations ({self.state.max_iterations}) reached -- accepting as-is")
            return "done"
        print(f"   >> NEEDS REVISION (round {self.state.iterations}/{self.state.max_iterations})")
        return "needs_revision"

    # ======================================================================
    # Revision loop
    # ======================================================================

    @listen("needs_revision")
    def revise_code(self) -> Dict[str, Any]:
        """Re-generate code incorporating reviewer feedback."""
        return self._coding_phase()

    @listen(revise_code)
    def re_execute(self) -> Dict[str, Any]:
        return self._execution_phase()

    @listen(re_execute)
    def re_review(self) -> Dict[str, Any]:
        return self._review_phase()

    @router(re_review)
    def re_check_approval(self) -> str:
        return self._check_approval()

    # ======================================================================
    # Done
    # ======================================================================

    @listen("done")
    def finalize(self) -> Dict[str, Any]:
        """Mark workflow complete, persist files."""
        self.state.status = "completed"
        self._save_code()
        self._save_readme()

        self._store_memory("question", self.state.research_question)
        self._store_memory("analysis_plan", self.state.analysis_plan or "")
        self._store_memory("statistical_approach", self.state.statistical_approach or "")
        self._store_memory("generated_code", self.state.generated_code or "")
        if self.state.execution_stdout:
            self._store_memory("execution_stdout", self.state.execution_stdout)
        if self.state.execution_stderr:
            self._store_memory("execution_stderr", self.state.execution_stderr)
        if self.state.review_report:
            self._store_memory("review_report", self.state.review_report)

        print(f"\n>>> Workflow {self.state.workflow_id} completed!")
        print(f"   Iterations: {self.state.iterations}")
        print(f"   Approved  : {self.state.approved}")

        return {
            "workflow_id": self.state.workflow_id,
            "status": "completed",
            "iterations": self.state.iterations,
            "approved": self.state.approved,
            "code_file": self.get_code_path(),
        }

    # ======================================================================
    # File helpers
    # ======================================================================

    def _save_code(self):
        code_content = self.state.generated_code or ""

        code_path = self.get_code_path()
        os.makedirs(os.path.dirname(code_path), exist_ok=True)
        with open(code_path, "w") as f:
            f.write(code_content)
        print(f"   Code saved to: {code_path}")

        results_code_path = self.get_results_code_path()
        os.makedirs(os.path.dirname(results_code_path), exist_ok=True)
        with open(results_code_path, "w") as f:
            f.write(code_content)
        print(f"   Code saved to: {results_code_path}")

    def _save_readme(self):
        readme_path = self.get_readme_path()
        artifact_prefix = f"../results/{self.state.workflow_id}/"
        with open(readme_path, "w") as f:
            f.write(self.generate_readme(artifact_prefix=artifact_prefix))
        print(f"   README saved to: {readme_path}")

        results_readme_path = self.get_results_readme_path()
        os.makedirs(os.path.dirname(results_readme_path), exist_ok=True)
        with open(results_readme_path, "w") as f:
            f.write(self.generate_readme(artifact_prefix=""))
        print(f"   README saved to: {results_readme_path}")

    def get_code_path(self) -> str:
        return os.path.join(
            self.wf_config.output_dir,
            f"workflow_{self.state.workflow_id}.py",
        )

    def get_readme_path(self) -> str:
        return os.path.join(
            self.wf_config.output_dir,
            f"README_{self.state.workflow_id}.md",
        )

    def get_results_code_path(self) -> str:
        return os.path.join(
            self._results_dir,
            f"workflow_{self.state.workflow_id}.py",
        )

    def get_results_readme_path(self) -> str:
        return os.path.join(
            self._results_dir,
            "README.md",
        )

    def get_results_dir(self) -> str:
        return self._results_dir

    def generate_readme(self, artifact_prefix: str = "") -> str:
        exec_section = ""
        if self.state.execution_stdout:
            exec_section = (
                f"\n## Execution Output\n\n"
                f"```\n{self.state.execution_stdout}\n```\n"
            )
        artifacts_section = ""
        if self.state.execution_artifacts:
            def _artifact_name(path: str) -> str:
                return os.path.basename(path)

            lines = "\n".join(
                f"- `{artifact_prefix}{_artifact_name(a)}`" for a in self.state.execution_artifacts
            )

            preview_exts = {".png", ".jpg", ".jpeg", ".svg", ".gif"}
            previews = []
            for a in self.state.execution_artifacts:
                name = _artifact_name(a)
                if os.path.splitext(name)[1].lower() in preview_exts:
                    previews.append(f"![{name}]({artifact_prefix}{name})")

            preview_section = ""
            if previews:
                preview_section = "\n\n" + "\n\n".join(previews) + "\n"

            artifacts_section = f"\n## Generated Artifacts\n\n{lines}{preview_section}"

        return f"""# Workflow {self.state.workflow_id}

Generated: {self.state.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Review Iterations: {self.state.iterations}
Approved: {self.state.approved}

---

## Task Request

{self.state.research_question}

---

## Analysis Plan

{self.state.analysis_plan or 'N/A (simple path -- skipped)'}

---

## Statistical Approach

{self.state.statistical_approach or 'N/A (simple path -- skipped)'}
{exec_section}
{artifacts_section}
---

## Code Review

{self.state.review_report}

---

## Usage

```bash
pip install numpy pandas matplotlib scipy
python workflow_{self.state.workflow_id}.py
```

---

Generated by CrewAI Workflow System
"""


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_workflow(
    research_question: str,
    task_complexity: int = -1,
) -> Dict[str, Any]:
    """Run a complete workflow and return final state as a dict."""
    state = WorkflowState(
        research_question=research_question,
        task_complexity=task_complexity,
    )
    workflow = AstronomyWorkflow(state=state)
    workflow.kickoff()

    return {
        "workflow_id": state.workflow_id,
        "status": state.status,
        "iterations": state.iterations,
        "approved": state.approved,
        "analysis_plan": state.analysis_plan,
        "statistical_approach": state.statistical_approach,
        "generated_code": state.generated_code,
        "execution_stdout": state.execution_stdout,
        "execution_stderr": state.execution_stderr,
        "execution_artifacts": state.execution_artifacts,
        "review_report": state.review_report,
        "code_file": workflow.get_code_path(),
    }
