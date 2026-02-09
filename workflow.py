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
import inspect

from agents import (
    create_analyst_agent,
    create_coder_agent,
    create_executor_agent,
    create_planner_agent,
    create_reviewer_agent,
    create_summarizer_agent,
)
from config import (
    get_execution_config,
    get_workflow_config,
    get_llm_config,
    get_memory_config,
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
    os.makedirs(results_dir, exist_ok=True)

    container_name = "code-interpreter-direct"
    # Remove stale container (ignore errors)
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Install libraries inside the container and run code
    install_cmds = ""
    if libraries:
        pkgs = " ".join(libraries)
        install_cmds = f"pip install --quiet {pkgs} 2>/dev/null; "

    shell_cmd = f'{install_cmds}python3 -c {_shell_quote(code)}'

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


def _shell_quote(s: str) -> str:
    """Shell-quote a string for safe embedding in sh -c '...'."""
    import shlex
    return shlex.quote(s)


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
        self.mem_config = get_memory_config()
        self.memory = None

        # Agents (created lazily via @property)
        self._planner = None
        self._analyst = None
        self._coder = None
        self._executor = None
        self._reviewer = None
        self._summarizer = None

        super().__init__(state=state)

        # Per-workflow results directory
        self._results_dir = os.path.join(
            self.wf_config.results_dir, self.state.workflow_id
        )
        os.makedirs(self._results_dir, exist_ok=True)

        # External RAG store removed for startup speed. Agent-local memory remains.

        # Honour workflow-config max iterations
        self.state.max_iterations = self.wf_config.max_review_iterations

    # -- lazy agent accessors -----------------------------------------------

    @property
    def planner(self):
        if self._planner is None:
            self._planner = create_planner_agent()
        return self._planner

    @property
    def analyst(self):
        if self._analyst is None:
            self._analyst = create_analyst_agent()
        return self._analyst

    @property
    def coder(self):
        if self._coder is None:
            self._coder = create_coder_agent()
        return self._coder

    @property
    def executor(self):
        if self._executor is None:
            self._executor = create_executor_agent()
        return self._executor

    @property
    def reviewer(self):
        if self._reviewer is None:
            self._reviewer = create_reviewer_agent()
        return self._reviewer

    @property
    def summarizer(self):
        if self._summarizer is None:
            self._summarizer = create_summarizer_agent()
        return self._summarizer

    # -- internal helpers ---------------------------------------------------

    def _make_crew(self, agents: list, tasks: list[Task]) -> Crew:
        """Create a Crew instance with optional memory support if available."""
        crew_kwargs = {
            "agents": agents,
            "tasks": tasks,
            "process": Process.sequential,
            "verbose": self.wf_config.verbose,
        }
        try:
            sig = inspect.signature(Crew)
            if "memory" in sig.parameters:
                crew_kwargs["memory"] = bool(self.mem_config.enabled)
            # Attach per-profile embedder when memory is enabled
            if self.mem_config.enabled and "embedder" in sig.parameters:
                from config import get_crewai_embedder_dict
                embedder = get_crewai_embedder_dict()
                if embedder:
                    crew_kwargs["embedder"] = embedder
        except Exception:
            pass
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

        task = Task(
            description=(
                f"Summarize the following {label} into <= {llm_cfg.summary_target_tokens} tokens. "
                "Preserve constraints, data details, and actionable steps. "
                "Output concise bullet points.\n\n"
                f"CONTENT:\n{text}"
            ),
            expected_output="Concise bullet summary.",
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

        task = Task(
            description=(
                f'Create a detailed plan for this task:\n'
                f'"{self.state.research_question}"\n\n'
                f'{memory_context}\n'
                'Your plan should include:\n'
                '1. Inputs and assumptions\n'
                '2. Data handling steps (if any)\n'
                '3. Key implementation steps\n'
                '4. Expected outputs\n\n'
                'Be specific and practical.'
            ),
            expected_output="Detailed analysis plan with clear steps",
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

        task = Task(
            description=(
                f'Design the analysis approach for this plan:\n\n'
                f'{self.state.analysis_plan}\n\n'
                f'{memory_context}\n'
                'Specify:\n'
                '1. Methods to use\n'
                '2. Visualization strategies (plots, diagrams)\n'
                '3. Quality checks and validation\n'
                '4. Expected outputs\n\n'
                'Focus on practical, executable steps.'
            ),
            expected_output="Statistical analysis strategy with methods",
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

        task = Task(
            description=(
                f'Generate a complete, self-contained Python script for:\n'
                f'"{self.state.research_question}"\n\n'
                f'{context}'
                f'{revision_hint}\n\n'
                f'CRITICAL requirements:\n'
                f'- Use ONLY these libraries (already installed): {pre_install}\n'
                f'- PRINT all key results to stdout\n'
                f'- For plots: use matplotlib with Agg backend:\n'
                f'    import matplotlib\n'
                f'    matplotlib.use("Agg")\n'
                f'    import matplotlib.pyplot as plt\n'
                f'- Save ALL plots to the CURRENT WORKING DIRECTORY, e.g.:\n'
                f'    plt.savefig("plot.png", dpi=150, bbox_inches="tight")\n'
                f'- After saving, print the filename: print("SAVED: plot.png")\n'
                f'- Include proper error handling\n'
                f'- The script must work standalone (no external data files)\n'
                f'- For simple tasks like plotting math functions, use numpy\n'
                f'  to generate data directly — do NOT fetch remote data.\n\n'
                f'Return ONLY the Python code inside a ```python code fence.'
            ),
            expected_output="Complete executable Python script",
            agent=self.coder,
        )

        self._apply_token_budget(task, [self.coder])
        crew = self._make_crew([self.coder], [task])
        result = crew.kickoff()
        self.state.generated_code = _extract_code_block(self._get_result_text(result))
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

        Strategy:
        1. Try via the CrewAI executor agent + CodeInterpreterTool.
        2. If the agent returns empty / hallucinated output (no real tool
           call), fall back to direct Docker execution.
        """
        print("\n--- Executing Code in Docker Sandbox...")
        self.state.status = "executing"

        code_text = self.state.generated_code or ""
        if not code_text.strip():
            self.state.execution_stderr = "No code to execute."
            return {
                "execution_stdout": "",
                "execution_stderr": self.state.execution_stderr,
            }

        # Clean up any leftover code-interpreter container from previous runs
        subprocess.run(
            ["docker", "rm", "-f", "code-interpreter"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        pre_install = ", ".join(self.exec_config.pre_install)
        escaped_code = code_text.replace('\\', '\\\\').replace('`', '\\`')

        task = Task(
            description=(
                'You have a single tool: Code Interpreter.\n'
                'Use it to execute the Python code below.\n\n'
                'INSTRUCTIONS:\n'
                '1. Call the code_interpreter tool with the EXACT code below '
                'as the "code" argument.\n'
                '2. For "libraries_used" pass: '
                f'[{chr(34) + (chr(34) + ", " + chr(34)).join(self.exec_config.pre_install) + chr(34)}]\n'
                '3. Do NOT modify, summarize or rewrite the code.\n'
                '4. Respond with a SINGLE tool call and nothing else.\n'
                '5. The tool input MUST be valid JSON with double-quoted keys.\n'
                '   Example format: {"code": "<CODE>", "libraries_used": ["numpy", "pandas"]}\n'
                '6. Return the COMPLETE output from the tool.\n\n'
                f'CODE TO EXECUTE:\n'
                f'{escaped_code}'
            ),
            expected_output=(
                "The complete stdout/stderr output from running the code."
            ),
            agent=self.executor,
        )

        self._apply_token_budget(task, [self.executor])
        crew = self._make_crew([self.executor], [task])

        # Change CWD to results dir so CodeInterpreterTool bind-mounts it
        # into the Docker container at /workspace — any files saved by the
        # script will appear here on the host.
        os.makedirs(self._results_dir, exist_ok=True)
        prev_cwd = os.getcwd()
        os.chdir(self._results_dir)
        try:
            result = crew.kickoff()
        finally:
            os.chdir(prev_cwd)

        output = self._get_result_text(result)

        # Detect whether the executor actually ran the tool.  If the output
        # is empty or doesn't look like real execution output, fall back to
        # running the code directly in Docker.
        artifacts_after_crew = _discover_artifacts(self._results_dir)
        tool_actually_ran = bool(
            artifacts_after_crew
            or "SAVED:" in output
            or "Traceback" in output
            or len(output.strip()) > 80
        )

        if not tool_actually_ran:
            print("   >> Executor did not call tool — falling back to direct Docker execution")
            stdout, stderr = _run_code_in_docker_direct(
                code=code_text,
                image=self.exec_config.image,
                results_dir=self._results_dir,
                libraries=self.exec_config.pre_install,
                timeout=self.wf_config.max_retries * 120,
            )
            output = stdout or stderr
            if stderr and "error" in stderr.lower():
                self.state.execution_stderr = stderr
                self.state.execution_stdout = stdout
            else:
                self.state.execution_stdout = output
                self.state.execution_stderr = ""
        else:
            if "error" in output.lower() or "traceback" in output.lower():
                self.state.execution_stderr = output
                self.state.execution_stdout = ""
            else:
                self.state.execution_stdout = output
                self.state.execution_stderr = ""

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

        memory_context = self._retrieve_memory(
            "review",
            f"{self.state.research_question}",
        )

        task = Task(
            description=(
                f'Review this Python code AND its execution output.\n\n'
                f'--- CODE ---\n{self.state.generated_code}\n\n'
                f'--- EXECUTION OUTPUT ---\n{exec_summary}\n\n'
                f'{memory_context}\n'
                'Check:\n'
                '1. Did the code execute without errors?\n'
                '2. Are the results reasonable and correct?\n'
                '3. Are plots saved to files (not shown interactively)?\n'
                '4. Does stdout contain meaningful output?\n\n'
                'IMPORTANT: If the code ran successfully and produced correct '
                'results, mark it APPROVED. Only reject if there are real '
                'errors or clearly wrong results.\n\n'
                'End your review with EXACTLY one of these verdicts on its own line:\n'
                'APPROVED\n'
                'NEEDS REVISION\n\n'
                'If NEEDS REVISION, list the specific issues as bullet points.'
            ),
            expected_output=(
                "Code review report ending with APPROVED or NEEDS REVISION"
            ),
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
