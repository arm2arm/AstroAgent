"""
CrewAI Workflow for Astronomy Data Analysis

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
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from crewai import Crew, Process, Task
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel, Field

from agents import (
    create_analyst_agent,
    create_coder_agent,
    create_executor_agent,
    create_planner_agent,
    create_reviewer_agent,
)
from config import get_execution_config, get_workflow_config


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class WorkflowState(BaseModel):
    """Shared state threaded through every phase."""

    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = Field(default_factory=datetime.now)

    # User inputs
    research_question: str = ""
    data_source: str = "gaia_dr3"
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


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class AstronomyWorkflow(Flow[WorkflowState]):
    """
    Orchestrates the full research pipeline.

    Simple tasks  (complexity <= 3): Coder -> Executor -> Reviewer (+ loop)
    Complex tasks (complexity > 3):  Planner -> Analyst -> Coder -> Executor -> Reviewer (+ loop)
    """

    def __init__(self, state: WorkflowState):
        # Set instance attributes BEFORE super().__init__() because Flow's
        # __init__ calls dir(self)/getattr which triggers @property accessors.
        self.wf_config = get_workflow_config()
        self.exec_config = get_execution_config()

        # Agents (created lazily via @property)
        self._planner = None
        self._analyst = None
        self._coder = None
        self._executor = None
        self._reviewer = None

        super().__init__(state=state)

        # Per-workflow results directory
        self._results_dir = os.path.join(
            self.wf_config.results_dir, self.state.workflow_id
        )
        os.makedirs(self._results_dir, exist_ok=True)

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

    # ======================================================================
    # Phase 0 -- classify
    # ======================================================================

    @start()
    def classify_task(self) -> Dict[str, Any]:
        """Decide simple vs complex based on the task_complexity field."""
        print(f"\n>>> Starting workflow {self.state.workflow_id}")
        print(f"   Question : {self.state.research_question}")
        print(f"   Source   : {self.state.data_source}")
        print(f"   Complexity: {self.state.task_complexity}")
        self.state.status = "classifying"
        return {
            "research_question": self.state.research_question,
            "data_source": self.state.data_source,
        }

    @router(classify_task)
    def route_by_complexity(self) -> str:
        c = self.state.task_complexity
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

        task = Task(
            description=(
                f'Create a detailed analysis plan for this research question:\n'
                f'"{self.state.research_question}"\n\n'
                f'Data source: {self.state.data_source}\n\n'
                'Your plan should include:\n'
                '1. Data selection criteria (which columns, filters)\n'
                '2. Sample size considerations\n'
                '3. Key analysis steps\n'
                '4. Expected outputs\n\n'
                'Be specific and practical.'
            ),
            expected_output="Detailed analysis plan with clear steps",
            agent=self.planner,
        )

        crew = Crew(
            agents=[self.planner],
            tasks=[task],
            process=Process.sequential,
            verbose=self.wf_config.verbose,
        )
        result = crew.kickoff()
        self.state.analysis_plan = result.raw
        return {"analysis_plan": self.state.analysis_plan}

    @listen(planning_phase)
    def analysis_phase(self) -> Dict[str, Any]:
        """Phase: design statistical approach."""
        print("\n--- Designing Statistical Analysis...")
        self.state.status = "analyzing"

        task = Task(
            description=(
                f'Design the statistical analysis approach for this plan:\n\n'
                f'{self.state.analysis_plan}\n\n'
                'Specify:\n'
                '1. Statistical methods to use\n'
                '2. Visualization strategies (plots, diagrams)\n'
                '3. Quality checks and validation\n'
                '4. Expected insights\n\n'
                'Focus on practical astronomy analysis.'
            ),
            expected_output="Statistical analysis strategy with methods",
            agent=self.analyst,
        )

        crew = Crew(
            agents=[self.analyst],
            tasks=[task],
            process=Process.sequential,
            verbose=self.wf_config.verbose,
        )
        result = crew.kickoff()
        self.state.statistical_approach = result.raw
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

        # Build context from whatever is available
        context_parts: list[str] = []
        if self.state.analysis_plan:
            context_parts.append(f"Analysis Plan:\n{self.state.analysis_plan}")
        if self.state.statistical_approach:
            context_parts.append(f"Statistical Methods:\n{self.state.statistical_approach}")
        context = "\n\n".join(context_parts) or "(no additional context)"

        # If this is a revision, include the review feedback
        revision_hint = ""
        if self.state.iterations > 0 and self.state.review_report:
            revision_hint = (
                f"\n\n--- REVISION ROUND {self.state.iterations} ---\n"
                f"Previous code had these issues:\n{self.state.review_report}\n"
                f"Previous execution output:\n{self.state.execution_stdout or self.state.execution_stderr}\n"
                "Fix ALL issues listed above.\n"
            )

        pre_install = ", ".join(self.exec_config.pre_install)

        task = Task(
            description=(
                f'Generate a complete, self-contained Python script for:\n'
                f'"{self.state.research_question}"\n\n'
                f'Data source: {self.state.data_source}\n\n'
                f'{context}'
                f'{revision_hint}\n\n'
                f'Requirements:\n'
                f'- Use ONLY these libraries (already installed): {pre_install}\n'
                f'- PRINT all key results to stdout\n'
                f'- Save plots to files with matplotlib.pyplot.savefig() '
                f'  (use Agg backend: matplotlib.use("Agg"))\n'
                f'- Include proper error handling\n'
                f'- Make it executable as a script (if __name__ == "__main__")\n\n'
                f'Return ONLY the Python code, nothing else.'
            ),
            expected_output="Complete executable Python script",
            agent=self.coder,
        )

        crew = Crew(
            agents=[self.coder],
            tasks=[task],
            process=Process.sequential,
            verbose=self.wf_config.verbose,
        )
        result = crew.kickoff()
        self.state.generated_code = _extract_code_block(result.raw)
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
        """Run the generated code inside a Docker sandbox."""
        print("\n--- Executing Code in Docker Sandbox...")
        self.state.status = "executing"

        pre_install = ", ".join(self.exec_config.pre_install)

        task = Task(
            description=(
                'Execute the following Python code using the Code Interpreter tool.\n'
                'Do NOT modify the code -- run it exactly as given.\n'
                'Report the COMPLETE stdout output and any errors.\n\n'
                f'Libraries already installed: {pre_install}\n\n'
                f'```python\n{self.state.generated_code}\n```'
            ),
            expected_output=(
                "Full stdout output from executing the code, "
                "or the complete error traceback if it failed."
            ),
            agent=self.executor,
        )

        crew = Crew(
            agents=[self.executor],
            tasks=[task],
            process=Process.sequential,
            verbose=self.wf_config.verbose,
        )
        result = crew.kickoff()

        output = result.raw or ""
        if "error" in output.lower() or "traceback" in output.lower():
            self.state.execution_stderr = output
            self.state.execution_stdout = ""
        else:
            self.state.execution_stdout = output
            self.state.execution_stderr = ""

        # Discover any generated artifact files
        self.state.execution_artifacts = _discover_artifacts(self._results_dir)

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

        task = Task(
            description=(
                f'Review this Python code AND its execution output.\n\n'
                f'--- CODE ---\n{self.state.generated_code}\n\n'
                f'--- EXECUTION OUTPUT ---\n{exec_summary}\n\n'
                'Check:\n'
                '1. Did the code execute without errors?\n'
                '2. Are the results scientifically reasonable?\n'
                '3. Code quality, style, error handling\n'
                '4. Are plots saved correctly?\n\n'
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

        crew = Crew(
            agents=[self.reviewer],
            tasks=[task],
            process=Process.sequential,
            verbose=self.wf_config.verbose,
        )
        result = crew.kickoff()
        self.state.review_report = result.raw

        # Parse verdict
        upper = (result.raw or "").upper()
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
        code_path = self.get_code_path()
        os.makedirs(os.path.dirname(code_path), exist_ok=True)
        with open(code_path, "w") as f:
            f.write(self.state.generated_code or "")
        print(f"   Code saved to: {code_path}")

    def _save_readme(self):
        readme_path = self.get_readme_path()
        with open(readme_path, "w") as f:
            f.write(self.generate_readme())
        print(f"   README saved to: {readme_path}")

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

    def get_results_dir(self) -> str:
        return self._results_dir

    def generate_readme(self) -> str:
        exec_section = ""
        if self.state.execution_stdout:
            exec_section = (
                f"\n## Execution Output\n\n"
                f"```\n{self.state.execution_stdout}\n```\n"
            )
        artifacts_section = ""
        if self.state.execution_artifacts:
            lines = "\n".join(f"- `{a}`" for a in self.state.execution_artifacts)
            artifacts_section = f"\n## Generated Artifacts\n\n{lines}\n"

        return f"""# Astronomy Workflow {self.state.workflow_id}

Generated: {self.state.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Data Source: {self.state.data_source}
Review Iterations: {self.state.iterations}
Approved: {self.state.approved}

---

## Research Question

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
pip install numpy pandas matplotlib astropy scipy
python workflow_{self.state.workflow_id}.py
```

---

Generated by AstroAgent -- CrewAI Astronomy Workflow System
"""


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_workflow(
    research_question: str,
    data_source: str = "gaia_dr3",
    task_complexity: int = -1,
) -> Dict[str, Any]:
    """Run a complete workflow and return final state as a dict."""
    state = WorkflowState(
        research_question=research_question,
        data_source=data_source,
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
