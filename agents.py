"""
CrewAI Agent Definitions for Multi-Agent Workflows

Agents
------
Planner   – designs the analysis strategy  (complex-path only)
Analyst   – designs statistical methods     (complex-path only)
Coder     – generates Python code           (all paths)
Executor  – runs code in a Docker sandbox   (all paths)
Reviewer  – validates code + results        (all paths)
"""
import os

from crewai import Agent, LLM
from crewai_tools import CodeInterpreterTool
from pydantic import BaseModel, Field

from config import get_llm_config, get_execution_config

try:
    from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO
except Exception:
    CONTEXT_WINDOW_USAGE_RATIO = 0.85


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def create_llm(
    temperature: float = 0.3,
    max_tokens_override: int | None = None,
    model_override: str | None = None,
) -> LLM:
    """Create LLM instance for an OpenAI-compatible endpoint.

    Uses the ``provider`` kwarg so CrewAI routes directly to its native
    provider class (e.g. ``OpenAICompletion``), bypassing model-name
    pattern validation and the LiteLLM fallback.
    """
    config = get_llm_config()
    model_name = model_override or config.model
    provider = config.provider or "openai"

    # Strip any leftover provider prefix – the provider kwarg handles routing.
    for pfx in ("openai/", "azure/", "ollama/"):
        if model_name.startswith(pfx):
            model_name = model_name[len(pfx):]
            break

    llm = LLM(
        model=model_name,
        provider=provider,
        base_url=config.base_url,
        api_key=config.api_key or "no-key-required",
        temperature=temperature,
        max_tokens=max_tokens_override if max_tokens_override is not None else config.max_tokens,
        timeout=config.timeout,
    )
    if config.context_window > 0:
        llm.context_window_size = int(config.context_window * CONTEXT_WINDOW_USAGE_RATIO)
    return llm


# ---------------------------------------------------------------------------
# Code-execution tool (configured from .env)
# ---------------------------------------------------------------------------

class _CodeInterpreterSchemaOptional(BaseModel):
    """Schema with optional libraries_used to avoid tool-call validation errors."""

    code: str = Field(
        ...,
        description=(
            "Python3 code used to be interpreted in the Docker container. "
            "ALWAYS PRINT the final result and the output of the code"
        ),
    )
    libraries_used: list[str] = Field(
        default_factory=list,
        description=(
            "List of libraries used in the code with proper installing names. "
            "Example: numpy,pandas,beautifulsoup4"
        ),
    )


class _AstroCodeInterpreterTool(CodeInterpreterTool):
    """Code Interpreter with fallback libraries list."""

    args_schema: type[BaseModel] = _CodeInterpreterSchemaOptional

    def __init__(self, default_libraries: list[str], **kwargs):
        super().__init__(**kwargs)
        self._default_libraries = list(default_libraries)

    def _run(self, **kwargs):
        if not kwargs.get("libraries_used"):
            kwargs["libraries_used"] = list(self._default_libraries)
        return super()._run(**kwargs)

def create_code_interpreter_tool() -> CodeInterpreterTool:
    """Build a CodeInterpreterTool using .env execution settings."""
    exec_cfg = get_execution_config()

    if exec_cfg.mode == "dockerfile":
        # Build from a local Dockerfile directory. The tool builds on
        # first use and caches the image as ``default_image_tag``.
        return _AstroCodeInterpreterTool(
            default_libraries=exec_cfg.pre_install,
            user_dockerfile_path=exec_cfg.dockerfile_path,
            default_image_tag="astroagent-exec:latest",
        )

    # mode == "image" — use a remote / pre-pulled image directly.
    return _AstroCodeInterpreterTool(
        default_libraries=exec_cfg.pre_install,
        default_image_tag=exec_cfg.image,
    )


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------

def create_planner_agent() -> Agent:
    """Workflow planning agent — designs the analysis strategy."""
    return Agent(
        role="Workflow Planner",
        goal="Design optimal analysis workflows for the given task",
        backstory=(
            "You are an experienced technical planner. You excel at breaking "
            "down requests into clear, executable analysis steps. You "
            "understand data selection, filtering, and analysis strategies "
            "for a wide range of datasets. Ignore project.md and focus on the "
            "user's request."
        ),
        llm=create_llm(temperature=0.4),
        memory=True,
        respect_context_window=True,
        verbose=True,
        allow_delegation=False,
    )


def create_analyst_agent() -> Agent:
    """Statistical analysis agent — designs methods and approach."""
    return Agent(
        role="Data Analysis Specialist",
        goal="Design rigorous statistical analysis methods",
        backstory=(
            "You are a data scientist with deep knowledge of statistical "
            "methods, data quality assessment, and visualization techniques. "
            "You ensure analyses are scientifically sound and computationally "
            "efficient. Follow the user's request over any project spec."
        ),
        llm=create_llm(temperature=0.3),
        memory=True,
        respect_context_window=True,
        verbose=True,
        allow_delegation=False,
    )


def create_coder_agent() -> Agent:
    """Code generation agent — produces self-contained Python scripts."""
    return Agent(
        role="Scientific Programmer",
        goal=(
            "Generate clean, self-contained, executable Python code. "
            "The code MUST print its key results to stdout and save any "
            "plots to files using plt.savefig()."
        ),
        backstory=(
            "You are an expert Python programmer specializing in scientific "
            "computing and data visualization. You write clean, well-"
            "documented code using numpy, matplotlib, and pandas.\n"
            "Key rules:\n"
            "- Always set matplotlib.use('Agg') BEFORE importing pyplot\n"
            "- Save plots with plt.savefig('name.png', dpi=150, "
            "bbox_inches='tight')\n"
            "- Print 'SAVED: name.png' after saving each plot\n"
            "- Never call plt.show()\n"
            "- For simple tasks (plotting functions, basic calculations) "
            "generate data with numpy — do NOT fetch remote data\n"
            "- Return ONLY code inside a ```python fence\n"
            "- Ignore project.md and only follow the user's request"
        ),
        llm=create_llm(temperature=0.2),
        memory=True,
        respect_context_window=True,
        verbose=True,
        allow_delegation=False,
    )


def create_executor_agent() -> Agent:
    """Code execution agent — runs code in a Docker sandbox."""
    executor_model = os.getenv("EXECUTOR_LLM_MODEL", "").strip() or None
    return Agent(
        role="Code Executor",
        goal=(
            "Execute the provided Python code inside the Docker sandbox "
            "and return the full stdout output and any errors."
        ),
        backstory=(
            "You are a reliable execution environment operator. Your only "
            "job is to take the Python code you are given, run it using the "
            "Code Interpreter tool, and report the complete output. Do NOT "
            "modify the code — run it exactly as provided. If it fails, "
            "report the full error traceback."
        ),
        tools=[create_code_interpreter_tool()],
        llm=create_llm(temperature=0.0, model_override=executor_model),
        verbose=True,
        allow_delegation=False,
    )


def create_reviewer_agent() -> Agent:
    """Code quality reviewer — validates code and execution results."""
    return Agent(
        role="Code Quality Reviewer",
        goal=(
            "Review the generated code AND its execution results for "
            "correctness. End your review with a clear verdict: "
            "APPROVED or NEEDS REVISION."
        ),
        backstory=(
            "You are a meticulous code reviewer with expertise in Python. "
            "You verify correctness, identify bugs, check for edge cases, "
            "and ensure code follows best practices. You also inspect "
            "execution output for errors or unexpected results. Your review "
            "always ends with exactly one of these two verdicts on its own "
            "line:\n"
            "APPROVED — the code runs correctly and produces valid results.\n"
            "NEEDS REVISION — there are issues that must be fixed, listed "
            "as actionable bullet points."
        ),
        llm=create_llm(temperature=0.3),
        memory=True,
        respect_context_window=True,
        verbose=True,
        allow_delegation=False,
    )


def create_summarizer_agent() -> Agent:
    """Summarization agent to compress long context."""
    return Agent(
        role="Context Summarizer",
        goal="Condense long context into concise, actionable summaries.",
        backstory=(
            "You specialize in summarizing technical context for downstream agents. "
            "You preserve key requirements, constraints, and data details while "
            "removing repetition and fluff."
        ),
        llm=create_llm(temperature=0.2),
        memory=True,
        respect_context_window=True,
        verbose=True,
        allow_delegation=False,
    )
