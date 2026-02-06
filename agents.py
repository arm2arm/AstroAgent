"""
CrewAI Agent Definitions for Astronomy Workflows

Agents
------
Planner   – designs the analysis strategy  (complex-path only)
Analyst   – designs statistical methods     (complex-path only)
Coder     – generates Python code           (all paths)
Executor  – runs code in a Docker sandbox   (all paths)
Reviewer  – validates code + results        (all paths)
"""
from crewai import Agent, LLM
from crewai_tools import CodeInterpreterTool

from config import get_llm_config, get_execution_config


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def create_llm(temperature: float = 0.3) -> LLM:
    """Create LLM instance for an OpenAI-compatible endpoint."""
    config = get_llm_config()
    return LLM(
        model=f"openai/{config.model}",
        base_url=config.base_url,
        api_key=config.api_key or "no-key-required",
        temperature=temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )


# ---------------------------------------------------------------------------
# Code-execution tool (configured from .env)
# ---------------------------------------------------------------------------

def create_code_interpreter_tool() -> CodeInterpreterTool:
    """Build a CodeInterpreterTool using .env execution settings."""
    exec_cfg = get_execution_config()

    if exec_cfg.mode == "dockerfile":
        # Build from a local Dockerfile directory. The tool builds on
        # first use and caches the image as ``default_image_tag``.
        return CodeInterpreterTool(
            user_dockerfile_path=exec_cfg.dockerfile_path,
            default_image_tag="astroagent-exec:latest",
        )

    # mode == "image" — use a remote / pre-pulled image directly.
    return CodeInterpreterTool(
        default_image_tag=exec_cfg.image,
    )


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------

def create_planner_agent() -> Agent:
    """Workflow planning agent — designs the analysis strategy."""
    return Agent(
        role="Astronomy Workflow Planner",
        goal="Design optimal analysis workflows for astronomical data",
        backstory=(
            "You are an experienced observational astronomer with expertise "
            "in Gaia data analysis. You excel at breaking down research "
            "questions into clear, executable analysis steps. You understand "
            "data selection, filtering, and analysis strategies for large "
            "astronomical datasets."
        ),
        llm=create_llm(temperature=0.4),
        verbose=True,
        allow_delegation=False,
    )


def create_analyst_agent() -> Agent:
    """Statistical analysis agent — designs methods and approach."""
    return Agent(
        role="Data Analysis Specialist",
        goal="Design rigorous statistical analysis methods",
        backstory=(
            "You are a data scientist specializing in astronomical data. "
            "You have deep knowledge of statistical methods, data quality "
            "assessment, and visualization techniques. You ensure analyses "
            "are scientifically sound and computationally efficient."
        ),
        llm=create_llm(temperature=0.3),
        verbose=True,
        allow_delegation=False,
    )


def create_coder_agent() -> Agent:
    """Code generation agent — produces self-contained Python scripts."""
    return Agent(
        role="Scientific Programmer",
        goal=(
            "Generate clean, self-contained, executable Python code for "
            "astronomy analysis. The code MUST print its key results to "
            "stdout and save any plots to files."
        ),
        backstory=(
            "You are an expert Python programmer specializing in scientific "
            "computing. You write clean, well-documented code using astropy, "
            "pandas, matplotlib, and numpy. You always include error "
            "handling, logging, and follow PEP 8 standards. Your code is "
            "designed to run headless — plots are saved to files with "
            "matplotlib.pyplot.savefig(), never shown interactively."
        ),
        llm=create_llm(temperature=0.2),
        verbose=True,
        allow_delegation=False,
    )


def create_executor_agent() -> Agent:
    """Code execution agent — runs code in a Docker sandbox."""
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
        llm=create_llm(temperature=0.0),
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
            "You are a meticulous code reviewer with expertise in "
            "scientific Python. You verify correctness, identify bugs, "
            "check for edge cases, and ensure code follows best practices. "
            "You also inspect execution output for errors or unexpected "
            "results. Your review always ends with exactly one of these "
            "two verdicts on its own line:\n"
            "APPROVED — the code runs correctly and produces valid results.\n"
            "NEEDS REVISION — there are issues that must be fixed, listed "
            "as actionable bullet points."
        ),
        llm=create_llm(temperature=0.3),
        verbose=True,
        allow_delegation=False,
    )
