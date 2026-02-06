"""
CrewAI Agent Definitions for Astronomy Workflows
"""
from crewai import Agent, LLM
from config import get_llm_config


def create_llm(temperature: float = 0.3) -> LLM:
    """
    Create LLM instance for OpenAI-compatible endpoint
    """
    config = get_llm_config()
    return LLM(
        model=f"openai/{config.model}",
        base_url=config.base_url,
        api_key=config.api_key or "no-key-required",
        temperature=temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout
    )


def create_planner_agent() -> Agent:
    """
    Create workflow planning agent
    """
    return Agent(
        role="Astronomy Workflow Planner",
        goal="Design optimal analysis workflows for astronomical data",
        backstory="""You are an experienced observational astronomer with expertise
        in Gaia data analysis. You excel at breaking down research questions into
        clear, executable analysis steps. You understand data selection, filtering,
        and analysis strategies for large astronomical datasets.""",
        llm=create_llm(temperature=0.4),
        verbose=True,
        allow_delegation=False
    )


def create_analyst_agent() -> Agent:
    """
    Create statistical analysis agent
    """
    return Agent(
        role="Data Analysis Specialist",
        goal="Design rigorous statistical analysis methods",
        backstory="""You are a data scientist specializing in astronomical data.
        You have deep knowledge of statistical methods, data quality assessment,
        and visualization techniques. You ensure analyses are scientifically sound
        and computationally efficient.""",
        llm=create_llm(temperature=0.3),
        verbose=True,
        allow_delegation=False
    )


def create_coder_agent() -> Agent:
    """
    Create code generation agent
    """
    return Agent(
        role="Scientific Programmer",
        goal="Generate clean, efficient Python code for astronomy analysis",
        backstory="""You are an expert Python programmer specializing in scientific
        computing. You write clean, well-documented code using astropy, pandas,
        matplotlib, and numpy. You always include error handling, logging, and
        follow PEP 8 standards.""",
        llm=create_llm(temperature=0.2),  # Lower for code
        verbose=True,
        allow_delegation=False
    )


def create_reviewer_agent() -> Agent:
    """
    Create code review agent
    """
    return Agent(
        role="Code Quality Reviewer",
        goal="Ensure code correctness and best practices",
        backstory="""You are a meticulous code reviewer with expertise in scientific
        Python. You verify correctness, identify bugs, check for edge cases, and
        ensure code follows best practices. You provide constructive feedback.""",
        llm=create_llm(temperature=0.3),
        verbose=True,
        allow_delegation=False
    )
