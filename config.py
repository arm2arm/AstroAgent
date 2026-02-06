"""
Configuration for CrewAI Astronomy Workflow System
"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


# Load .env once at import time so config reads updated values.
load_dotenv()


@dataclass
class LLMProfile:
    """A single LLM endpoint profile."""
    name: str
    base_url: str
    api_key: str
    model: str


@dataclass
class LLMConfig:
    """Active LLM endpoint configuration (OpenAI-compatible /v1)"""
    base_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    model: str = "qwen3-coder:latest"
    temperature: float = 0.3
    max_tokens: int = 4000
    timeout: int = 120


@dataclass
class ExecutionConfig:
    """Code-execution Docker environment configuration."""
    mode: str = "image"                          # "image" or "dockerfile"
    image: str = "python:3.12-slim"              # remote image (when mode=image)
    dockerfile_path: str = "./Dockerfile.executor"  # local Dockerfile (when mode=dockerfile)
    pre_install: list[str] = field(default_factory=lambda: [
        "numpy", "pandas", "matplotlib", "astropy", "scipy"
    ])


@dataclass
class WorkflowConfig:
    """Workflow execution configuration"""
    max_retries: int = 3
    max_review_iterations: int = 3
    output_dir: str = "outputs/workflows"
    results_dir: str = "outputs/results"
    verbose: bool = True


def load_llm_profiles() -> list[LLMProfile]:
    """Scan environment for LLM_N_* profiles and return them in order."""
    profiles: list[LLMProfile] = []
    for i in range(1, 20):
        name = os.getenv(f"LLM_{i}_NAME")
        if not name:
            break
        base_url = os.getenv(f"LLM_{i}_BASE_URL", "")
        api_key = os.getenv(f"LLM_{i}_API_KEY", "")
        model = os.getenv(f"LLM_{i}_MODEL", "")
        profiles.append(LLMProfile(name=name, base_url=base_url, api_key=api_key, model=model))
    return profiles


def get_llm_config() -> LLMConfig:
    """Load active LLM configuration from environment"""
    return LLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("LLM_API_KEY", ""),
        model=os.getenv("LLM_MODEL", "qwen3-coder:latest"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4000")),
        timeout=int(os.getenv("LLM_TIMEOUT", "120"))
    )


def get_execution_config() -> ExecutionConfig:
    """Load code-execution environment configuration."""
    pre_install_str = os.getenv(
        "EXEC_PRE_INSTALL",
        "numpy,pandas,matplotlib,astropy,scipy"
    )
    return ExecutionConfig(
        mode=os.getenv("EXEC_MODE", "image"),
        image=os.getenv("EXEC_IMAGE", "python:3.12-slim"),
        dockerfile_path=os.getenv("EXEC_DOCKERFILE", "./Dockerfile.executor"),
        pre_install=[p.strip() for p in pre_install_str.split(",") if p.strip()],
    )


def get_workflow_config() -> WorkflowConfig:
    """Load workflow configuration"""
    return WorkflowConfig(
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        max_review_iterations=int(os.getenv("MAX_REVIEW_ITERATIONS", "3")),
        output_dir=os.getenv("OUTPUT_DIR", "outputs/workflows"),
        results_dir=os.getenv("RESULTS_DIR", "outputs/results"),
        verbose=os.getenv("VERBOSE", "true").lower() == "true"
    )


def init_directories():
    """Create necessary directories"""
    config = get_workflow_config()
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
