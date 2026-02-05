"""
Configuration for CrewAI Astronomy Workflow System
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """LLM endpoint configuration"""
    base_url: str = "https://ai.aip.de/api"
    api_key: str = "aip-local-key"  # Your AIP API key
    model: str = "llama-3-70b"  # Available model at AIP
    temperature: float = 0.3
    max_tokens: int = 4000
    timeout: int = 120


@dataclass
class WorkflowConfig:
    """Workflow execution configuration"""
    max_retries: int = 3
    output_dir: str = "outputs/workflows"
    results_dir: str = "outputs/results"
    verbose: bool = True


def get_llm_config() -> LLMConfig:
    """Load LLM configuration from environment"""
    return LLMConfig(
        base_url=os.getenv("AIP_LLM_ENDPOINT", "https://ai.aip.de/api"),
        api_key=os.getenv("AIP_API_KEY", "your-key-here"),
        model=os.getenv("AIP_MODEL", "llama-3-70b"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.3"))
    )


def get_workflow_config() -> WorkflowConfig:
    """Load workflow configuration"""
    return WorkflowConfig(
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        output_dir=os.getenv("OUTPUT_DIR", "outputs/workflows"),
        results_dir=os.getenv("RESULTS_DIR", "outputs/results"),
        verbose=os.getenv("VERBOSE", "true").lower() == "true"
    )


def init_directories():
    """Create necessary directories"""
    config = get_workflow_config()
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
