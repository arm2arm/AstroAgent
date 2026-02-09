"""
Configuration for CrewAI 1.9x Workflow System

Simplified single-LLM config loaded from ``.env``.  Agent prompts live in
``config/agents.yaml`` and ``config/tasks.yaml`` — see ``agents.py``.
"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


_DOTENV_LOADED = False


def _load_dotenv_once() -> None:
    """Load .env once per process to avoid repeated filesystem work."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    load_dotenv()
    _DOTENV_LOADED = True


@dataclass
class LLMConfig:
    """Active LLM endpoint configuration (OpenAI-compatible /v1)."""
    base_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    model: str = "qwen3-coder:latest"
    provider: str = "openai"           # CrewAI native provider name
    temperature: float = 0.3
    max_tokens: int = 4000
    timeout: int = 120
    context_window: int = 32768
    output_budget: int = 8192
    safety_margin: int = 512
    summary_trigger_tokens: int = 2000
    summary_target_tokens: int = 600
    embed_model: str = "nomic-embed-text:latest"
    embed_provider: str = "ollama"
    embed_base_url: str = ""
    embed_api_key: str = ""


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
    """Workflow execution configuration."""
    max_retries: int = 3
    max_review_iterations: int = 3
    output_dir: str = "outputs/workflows"
    results_dir: str = "outputs/results"
    verbose: bool = True


@dataclass
class StorageConfig:
    """CrewAI memory / storage configuration (SQLite-backed)."""
    enabled: bool = True
    db_path: str = "storage/memory.db"


def get_llm_config() -> LLMConfig:
    """Load active LLM configuration from environment."""
    _load_dotenv_once()
    return LLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("LLM_API_KEY", ""),
        model=os.getenv("LLM_MODEL", "qwen3-coder:latest"),
        provider=os.getenv("LLM_PROVIDER", "openai"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4000")),
        timeout=int(os.getenv("LLM_TIMEOUT", "120")),
        context_window=int(os.getenv("LLM_CONTEXT_WINDOW", "32768")),
        output_budget=int(os.getenv("LLM_OUTPUT_BUDGET", "8192")),
        safety_margin=int(os.getenv("LLM_SAFETY_MARGIN", "512")),
        summary_trigger_tokens=int(os.getenv("LLM_SUMMARY_TRIGGER_TOKENS", "2000")),
        summary_target_tokens=int(os.getenv("LLM_SUMMARY_TARGET_TOKENS", "600")),
        embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text:latest"),
        embed_provider=os.getenv("EMBED_PROVIDER", "ollama"),
        embed_base_url=os.getenv("EMBED_BASE_URL", ""),
        embed_api_key=os.getenv("EMBED_API_KEY", ""),
    )


def get_crewai_embedder_dict() -> dict | None:
    """Build the embedder config dict for ``Crew(embedder=...)``.

    Returns ``None`` if no embed_model is configured (falls back to CrewAI default).
    """
    cfg = get_llm_config()
    if not cfg.embed_model:
        return None

    embed_base_url = cfg.embed_base_url or cfg.base_url
    # Derive the base Ollama URL by stripping /v1 suffix
    base = embed_base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]

    provider = cfg.embed_provider.lower()
    if provider == "ollama":
        return {
            "provider": "ollama",
            "config": {
                "model": cfg.embed_model,
                "url": f"{base}/api/embeddings",
            },
        }
    elif provider == "openai":
        return {
            "provider": "openai",
            "config": {
                "model_name": cfg.embed_model,
                "api_key": cfg.embed_api_key or cfg.api_key or os.getenv("OPENAI_API_KEY", ""),
                "api_base": embed_base_url,
            },
        }
    else:
        # Generic provider — pass model + URL, let CrewAI handle it
        return {
            "provider": provider,
            "config": {
                "model": cfg.embed_model,
                "url": base,
            },
        }


def get_execution_config() -> ExecutionConfig:
    """Load code-execution environment configuration."""
    _load_dotenv_once()
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
    """Load workflow configuration."""
    _load_dotenv_once()
    return WorkflowConfig(
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        max_review_iterations=int(os.getenv("MAX_REVIEW_ITERATIONS", "3")),
        output_dir=os.getenv("OUTPUT_DIR", "outputs/workflows"),
        results_dir=os.getenv("RESULTS_DIR", "outputs/results"),
        verbose=os.getenv("VERBOSE", "true").lower() == "true"
    )


def get_storage_config() -> StorageConfig:
    """Load CrewAI memory/storage configuration."""
    _load_dotenv_once()
    enabled = os.getenv("MEMORY_ENABLED", "true").lower() in {"1", "true", "yes"}
    return StorageConfig(
        enabled=enabled,
        db_path=os.getenv("MEMORY_DB_PATH", "storage/memory.db"),
    )


def init_directories() -> None:
    """Create necessary directories."""
    _load_dotenv_once()
    config = get_workflow_config()
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    # Ensure storage directory exists
    storage = get_storage_config()
    if storage.enabled:
        os.makedirs(os.path.dirname(storage.db_path) or ".", exist_ok=True)
