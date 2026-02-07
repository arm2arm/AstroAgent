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
    context_window: int = 32768
    output_budget: int = 8192
    embed_model: str = ""          # embedding model name (e.g. "nomic-embed-text:latest")
    embed_provider: str = "ollama"  # embedding provider: "ollama", "openai", etc.
    embed_base_url: str = ""       # optional separate embedding endpoint base URL
    embed_api_key: str = ""        # optional separate embedding API key


@dataclass
class LLMConfig:
    """Active LLM endpoint configuration (OpenAI-compatible /v1)"""
    base_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    model: str = "qwen3-coder:latest"
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
    """Workflow execution configuration"""
    max_retries: int = 3
    max_review_iterations: int = 3
    output_dir: str = "outputs/workflows"
    results_dir: str = "outputs/results"
    verbose: bool = True


@dataclass
class MemoryConfig:
    """SQLite memory configuration for retrieval-augmented context."""
    enabled: bool = False
    db_path: str = ".crewai/memory_astroagent.db"
    index_paths: list[str] = field(default_factory=lambda: [
        "README.md",
        "QUICKSTART.md",
        "project.md",
        "example_tasks",
    ])
    chunk_tokens: int = 400
    top_k: int = 4
    force_reindex: bool = False


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
        context_window = int(os.getenv(f"LLM_{i}_CONTEXT_WINDOW", "32768"))
        output_budget = int(os.getenv(f"LLM_{i}_OUTPUT_BUDGET", "8192"))
        embed_model = os.getenv(f"LLM_{i}_EMBED_MODEL", "nomic-embed-text:latest")
        embed_provider = os.getenv(f"LLM_{i}_EMBED_PROVIDER", "ollama")
        embed_base_url = os.getenv(f"LLM_{i}_EMBED_BASE_URL", "")
        embed_api_key = os.getenv(f"LLM_{i}_EMBED_API_KEY", "")
        profiles.append(
            LLMProfile(
                name=name,
                base_url=base_url,
                api_key=api_key,
                model=model,
                context_window=context_window,
                output_budget=output_budget,
                embed_model=embed_model,
                embed_provider=embed_provider,
                embed_base_url=embed_base_url,
                embed_api_key=embed_api_key,
            )
        )
    return profiles


def get_llm_config() -> LLMConfig:
    """Load active LLM configuration from environment"""
    return LLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("LLM_API_KEY", ""),
        model=os.getenv("LLM_MODEL", "qwen3-coder:latest"),
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
    """Build the embedder config dict for CrewAI Crew(embedder=...).

    Returns None if no embed_model is configured (falls back to CrewAI default).
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
                "model_name": cfg.embed_model,
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
        # Generic provider — pass model_name + URL, let CrewAI handle it
        return {
            "provider": provider,
            "config": {
                "model_name": cfg.embed_model,
                "url": base,
            },
        }


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


def get_memory_config() -> MemoryConfig:
    """Load memory/RAG configuration."""
    enabled = os.getenv("MEMORY_ENABLED", "false").lower() in {"1", "true", "yes"}
    raw_paths = os.getenv(
        "MEMORY_INDEX_PATHS",
        "README.md,QUICKSTART.md,project.md,example_tasks",
    )
    index_paths = [p.strip() for p in raw_paths.split(",") if p.strip()]
    return MemoryConfig(
        enabled=enabled,
        db_path=os.getenv("MEMORY_DB_PATH", ".crewai/memory_astroagent.db"),
        index_paths=index_paths,
        chunk_tokens=int(os.getenv("MEMORY_CHUNK_TOKENS", "400")),
        top_k=int(os.getenv("MEMORY_TOP_K", "4")),
        force_reindex=os.getenv("MEMORY_FORCE_REINDEX", "false").lower() in {"1", "true", "yes"},
    )


def init_directories():
    """Create necessary directories"""
    config = get_workflow_config()
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
