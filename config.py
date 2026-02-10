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
    num_ctx: int = 0               # Ollama num_ctx (0 = use context_window)
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


def _resolve_profile_env(key: str, default: str = "") -> str:
    """Return the env value for *key*, with profile override.

    When ``LLM_PROFILE=N`` (N >= 1) is set, any ``LLM_N_<SUFFIX>`` variable
    takes precedence over the plain ``LLM_<SUFFIX>`` / ``EMBED_<SUFFIX>``.
    This lets users switch endpoints by changing a single number.
    """
    profile = os.getenv("LLM_PROFILE", "").strip()
    if profile and profile != "0":
        # Map plain keys to their profiled counterparts
        # LLM_BASE_URL   -> LLM_{N}_BASE_URL
        # EMBED_MODEL    -> LLM_{N}_EMBED_MODEL
        if key.startswith("LLM_"):
            suffix = key[4:]  # e.g. "BASE_URL", "MODEL"
            profiled = f"LLM_{profile}_{suffix}"
        elif key.startswith("EMBED_"):
            profiled = f"LLM_{profile}_{key}"  # e.g. LLM_1_EMBED_MODEL
        else:
            profiled = None

        if profiled:
            val = os.getenv(profiled, "").strip()
            if val:  # only override when the profile var is non-empty
                return val
    return os.getenv(key, default)


def get_active_profile_name() -> str:
    """Return the human-readable name of the active profile (or 'manual')."""
    _load_dotenv_once()
    profile = os.getenv("LLM_PROFILE", "").strip()
    if profile and profile != "0":
        name = os.getenv(f"LLM_{profile}_NAME", "").strip()
        return name or f"Profile {profile}"
    return "manual"


def list_profiles() -> list[dict[str, str]]:
    """Discover all LLM_N_* profiles defined in the environment."""
    _load_dotenv_once()
    profiles = []
    for n in range(1, 20):
        name = os.getenv(f"LLM_{n}_NAME", "").strip()
        if not name:
            model = os.getenv(f"LLM_{n}_MODEL", "").strip()
            if not model:
                continue
            name = f"Profile {n}"
        profiles.append({
            "id": str(n),
            "name": name,
            "model": os.getenv(f"LLM_{n}_MODEL", ""),
            "base_url": os.getenv(f"LLM_{n}_BASE_URL", ""),
        })
    return profiles


def get_llm_config() -> LLMConfig:
    """Load active LLM configuration from environment.

    When ``LLM_PROFILE=N`` is set, values from ``LLM_N_*`` override
    the plain ``LLM_*`` / ``EMBED_*`` keys.
    """
    _load_dotenv_once()
    return LLMConfig(
        base_url=_resolve_profile_env("LLM_BASE_URL", "http://localhost:11434/v1"),
        api_key=_resolve_profile_env("LLM_API_KEY", ""),
        model=_resolve_profile_env("LLM_MODEL", "qwen3-coder:latest"),
        provider=_resolve_profile_env("LLM_PROVIDER", "openai"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        max_tokens=int(_resolve_profile_env("LLM_MAX_TOKENS", "4000")),
        timeout=int(os.getenv("LLM_TIMEOUT", "120")),
        context_window=int(_resolve_profile_env("LLM_CONTEXT_WINDOW", "32768")),
        num_ctx=int(_resolve_profile_env("LLM_NUM_CTX", "0")),
        output_budget=int(_resolve_profile_env("LLM_OUTPUT_BUDGET", "8192")),
        safety_margin=int(os.getenv("LLM_SAFETY_MARGIN", "512")),
        summary_trigger_tokens=int(os.getenv("LLM_SUMMARY_TRIGGER_TOKENS", "2000")),
        summary_target_tokens=int(os.getenv("LLM_SUMMARY_TARGET_TOKENS", "600")),
        embed_model=_resolve_profile_env("EMBED_MODEL", "nomic-embed-text:latest"),
        embed_provider=_resolve_profile_env("EMBED_PROVIDER", "ollama"),
        embed_base_url=_resolve_profile_env("EMBED_BASE_URL", ""),
        embed_api_key=_resolve_profile_env("EMBED_API_KEY", ""),
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
