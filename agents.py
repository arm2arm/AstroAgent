"""
CrewAI 1.9x Agent & Task Factory — YAML-Configured

Loads agent definitions from ``config/agents.yaml`` and task templates from
``config/tasks.yaml``.  Agent roles, goals, and backstories are fully
configurable by editing the YAML files — no Python changes needed.

Agents
------
Planner   – designs the analysis strategy  (complex-path only)
Analyst   – designs statistical methods     (complex-path only)
Coder     – generates Python code           (all paths)
Executor  – runs code in a Docker sandbox   (all paths)
Reviewer  – validates code + results        (all paths)
Summarizer – compresses long context        (internal utility)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, LLM
from crewai_tools import CodeInterpreterTool
from pydantic import BaseModel, Field

from config import get_llm_config, get_execution_config

try:
    from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO
except Exception:
    CONTEXT_WINDOW_USAGE_RATIO = 0.85

# ---------------------------------------------------------------------------
# YAML config paths (relative to project root)
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).resolve().parent / "config"
_AGENTS_YAML = _CONFIG_DIR / "agents.yaml"
_TASKS_YAML = _CONFIG_DIR / "tasks.yaml"


# ---------------------------------------------------------------------------
# YAML loaders (cached per-process)
# ---------------------------------------------------------------------------
_agents_cache: dict[str, Any] | None = None
_tasks_cache: dict[str, Any] | None = None


def load_agents_config(force: bool = False) -> dict[str, Any]:
    """Load and cache ``config/agents.yaml``."""
    global _agents_cache
    if _agents_cache is None or force:
        with open(_AGENTS_YAML, "r", encoding="utf-8") as fh:
            _agents_cache = yaml.safe_load(fh) or {}
    return _agents_cache


def load_tasks_config(force: bool = False) -> dict[str, Any]:
    """Load and cache ``config/tasks.yaml``."""
    global _tasks_cache
    if _tasks_cache is None or force:
        with open(_TASKS_YAML, "r", encoding="utf-8") as fh:
            _tasks_cache = yaml.safe_load(fh) or {}
    return _tasks_cache


def reload_configs() -> None:
    """Force-reload both YAML configs (useful after editing files)."""
    load_agents_config(force=True)
    load_tasks_config(force=True)


# ---------------------------------------------------------------------------
# LLM builder
# ---------------------------------------------------------------------------

def build_llm(
    temperature: float = 0.3,
    max_tokens_override: int | None = None,
    model_override: str | None = None,
) -> LLM:
    """Create an LLM instance from .env configuration.

    Uses the ``provider`` kwarg so CrewAI routes directly to its native
    provider class (e.g. ``OpenAICompletion``), bypassing model-name
    pattern validation and the LiteLLM fallback.
    """
    cfg = get_llm_config()
    model_name = model_override or cfg.model
    provider = cfg.provider or "openai"

    # Strip leftover provider prefix — the provider kwarg handles routing.
    for pfx in ("openai/", "azure/", "ollama/"):
        if model_name.startswith(pfx):
            model_name = model_name[len(pfx):]
            break

    # Determine num_ctx for Ollama (fallback to context_window)
    num_ctx = cfg.num_ctx if cfg.num_ctx > 0 else cfg.context_window

    llm = LLM(
        model=model_name,
        provider=provider,
        base_url=cfg.base_url,
        api_key=cfg.api_key or "no-key-required",
        temperature=temperature,
        max_tokens=max_tokens_override if max_tokens_override is not None else cfg.max_tokens,
        timeout=cfg.timeout,
        extra_body={"num_ctx": num_ctx} if num_ctx > 0 else None,
    )
    if cfg.context_window > 0:
        llm.context_window_size = int(cfg.context_window * CONTEXT_WINDOW_USAGE_RATIO)
    return llm


# ---------------------------------------------------------------------------
# Docker code-execution tool
# ---------------------------------------------------------------------------

class _CodeInterpreterSchemaOptional(BaseModel):
    """Schema with optional ``libraries_used`` to avoid tool-call validation errors."""

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
    """Build a ``CodeInterpreterTool`` using .env execution settings."""
    exec_cfg = get_execution_config()

    if exec_cfg.mode == "dockerfile":
        return _AstroCodeInterpreterTool(
            default_libraries=exec_cfg.pre_install,
            user_dockerfile_path=exec_cfg.dockerfile_path,
            default_image_tag="astroagent-exec:latest",
        )

    # mode == "image"
    return _AstroCodeInterpreterTool(
        default_libraries=exec_cfg.pre_install,
        default_image_tag=exec_cfg.image,
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

class AgentFactory:
    """Creates CrewAI Agent instances from ``config/agents.yaml``.

    Each factory method merges the YAML config (role/goal/backstory) with
    the runtime LLM and optional tools.  Edit the YAML to change prompts
    without touching Python code.
    """

    def __init__(self) -> None:
        self._cfg = load_agents_config()

    def _build_agent(
        self,
        key: str,
        *,
        temperature: float = 0.3,
        tools: list | None = None,
        model_override: str | None = None,
    ) -> Agent:
        """Generic agent builder from YAML config + runtime LLM."""
        agent_cfg = dict(self._cfg[key])  # shallow copy to avoid mutation
        llm = build_llm(temperature=temperature, model_override=model_override)
        return Agent(
            config=agent_cfg,
            llm=llm,
            tools=tools or [],
            memory=True,
            respect_context_window=True,
        )

    # -- public per-agent methods ------------------------------------------

    def planner(self) -> Agent:
        """Workflow planning agent — designs the analysis strategy."""
        return self._build_agent("planner", temperature=0.4)

    def analyst(self) -> Agent:
        """Statistical analysis agent — designs methods and approach."""
        return self._build_agent("analyst", temperature=0.3)

    def coder(self) -> Agent:
        """Code generation agent — expert Python programmer."""
        return self._build_agent("coder", temperature=0.2)

    def executor(self) -> Agent:
        """Code execution agent — runs code in a Docker sandbox."""
        executor_model = os.getenv("EXECUTOR_LLM_MODEL", "").strip() or None
        return self._build_agent(
            "executor",
            temperature=0.0,
            tools=[create_code_interpreter_tool()],
            model_override=executor_model,
        )

    def reviewer(self) -> Agent:
        """Code quality reviewer — validates code and results."""
        return self._build_agent("reviewer", temperature=0.3)

    def summarizer(self) -> Agent:
        """Summarization agent — compresses long context."""
        return self._build_agent("summarizer", temperature=0.2)


# ---------------------------------------------------------------------------
# Task template helper
# ---------------------------------------------------------------------------

def get_task_template(task_key: str) -> dict[str, str]:
    """Return a *copy* of a task template from ``config/tasks.yaml``.

    The caller should ``.format(**kwargs)`` the ``description`` field to
    fill in runtime placeholders.
    """
    tasks_cfg = load_tasks_config()
    if task_key not in tasks_cfg:
        raise KeyError(f"Task template '{task_key}' not found in {_TASKS_YAML}")
    return {k: str(v) for k, v in tasks_cfg[task_key].items()}


# ---------------------------------------------------------------------------
# Backward-compatible factory functions (used by workflow.py)
# ---------------------------------------------------------------------------
_factory: AgentFactory | None = None


def _get_factory() -> AgentFactory:
    global _factory
    if _factory is None:
        _factory = AgentFactory()
    return _factory


def create_planner_agent() -> Agent:
    return _get_factory().planner()

def create_analyst_agent() -> Agent:
    return _get_factory().analyst()

def create_coder_agent() -> Agent:
    return _get_factory().coder()

def create_executor_agent() -> Agent:
    return _get_factory().executor()

def create_reviewer_agent() -> Agent:
    return _get_factory().reviewer()

def create_summarizer_agent() -> Agent:
    return _get_factory().summarizer()
