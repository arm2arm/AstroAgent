"""
Streamlit Dashboard for CrewAI Multi-Agent Workflows
Production-ready UI with clean design
"""
import os
import argparse

# Disable CrewAI telemetry to avoid signal handling in background threads.
os.environ.setdefault("CREWAI_TELEMETRY_ENABLED", "false")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

import streamlit as st
import sys
import yaml
import requests
from typing import TYPE_CHECKING
from importlib import metadata
from datetime import datetime
from pathlib import Path
import contextlib
import io
import queue
import threading
import time
import logging

if TYPE_CHECKING:
    from workflow import WorkflowState, AstronomyWorkflow
from config import (
    get_execution_config,
    get_llm_config,
    get_workflow_config,
    init_directories,
    load_llm_profiles,
)

# Page config
st.set_page_config(
    page_title="CrewAI Workflows",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .status-completed { color: green; font-weight: bold; }
    .status-failed    { color: red;   font-weight: bold; }
    .status-running   { color: orange; font-weight: bold; }
    /* Full-width log area */
    .stExpander [data-testid="stExpanderDetails"] {
        max-width: 100%;
    }
    .log-output {
        background-color: #0e1117;
        color: #e0e0e0;
        font-family: 'Fira Code', 'Consolas', 'Courier New', monospace;
        font-size: 0.82rem;
        line-height: 1.45;
        padding: 1rem;
        border-radius: 0.5rem;
        overflow-x: auto;
        white-space: pre-wrap;
        word-break: break-word;
        max-height: 600px;
        overflow-y: auto;
        width: 100%;
        box-sizing: border-box;
    }
    /* Force the Live Agent Output expander to break out to full viewport width */
    [data-testid="stExpander"]:has(.log-output) {
        width: 100vw;
        position: relative;
        left: 50%;
        margin-left: -50vw;
        padding-left: 1rem;
        padding-right: 1rem;
        box-sizing: border-box;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Session state
if "workflows" not in st.session_state:
    st.session_state.workflows = []
if "current_workflow" not in st.session_state:
    st.session_state.current_workflow = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _init_directories_once() -> None:
    init_directories()

def load_example_tasks(task_dir: Path) -> list[dict]:
    """Load example tasks from YAML files."""
    signature = _example_tasks_signature(task_dir)
    return _load_example_tasks_cached(str(task_dir), signature)


def _example_tasks_signature(task_dir: Path) -> tuple[tuple[str, float], ...]:
    if not task_dir.exists():
        return ()
    files = sorted(task_dir.glob("*.yaml"))
    signature: list[tuple[str, float]] = []
    for path in files:
        try:
            signature.append((path.name, path.stat().st_mtime))
        except OSError:
            continue
    return tuple(signature)


@st.cache_data(ttl=300, show_spinner=False)
def _load_example_tasks_cached(task_dir: str, signature: tuple[tuple[str, float], ...]) -> list[dict]:
    tasks = []
    base = Path(task_dir)
    if not base.exists():
        return tasks

    for task_file in sorted(base.glob("*.yaml")):
        try:
            with task_file.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except Exception:
            continue

        title = (data.get("title") or task_file.stem).strip()
        question = (data.get("question") or "").strip()
        complexity = int(data.get("complexity", -1))

        if question:
            tasks.append({
                "title": title,
                "question": question,
                "complexity": complexity,
            })

    return tasks


def _escape_html(text: str) -> str:
    """Escape HTML special chars so raw agent output renders safely."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


class _QueueWriter(io.TextIOBase):
    """Write text to a queue for live UI streaming."""

    def __init__(self, target_queue: queue.Queue) -> None:
        self._queue = target_queue

    def write(self, text) -> int:
        if text:
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
            self._queue.put(str(text))
        return len(text)

    def flush(self) -> None:
        return None


def run_with_live_output(
    func,
    placeholder: st.delta_generator.DeltaGenerator,
) -> tuple[object, str]:
    """Run a function in a thread and stream stdout/stderr to the UI."""
    output_queue: queue.Queue = queue.Queue()
    output_chunks: list[str] = []
    done = threading.Event()
    result_container: dict = {}
    error_container: dict = {}

    writer = _QueueWriter(output_queue)

    def target() -> None:
        # Suppress CrewAI telemetry signal-handler warnings when running
        # in a background thread (signal registration is main-thread only).
        logging.getLogger("crewai.telemetry").setLevel(logging.CRITICAL)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                result_container["result"] = func()
        except Exception as exc:
            error_container["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()

    while not done.is_set() or not output_queue.empty():
        try:
            chunk = output_queue.get(timeout=0.1)
            output_chunks.append(chunk)
            combined = "".join(output_chunks)
            placeholder.markdown(
                f'<div class="log-output">{_escape_html(combined)}</div>',
                unsafe_allow_html=True,
            )
        except queue.Empty:
            if not output_chunks:
                placeholder.markdown("_Working..._")
            time.sleep(0.05)

    stream_text = "".join(output_chunks).strip()
    if stream_text:
        placeholder.markdown(
            f'<div class="log-output">{_escape_html(stream_text)}</div>',
            unsafe_allow_html=True,
        )
    else:
        placeholder.markdown("_No live output available._")

    if "error" in error_container:
        raise error_container["error"]

    return result_container.get("result"), stream_text


# ---------------------------------------------------------------------------
# LLM profile helpers
# ---------------------------------------------------------------------------

def init_llm_session_state():
    """Initialize LLM session defaults from .env profiles."""
    profiles = load_llm_profiles()
    if "llm_profiles" not in st.session_state:
        st.session_state.llm_profiles = profiles
    if "llm_profile_idx" not in st.session_state:
        st.session_state.llm_profile_idx = 0
    if "llm_model_choice" not in st.session_state:
        if profiles:
            st.session_state.llm_model_choice = profiles[0].model
        else:
            st.session_state.llm_model_choice = get_llm_config().model


def _active_profile():
    """Return the currently selected profile (or a fallback)."""
    profiles = st.session_state.get("llm_profiles", [])
    idx = st.session_state.get("llm_profile_idx", 0)
    if profiles and 0 <= idx < len(profiles):
        return profiles[idx]
    cfg = get_llm_config()
    from config import LLMProfile
    return LLMProfile(
        name="default",
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        model=cfg.model,
        provider=cfg.provider,
        context_window=cfg.context_window,
        output_budget=cfg.output_budget,
        embed_model=cfg.embed_model,
        embed_provider=cfg.embed_provider,
        embed_base_url=cfg.embed_base_url,
        embed_api_key=cfg.embed_api_key,
    )


def apply_llm_overrides():
    """Apply active profile + selected model to runtime environment."""
    profile = _active_profile()
    base_url = profile.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    api_key = profile.api_key or "no-key-required"
    os.environ["LLM_BASE_URL"] = base_url
    os.environ["LLM_API_KEY"] = api_key
    os.environ["LLM_MODEL"] = st.session_state.get("llm_model_choice", profile.model)
    os.environ["LLM_PROVIDER"] = getattr(profile, "provider", "openai")
    os.environ["LLM_CONTEXT_WINDOW"] = str(profile.context_window)
    os.environ["LLM_OUTPUT_BUDGET"] = str(profile.output_budget)
    # Embedding model config for CrewAI memory
    os.environ["EMBED_MODEL"] = profile.embed_model or ""
    os.environ["EMBED_PROVIDER"] = profile.embed_provider or "ollama"
    if profile.embed_base_url:
        os.environ["EMBED_BASE_URL"] = profile.embed_base_url
    else:
        os.environ.pop("EMBED_BASE_URL", None)
    if profile.embed_api_key:
        os.environ["EMBED_API_KEY"] = profile.embed_api_key
    else:
        os.environ.pop("EMBED_API_KEY", None)
    # CrewAI's OllamaProvider reads these env vars directly as fallback
    if (profile.embed_provider or "ollama").lower() == "ollama" and profile.embed_model:
        ollama_base = (profile.embed_base_url or base_url).rstrip("/")
        if ollama_base.endswith("/v1"):
            ollama_base = ollama_base[:-3]
        os.environ["EMBEDDINGS_OLLAMA_MODEL_NAME"] = profile.embed_model
        os.environ["EMBEDDINGS_OLLAMA_URL"] = f"{ollama_base}/api/embeddings"
    # LiteLLM / openai client also reads OPENAI_API_KEY directly
    os.environ["OPENAI_API_KEY"] = api_key
    # Ensure OpenAI-compatible clients use the correct base URL
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_BASE"] = base_url


@st.cache_data(ttl=300, show_spinner=False)
def fetch_available_models(base_url: str, api_key: str) -> list[str]:
    """Fetch model IDs from an OpenAI-compatible /v1 endpoint."""
    base = base_url.rstrip("/")
    url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"

    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", []) if isinstance(payload, dict) else []
        models = [item.get("id") for item in data if isinstance(item, dict)]
        return sorted({m for m in models if m})
    except Exception:
        return []


# =========================================================================
# Main
# =========================================================================

def main():
    """Main dashboard."""
    _init_directories_once()
    init_llm_session_state()
    apply_llm_overrides()

    # Header
    st.markdown(
        '<div class="main-header">🧠 CrewAI Workflows</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">AI-Powered Multi-Agent System for Code and Data Tasks</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "Select Page",
            ["🚀 New Workflow", "📊 Workflow History", "⚙️ Configuration"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Status
        st.markdown("### System Status")
        profile = _active_profile()
        llm_config = get_llm_config()
        st.success("✅ LLM: Ready")
        st.info(f"🧭 Profile: {profile.name}")
        st.info(f"🔗 Endpoint: {llm_config.base_url}")
        st.info(f"🤖 Model: {llm_config.model}")

        st.markdown("---")
        st.markdown("### Quick Stats")
        st.metric("Total Workflows", len(st.session_state.workflows))
        completed = len([w for w in st.session_state.workflows if w["status"] == "completed"])
        st.metric("Completed", completed)

    # Main content
    if page == "🚀 New Workflow":
        render_new_workflow_page()
    elif page == "📊 Workflow History":
        render_history_page()
    elif page == "⚙️ Configuration":
        render_config_page()


# =========================================================================
# New Workflow Page
# =========================================================================

def render_new_workflow_page():
    """Render new workflow creation page."""
    st.markdown("## 🚀 Create New Workflow")

    # ── Endpoint & Model selectors ──
    profiles = st.session_state.get("llm_profiles", [])
    if profiles:
        profile_names = [p.name for p in profiles]
        col_ep, col_model = st.columns([1, 1])
        with col_ep:
            selected_name = st.selectbox(
                "LLM Endpoint",
                profile_names,
                index=st.session_state.get("llm_profile_idx", 0),
                key="_profile_select",
                help="Choose which OpenAI-compatible endpoint to use",
            )
            new_idx = profile_names.index(selected_name)
            if new_idx != st.session_state.get("llm_profile_idx"):
                st.session_state.llm_profile_idx = new_idx
                st.session_state.llm_model_choice = profiles[new_idx].model
                apply_llm_overrides()
                st.rerun()

        profile = _active_profile()
        if "model_cache" not in st.session_state:
            st.session_state.model_cache = {}
        cache_key = f"models_{st.session_state.get('llm_profile_idx', 0)}"

        with col_model:
            if st.button("Discover Models", key=f"discover_{cache_key}"):
                st.session_state.model_cache[cache_key] = fetch_available_models(
                    profile.base_url,
                    profile.api_key,
                )
            discovered = st.session_state.model_cache.get(cache_key, [])
            if discovered:
                current = st.session_state.get("llm_model_choice", profile.model)
                idx = discovered.index(current) if current in discovered else 0
                st.selectbox(
                    "Model", discovered, index=idx, key="llm_model_choice",
                    help="Models discovered from the endpoint",
                )
            else:
                st.text_input(
                    "Model", key="llm_model_choice",
                    help="Could not discover models — enter the name manually",
                )
        apply_llm_overrides()

    # ── Example tasks ──
    with st.expander("💡 Example Tasks", expanded=False):
        task_dir = Path("example_tasks")
        examples = load_example_tasks(task_dir)
        if not examples:
            examples = [
                {"title": "Sine Plot", "question": "Plot sin(x) in Python and save the figure", "complexity": 2},
                {"title": "CSV Summary", "question": "Load a CSV and summarize columns with basic stats", "complexity": 3},
            ]

        for i, example in enumerate(examples):
            label = example.get("title") or f"Example {i+1}"
            if st.button(label, key=f"ex_{i}"):
                st.session_state.example_question = example["question"]
                st.session_state.example_complexity = example.get("complexity", -1)
                st.rerun()

    # ── Research question ──
    default_question = st.session_state.get("example_question", "")
    research_question = st.text_area(
        "Task Request",
        value=default_question,
        height=120,
        placeholder="Describe the task you want to run...",
        help="Describe what you want to build or analyze. Be specific.",
    )

    # ── Task complexity ──
    col1, col2 = st.columns([2, 1])
    with col2:
        default_complexity = st.session_state.get("example_complexity", -1)
        if default_complexity < 0:
            default_complexity = 5
        task_complexity = st.slider(
            "Task Complexity",
            min_value=1, max_value=10, value=default_complexity,
            help="1-3 = simple (skip Planner/Analyst), 4-10 = complex (full pipeline)",
        )

    # ── Advanced options ──
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            st.slider("LLM Temperature", 0.0, 1.0, 0.3, 0.1)
        with col2:
            st.number_input("Max Review Iterations", 1, 5, get_workflow_config().max_review_iterations)

    # ── Launch button ──
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Launch Workflow", type="primary", use_container_width=True):
            if not research_question:
                st.error("⚠️ Please enter a research question")
                return
            launch_workflow(research_question, task_complexity)


# =========================================================================
# Launch Workflow
# =========================================================================

def launch_workflow(research_question: str, task_complexity: int):
    """Launch new workflow with progress tracking and live output."""
    from workflow import WorkflowState, AstronomyWorkflow
    progress_container = st.container()

    with progress_container:
        st.markdown("### 🔄 Workflow Execution")

        progress_bar = st.progress(0)
        status_text = st.empty()

        # -- Dynamic expanders based on complexity --
        is_complex = task_complexity > 3
        path_label = "Complex" if is_complex else "Simple"
        st.info(f"**Path:** {path_label} (complexity = {task_complexity})")

        if is_complex:
            plan_exp = st.expander("📋 Phase 1: Planning", expanded=True)
            analysis_exp = st.expander("📊 Phase 2: Analysis Design", expanded=False)
            code_exp = st.expander("💻 Phase 3: Code Generation", expanded=False)
            exec_exp = st.expander("🐳 Phase 4: Execution", expanded=False)
            review_exp = st.expander("✅ Phase 5: Code Review", expanded=False)
        else:
            code_exp = st.expander("💻 Phase 1: Code Generation", expanded=True)
            exec_exp = st.expander("🐳 Phase 2: Execution", expanded=False)
            review_exp = st.expander("✅ Phase 3: Code Review", expanded=False)

        # Live output area (captures the full CrewAI output stream)
        stream_exp = st.expander("📡 Live Agent Output", expanded=True)
        with stream_exp:
            stream_placeholder = st.empty()

        try:
            # Initialize workflow
            status_text.markdown("**Status:** Initializing workflow...")
            progress_bar.progress(5)

            state = WorkflowState(
                research_question=research_question,
                task_complexity=task_complexity,
            )
            workflow = AstronomyWorkflow(state=state)

            # Run the entire flow with live output capture
            status_text.markdown("**Status:** 🔄 Running workflow...")
            progress_bar.progress(10)

            _, stream_text = run_with_live_output(
                lambda: workflow.kickoff(),
                stream_placeholder,
            )

            # Flow updates its own state instance; grab the updated state.
            state = workflow.state
            if not state.execution_artifacts:
                results_dir = Path(workflow.get_results_dir())
                if results_dir.exists():
                    patterns = ("*.png", "*.jpg", "*.jpeg", "*.svg", "*.pdf")
                    artifacts: list[str] = []
                    for pattern in patterns:
                        artifacts.extend([str(p) for p in results_dir.glob(pattern)])
                    state.execution_artifacts = sorted(set(artifacts))

            progress_bar.progress(100)
            status_text.markdown("**Status:** ✨ Workflow completed!")

            # -- Fill in the phase expanders with results --
            if is_complex:
                with plan_exp:
                    if state.analysis_plan:
                        st.markdown("**Analysis Plan:**")
                        st.markdown(state.analysis_plan)
                    else:
                        st.markdown("_Skipped_")

                with analysis_exp:
                    if state.statistical_approach:
                        st.markdown("**Statistical Approach:**")
                        st.markdown(state.statistical_approach)
                    else:
                        st.markdown("_Skipped_")

            with code_exp:
                if state.generated_code:
                    st.markdown("**Generated Code:**")
                    st.code(state.generated_code, language="python")
                else:
                    st.markdown("_No code generated._")

            with exec_exp:
                if state.execution_stdout:
                    st.markdown("**Stdout:**")
                    st.code(state.execution_stdout)
                if state.execution_stderr:
                    st.markdown("**Stderr / Errors:**")
                    st.code(state.execution_stderr)
                if not state.execution_stdout and not state.execution_stderr:
                    st.markdown("_No execution output._")

                # Display artifacts (images)
                if state.execution_artifacts:
                    st.markdown("**Generated Artifacts:**")
                    for artifact in state.execution_artifacts:
                        try:
                            st.image(artifact, caption=os.path.basename(artifact))
                        except Exception:
                            st.markdown(f"- `{artifact}`")

            with review_exp:
                if state.review_report:
                    st.markdown("**Review Report:**")
                    st.markdown(state.review_report)
                st.markdown(f"**Iterations:** {state.iterations}")
                st.markdown(f"**Approved:** {'✅ Yes' if state.approved else '⚠️ No (max iterations reached)'}")

            # Success
            st.success(f"✅ Workflow {state.workflow_id} completed! ({state.iterations} review round(s), approved={state.approved})")

            # File locations
            st.info(f"""
            **Generated Files:**
            - 📄 Python Script: `{workflow.get_code_path()}`
            - 📖 Documentation: `{workflow.get_readme_path()}`
            - 📄 Results Script: `{workflow.get_results_code_path()}`
            - 📖 Results README: `{workflow.get_results_readme_path()}`
            - 📁 Results Dir: `{workflow.get_results_dir()}`
            """)

            # Downloads
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download Python Script",
                    state.generated_code or "",
                    file_name=f"workflow_{state.workflow_id}.py",
                    mime="text/x-python",
                    use_container_width=True,
                )
            with col2:
                readme_content = workflow.generate_readme(artifact_prefix="")
                st.download_button(
                    "📥 Download README.md",
                    readme_content,
                    file_name=f"README_{state.workflow_id}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            # Save to history
            workflow_data = {
                "workflow_id": state.workflow_id,
                "timestamp": datetime.now().isoformat(),
                "research_question": research_question,
                "task_complexity": task_complexity,
                "status": "completed",
                "stream_text": stream_text,
                "analysis_plan": state.analysis_plan,
                "statistical_approach": state.statistical_approach,
                "generated_code": state.generated_code,
                "execution_stdout": state.execution_stdout,
                "execution_stderr": state.execution_stderr,
                "execution_artifacts": list(state.execution_artifacts),
                "review_report": state.review_report,
                "iterations": state.iterations,
                "approved": state.approved,
                "code_file": workflow.get_code_path(),
                "readme_file": workflow.get_readme_path(),
                "results_code_file": workflow.get_results_code_path(),
                "results_readme_file": workflow.get_results_readme_path(),
                "results_dir": workflow.get_results_dir(),
            }
            st.session_state.workflows.append(workflow_data)

        except Exception as e:
            st.error(f"❌ Workflow failed: {str(e)}")
            status_text.markdown(f"**Status:** Failed - {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())


# =========================================================================
# History Page
# =========================================================================

def render_history_page():
    """Render workflow history page."""
    st.markdown("## 📊 Workflow History")

    if not st.session_state.workflows:
        st.info("📭 No workflows yet. Create your first workflow from the 🚀 New Workflow page.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    total = len(st.session_state.workflows)
    completed = len([w for w in st.session_state.workflows if w["status"] == "completed"])
    failed = len([w for w in st.session_state.workflows if w["status"] == "failed"])
    col1.metric("Total Workflows", total)
    col2.metric("Completed", completed)
    col3.metric("Failed", failed)
    col4.metric("Success Rate", f"{(completed/total*100):.0f}%" if total > 0 else "0%")

    st.markdown("---")

    for workflow in reversed(st.session_state.workflows):
        q = workflow["research_question"][:60]
        with st.expander(f"🔬 {workflow['workflow_id']} - {q}...", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Question:** {workflow['research_question']}")
                st.markdown(f"**Complexity:** {workflow.get('task_complexity', '?')}")
                st.markdown(f"**Timestamp:** {workflow['timestamp']}")
            with col2:
                status_class = f"status-{workflow['status']}"
                st.markdown(
                    f"**Status:** <span class='{status_class}'>{workflow['status'].upper()}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Iterations:** {workflow.get('iterations', '?')}")
                st.markdown(f"**Approved:** {'✅' if workflow.get('approved') else '⚠️'}")

            # Tabs
            tab_names = ["📋 Plan", "📊 Analysis", "💻 Code", "🐳 Execution", "✅ Review", "📡 Log"]
            tab_plan, tab_analysis, tab_code, tab_exec, tab_review, tab_log = st.tabs(tab_names)

            with tab_plan:
                st.markdown(workflow.get("analysis_plan") or "_N/A (simple path)_")

            with tab_analysis:
                st.markdown(workflow.get("statistical_approach") or "_N/A (simple path)_")

            with tab_code:
                if workflow.get("generated_code"):
                    st.code(workflow["generated_code"], language="python")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 Download Script",
                            workflow["generated_code"],
                            file_name=f"workflow_{workflow['workflow_id']}.py",
                            key=f"dl_py_{workflow['workflow_id']}",
                        )
                else:
                    st.markdown("_No code._")

            with tab_exec:
                if workflow.get("execution_stdout"):
                    st.markdown("**Stdout:**")
                    st.code(workflow["execution_stdout"])
                if workflow.get("execution_stderr"):
                    st.markdown("**Stderr / Errors:**")
                    st.code(workflow["execution_stderr"])
                if workflow.get("execution_artifacts"):
                    st.markdown("**Artifacts:**")
                    for art in workflow["execution_artifacts"]:
                        try:
                            st.image(art, caption=os.path.basename(art))
                        except Exception:
                            st.markdown(f"- `{art}`")
                if not workflow.get("execution_stdout") and not workflow.get("execution_stderr"):
                    st.markdown("_No execution output._")

            with tab_review:
                st.markdown(workflow.get("review_report") or "_No review._")

            with tab_log:
                log_text = workflow.get("stream_text")
                if log_text:
                    st.markdown(
                        f'<div class="log-output">{_escape_html(log_text)}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("_No log captured._")


# =========================================================================
# Configuration Page
# =========================================================================

def render_config_page():
    """Render configuration page."""
    st.markdown("## ⚙️ Configuration")

    # -- LLM Profiles --
    st.markdown("### 🤖 LLM Endpoint Profiles")
    profiles = st.session_state.get("llm_profiles", [])
    if profiles:
        for i, p in enumerate(profiles):
            active = "(active)" if i == st.session_state.get("llm_profile_idx", 0) else ""
            with st.expander(
                f"**{p.name}** {active}",
                expanded=(i == st.session_state.get("llm_profile_idx", 0)),
            ):
                st.text_input("Base URL", value=p.base_url, disabled=True, key=f"cfg_url_{i}")
                st.text_input("API Key", value="••••" if p.api_key else "(none)", disabled=True, key=f"cfg_key_{i}")
                st.text_input("Default Model", value=p.model, disabled=True, key=f"cfg_model_{i}")
                st.number_input("Context Window", value=p.context_window, disabled=True, key=f"cfg_ctx_{i}")
                st.number_input("Output Budget", value=p.output_budget, disabled=True, key=f"cfg_budget_{i}")
    else:
        st.warning("No profiles found. Add `LLM_1_*` entries to your `.env` file.")

    llm_config = get_llm_config()
    st.markdown("### Active LLM Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Active Endpoint", value=llm_config.base_url, disabled=True)
        st.text_input("Active Model", value=llm_config.model, disabled=True)
    with col2:
        st.number_input("Temperature", value=llm_config.temperature, disabled=True)
        st.number_input("Max Tokens", value=llm_config.max_tokens, disabled=True)
        st.number_input("Context Window", value=llm_config.context_window, disabled=True)
        st.number_input("Output Budget", value=llm_config.output_budget, disabled=True)

    with st.expander("🔐 Adding Profiles"):
        st.markdown("""
        Define profiles in your `.env` file using numbered keys:

        ```
        LLM_1_NAME=Ollama Local
        LLM_1_BASE_URL=http://localhost:11434/v1
        LLM_1_API_KEY=
        LLM_1_MODEL=qwen3-coder:latest

        LLM_2_NAME=AIP
        LLM_2_BASE_URL=https://ai.aip.de/api
        LLM_2_API_KEY=your-api-key
        LLM_2_MODEL=Qwen2.5-32B-Instruct
        ```

        All endpoints must be OpenAI-compatible (`/v1`). Leave `API_KEY` empty for
        endpoints that do not require authentication (e.g. local Ollama).
        Restart the app after editing `.env`.
        """)

    # -- Execution Environment --
    st.markdown("### 🐳 Execution Environment")
    exec_config = get_execution_config()
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Mode", value=exec_config.mode, disabled=True)
        st.text_input(
            "Image" if exec_config.mode == "image" else "Dockerfile",
            value=exec_config.image if exec_config.mode == "image" else exec_config.dockerfile_path,
            disabled=True,
        )
    with col2:
        st.text_input("Pre-installed Packages", value=", ".join(exec_config.pre_install), disabled=True)

    # -- Workflow Settings --
    st.markdown("### 📁 Workflow Settings")
    workflow_config = get_workflow_config()
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Output Directory", value=workflow_config.output_dir, disabled=True)
        st.number_input("Max Review Iterations", value=workflow_config.max_review_iterations, disabled=True)
    with col2:
        st.text_input("Results Directory", value=workflow_config.results_dir, disabled=True)
        st.checkbox("Verbose Logging", value=workflow_config.verbose, disabled=True)

    # -- System Info --
    st.markdown("### 📊 System Information")
    if st.button("Load System Info", key="load_sys_info"):
        st.session_state["show_sys_info"] = True

    if st.session_state.get("show_sys_info"):
        try:
            crewai_version = metadata.version("crewai")
        except metadata.PackageNotFoundError:
            crewai_version = "not installed"
        try:
            crewai_tools_version = metadata.version("crewai-tools")
        except metadata.PackageNotFoundError:
            crewai_tools_version = "not installed"

        st.markdown(f"""
        - **Python Version:** {sys.version.split()[0]}
        - **Streamlit Version:** {st.__version__}
        - **CrewAI Version:** {crewai_version}
        - **CrewAI Tools Version:** {crewai_tools_version}
        - **Working Directory:** `{os.getcwd()}`
        """)
    else:
        st.markdown("System info is loaded on demand to speed up page render.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the AstroAgent UI or CLI workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python app.py -q \"plot sin plot in green\"\n"
            "  python app.py -q \"plot sin plot in green\" -c 1\n"
        ),
    )
    parser.add_argument(
        "-q",
        "--question",
        help="Workflow prompt to run from the CLI",
        default="",
    )
    parser.add_argument(
        "-c",
        "--complexity",
        type=int,
        default=-1,
        help="Task complexity (-1 to auto-detect)",
    )
    args = parser.parse_args()

    if args.question:
        from workflow import AstronomyWorkflow, WorkflowState

        state = WorkflowState(
            research_question=args.question,
            task_complexity=args.complexity,
        )
        workflow = AstronomyWorkflow(state=state)
        workflow.kickoff()
        result_state = workflow.state
        print("\n--- Workflow Result ---")
        print(f"Workflow ID: {result_state.workflow_id}")
        print(f"Status     : {result_state.status}")
        print(f"Approved   : {result_state.approved}")
        if result_state.execution_artifacts:
            print("Artifacts  :")
            for path in result_state.execution_artifacts:
                print(f"  - {path}")
        else:
            print("Artifacts  : (none)")
        print("\n--- Review ---")
        print(result_state.review_report or "(no review report)")
    else:
        main()
