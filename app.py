"""
Streamlit Dashboard for CrewAI Astronomy Workflows
Production-ready UI with clean design
"""
import streamlit as st
import os
import sys
import yaml
import requests
from importlib import metadata
from datetime import datetime
from pathlib import Path
import contextlib
import io
import queue
import threading
import time

from workflow import WorkflowState, AstronomyWorkflow
from config import get_llm_config, get_workflow_config, init_directories, load_llm_profiles

# Initialize
init_directories()

# Page config
st.set_page_config(
    page_title="CrewAI Astronomy Workflows",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
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
    .status-completed {
        color: green;
        font-weight: bold;
    }
    .status-failed {
        color: red;
        font-weight: bold;
    }
    .status-running {
        color: orange;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'workflows' not in st.session_state:
    st.session_state.workflows = []
if 'current_workflow' not in st.session_state:
    st.session_state.current_workflow = None


def load_example_tasks(task_dir: Path) -> list[dict]:
    """Load example tasks from YAML files."""
    tasks = []
    if not task_dir.exists():
        return tasks

    for task_file in sorted(task_dir.glob("*.yaml")):
        try:
            with task_file.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except Exception:
            continue

        title = (data.get("title") or task_file.stem).strip()
        question = (data.get("question") or "").strip()
        data_source = (data.get("data_source") or "").strip()

        if question:
            tasks.append({
                "title": title,
                "question": question,
                "data_source": data_source
            })

    return tasks


class _QueueWriter(io.TextIOBase):
    """Write text to a queue for live UI streaming."""

    def __init__(self, target_queue: queue.Queue) -> None:
        self._queue = target_queue

    def write(self, text: str) -> int:
        if text:
            self._queue.put(text)
        return len(text)

    def flush(self) -> None:
        return None


def run_with_live_output(func, placeholder: st.delta_generator.DeltaGenerator) -> tuple[object, str]:
    """Run a function in a thread and stream stdout/stderr to the UI."""
    output_queue: queue.Queue = queue.Queue()
    output_chunks: list[str] = []
    done = threading.Event()
    result_container: dict = {}
    error_container: dict = {}

    writer = _QueueWriter(output_queue)

    def target() -> None:
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
            placeholder.markdown("".join(output_chunks) or "_Working..._")
        except queue.Empty:
            if not output_chunks:
                placeholder.markdown("_Working..._")
            time.sleep(0.05)

    stream_text = "".join(output_chunks).strip()
    placeholder.markdown(stream_text or "_No live output available._")

    if "error" in error_container:
        raise error_container["error"]

    return result_container.get("result"), stream_text


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
    # fallback: build a virtual profile from env
    cfg = get_llm_config()
    from config import LLMProfile
    return LLMProfile(name="default", base_url=cfg.base_url, api_key=cfg.api_key, model=cfg.model)


def apply_llm_overrides():
    """Apply active profile + selected model to runtime environment."""
    profile = _active_profile()
    base_url = profile.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    os.environ["LLM_BASE_URL"] = base_url
    os.environ["LLM_API_KEY"] = profile.api_key or ""
    os.environ["LLM_MODEL"] = st.session_state.get("llm_model_choice", profile.model)


def fetch_available_models(base_url: str, api_key: str) -> list[str]:
    """Fetch available model IDs from an OpenAI-compatible /v1 endpoint."""
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        url = f"{base}/models"
    else:
        url = f"{base}/v1/models"

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


def main():
    """Main dashboard"""
    init_llm_session_state()
    apply_llm_overrides()

    # Header
    st.markdown('<div class="main-header">🔭 CrewAI Astronomy Workflows</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Multi-Agent System for Astronomical Data Analysis</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "Select Page",
            ["🚀 New Workflow", "📊 Workflow History", "⚙️ Configuration"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Status
        st.markdown("### System Status")
        llm_config = get_llm_config()
        profile = _active_profile()
        st.success(f"✅ LLM: Ready")
        st.info(f"🧭 Profile: {profile.name}")
        st.info(f"🔗 Endpoint: {llm_config.base_url}")
        st.info(f"🤖 Model: {llm_config.model}")
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### Quick Stats")
        st.metric("Total Workflows", len(st.session_state.workflows))
        completed = len([w for w in st.session_state.workflows if w['status'] == 'completed'])
        st.metric("Completed", completed)

    # Main content
    if page == "🚀 New Workflow":
        render_new_workflow_page()
    elif page == "📊 Workflow History":
        render_history_page()
    elif page == "⚙️ Configuration":
        render_config_page()


def render_new_workflow_page():
    """Render new workflow creation page"""
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
                help="Choose which OpenAI-compatible endpoint to use"
            )
            new_idx = profile_names.index(selected_name)
            if new_idx != st.session_state.get("llm_profile_idx"):
                st.session_state.llm_profile_idx = new_idx
                # reset model choice to the profile default
                st.session_state.llm_model_choice = profiles[new_idx].model
                apply_llm_overrides()
                st.rerun()

        profile = _active_profile()

        with col_model:
            discovered = fetch_available_models(profile.base_url, profile.api_key)
            if discovered:
                current = st.session_state.get("llm_model_choice", profile.model)
                idx = discovered.index(current) if current in discovered else 0
                st.selectbox("Model", discovered, index=idx, key="llm_model_choice",
                             help="Models discovered from the endpoint")
            else:
                st.text_input("Model", key="llm_model_choice",
                              help="Could not discover models — enter the name manually")
        apply_llm_overrides()
    
    # Quick start examples
    with st.expander("💡 Example Research Questions", expanded=False):
        task_dir = Path("example_tasks")
        examples = load_example_tasks(task_dir)
        if not examples:
            examples = [
                {
                    "title": "Example 1",
                    "question": "Analyze the color-magnitude distribution of red giant stars in the Galactic bulge",
                    "data_source": "gaia.aip.de"
                },
                {
                    "title": "Example 2",
                    "question": "Study the proper motion distribution of stars in the solar neighborhood",
                    "data_source": "gaia.aip.de"
                },
                {
                    "title": "Example 3",
                    "question": "Investigate the relationship between stellar metallicity and kinematics",
                    "data_source": "gaia.aip.de"
                },
                {
                    "title": "Example 4",
                    "question": "Create an HR diagram for open cluster NGC 2516",
                    "data_source": "gaia.aip.de"
                }
            ]

        for i, example in enumerate(examples):
            label = example.get("title") or f"Example {i+1}"
            if st.button(label, key=f"ex_{i}"):
                st.session_state.example_question = example["question"]
                if example.get("data_source"):
                    st.session_state.example_data_source = example["data_source"]
                    st.session_state.data_source = example["data_source"]
                st.rerun()

    # Research question input
    default_question = st.session_state.get('example_question', '')
    research_question = st.text_area(
        "Research Question",
        value=default_question,
        height=120,
        placeholder="Enter your astronomy research question...\n\nExample: Analyze the spatial distribution and kinematics of red giant stars in the Galactic bulge using Gaia DR3 data",
        help="Describe what you want to analyze. Be specific about the data and analysis goals."
    )
    
    # Data source
    col1, col2 = st.columns([2, 1])
    with col1:
        data_source_options = [
            "gaia_dr3",
            "gaia_dr2",
            "sdss",
            "2mass",
            "gaia.aip.de",
            "data.aip.de",
            "numpy"
        ]
        if "data_source" not in st.session_state:
            default_source = st.session_state.get("example_data_source")
            st.session_state.data_source = default_source or data_source_options[0]
        selected = st.session_state.data_source
        index = data_source_options.index(selected) if selected in data_source_options else 0
        data_source = st.selectbox(
            "Data Source",
            data_source_options,
            index=index,
            key="data_source",
            help="Select the astronomical data catalog or data source to use"
        )
    
    with col2:
        use_cache = st.checkbox("Use cached data", value=True)

    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            llm_temp = st.slider("LLM Temperature", 0.0, 1.0, 0.3, 0.1)
        with col2:
            max_retries = st.number_input("Max Retries", 1, 5, 3)

    # Launch button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Launch Workflow", type="primary", use_container_width=True):
            if not research_question:
                st.error("⚠️ Please enter a research question")
                return
            
            launch_workflow(research_question, data_source)


def launch_workflow(research_question: str, data_source: str):
    """Launch new workflow with progress tracking"""
    # Progress container
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### 🔄 Workflow Execution")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Phase containers
        plan_container = st.expander("📋 Phase 1: Planning", expanded=True)
        analysis_container = st.expander("📊 Phase 2: Analysis Design", expanded=False)
        code_container = st.expander("💻 Phase 3: Code Generation", expanded=False)
        review_container = st.expander("✅ Phase 4: Code Review", expanded=False)
        
        try:
            # Initialize workflow
            status_text.markdown("**Status:** Initializing workflow...")
            progress_bar.progress(10)
            
            state = WorkflowState(
                research_question=research_question,
                data_source=data_source
            )
            workflow = AstronomyWorkflow(state=state)
            
            # Phase 1: Planning
            status_text.markdown("**Status:** 📋 Creating analysis plan...")
            progress_bar.progress(25)
            
            init_result = workflow.initialize()
            with plan_container:
                st.markdown("**Live Output:**")
                plan_stream_placeholder = st.empty()
            plan_result, plan_stream = run_with_live_output(
                lambda: workflow.planning_phase(init_result),
                plan_stream_placeholder
            )
            
            with plan_container:
                st.markdown("**Analysis Plan:**")
                st.markdown(state.analysis_plan)
            
            # Phase 2: Analysis
            status_text.markdown("**Status:** 📊 Designing statistical approach...")
            progress_bar.progress(50)
            
            with analysis_container:
                st.markdown("**Live Output:**")
                analysis_stream_placeholder = st.empty()
            analysis_result, analysis_stream = run_with_live_output(
                lambda: workflow.analysis_phase(plan_result),
                analysis_stream_placeholder
            )
            
            with analysis_container:
                st.markdown("**Statistical Approach:**")
                st.markdown(state.statistical_approach)
            
            # Phase 3: Coding
            status_text.markdown("**Status:** 💻 Generating Python code...")
            progress_bar.progress(75)
            
            with code_container:
                st.markdown("**Live Output:**")
                code_stream_placeholder = st.empty()
            code_result, code_stream = run_with_live_output(
                lambda: workflow.coding_phase(analysis_result),
                code_stream_placeholder
            )
            
            with code_container:
                st.markdown("**Generated Code:**")
                st.code(state.generated_code, language='python')
            
            # Phase 4: Review
            status_text.markdown("**Status:** ✅ Reviewing code quality...")
            progress_bar.progress(90)
            
            with review_container:
                st.markdown("**Live Output:**")
                review_stream_placeholder = st.empty()
            review_result, review_stream = run_with_live_output(
                lambda: workflow.review_phase(code_result),
                review_stream_placeholder
            )
            
            with review_container:
                st.markdown("**Review Report:**")
                st.markdown(state.review_report)
            
            # Complete
            progress_bar.progress(100)
            status_text.markdown("**Status:** ✨ Workflow completed!")
            
            # Save to history
            workflow_data = {
                'workflow_id': state.workflow_id,
                'timestamp': datetime.now().isoformat(),
                'research_question': research_question,
                'data_source': data_source,
                'status': 'completed',
                'plan_stream': plan_stream,
                'analysis_stream': analysis_stream,
                'code_stream': code_stream,
                'review_stream': review_stream,
                'analysis_plan': state.analysis_plan,
                'statistical_approach': state.statistical_approach,
                'generated_code': state.generated_code,
                'review_report': state.review_report,
                'code_file': workflow.get_code_path(),
                'readme_file': workflow.get_readme_path()
            }
            st.session_state.workflows.append(workflow_data)
            
            # Success message
            st.success(f"✅ Workflow {state.workflow_id} completed successfully!")
            
            # Show file locations
            st.info(f"""
            **Generated Files:**
            - 📄 Python Script: `{workflow.get_code_path()}`
            - 📖 Documentation: `{workflow.get_readme_path()}`
            """)
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download Python Script",
                    state.generated_code,
                    file_name=f"workflow_{state.workflow_id}.py",
                    mime="text/x-python",
                    use_container_width=True
                )
            with col2:
                readme_content = workflow.generate_readme()
                st.download_button(
                    "📥 Download README.md",
                    readme_content,
                    file_name=f"README_{state.workflow_id}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"❌ Workflow failed: {str(e)}")
            status_text.markdown(f"**Status:** Failed - {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())


def render_history_page():
    """Render workflow history page"""
    st.markdown("## 📊 Workflow History")
    
    if not st.session_state.workflows:
        st.info("📭 No workflows yet. Create your first workflow from the 🚀 New Workflow page.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(st.session_state.workflows)
    completed = len([w for w in st.session_state.workflows if w['status'] == 'completed'])
    failed = len([w for w in st.session_state.workflows if w['status'] == 'failed'])
    
    col1.metric("Total Workflows", total)
    col2.metric("Completed", completed)
    col3.metric("Failed", failed)
    col4.metric("Success Rate", f"{(completed/total*100):.0f}%" if total > 0 else "0%")
    
    st.markdown("---")
    
    # Workflow list
    for workflow in reversed(st.session_state.workflows):
        with st.expander(
            f"🔬 {workflow['workflow_id']} - {workflow['research_question'][:60]}...",
            expanded=False
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Question:** {workflow['research_question']}")
                st.markdown(f"**Data Source:** {workflow['data_source']}")
                st.markdown(f"**Timestamp:** {workflow['timestamp']}")
            
            with col2:
                status_class = f"status-{workflow['status']}"
                st.markdown(f"**Status:** <span class='{status_class}'>{workflow['status'].upper()}</span>", 
                           unsafe_allow_html=True)
            
            # Tabs for details
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Plan", "📊 Analysis", "💻 Code", "✅ Review"])
            
            with tab1:
                if workflow.get('plan_stream'):
                    st.markdown("**Live Output:**")
                    st.markdown(workflow['plan_stream'])
                st.markdown(workflow.get('analysis_plan', 'N/A'))
            
            with tab2:
                if workflow.get('analysis_stream'):
                    st.markdown("**Live Output:**")
                    st.markdown(workflow['analysis_stream'])
                st.markdown(workflow.get('statistical_approach', 'N/A'))
            
            with tab3:
                if workflow.get('code_stream'):
                    st.markdown("**Live Output:**")
                    st.markdown(workflow['code_stream'])
                if workflow.get('generated_code'):
                    st.code(workflow['generated_code'], language='python')
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 Download Python Script",
                            workflow['generated_code'],
                            file_name=f"workflow_{workflow['workflow_id']}.py",
                            key=f"download_py_{workflow['workflow_id']}"
                        )
                    with col2:
                        # Generate README content
                        readme_content = f"""# Workflow {workflow['workflow_id']}

Research Question: {workflow['research_question']}
Data Source: {workflow['data_source']}
Timestamp: {workflow['timestamp']}

## Analysis Plan

{workflow.get('analysis_plan', 'N/A')}

## Statistical Approach

{workflow.get('statistical_approach', 'N/A')}

## Usage

```bash
python workflow_{workflow['workflow_id']}.py
```

## Code Review

{workflow.get('review_report', 'N/A')}
"""
                        st.download_button(
                            "📥 Download README.md",
                            readme_content,
                            file_name=f"README_{workflow['workflow_id']}.md",
                            mime="text/markdown",
                            key=f"download_md_{workflow['workflow_id']}"
                        )
            
            with tab4:
                if workflow.get('review_stream'):
                    st.markdown("**Live Output:**")
                    st.markdown(workflow['review_stream'])
                st.markdown(workflow.get('review_report', 'N/A'))


def render_config_page():
    """Render configuration page"""
    st.markdown("## ⚙️ Configuration")
    
    # LLM Profiles
    st.markdown("### 🤖 LLM Endpoint Profiles")
    profiles = st.session_state.get("llm_profiles", [])
    if profiles:
        for i, p in enumerate(profiles):
            active = "(active)" if i == st.session_state.get("llm_profile_idx", 0) else ""
            with st.expander(f"**{p.name}** {active}", expanded=(i == st.session_state.get("llm_profile_idx", 0))):
                st.text_input("Base URL", value=p.base_url, disabled=True, key=f"cfg_url_{i}")
                st.text_input("API Key", value="••••" if p.api_key else "(none)", disabled=True, key=f"cfg_key_{i}")
                st.text_input("Default Model", value=p.model, disabled=True, key=f"cfg_model_{i}")
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
        
        All endpoints must be OpenAI-compatible (`/v1`). Leave `API_KEY` empty for endpoints
        that do not require authentication (e.g. local Ollama).  
        Restart the app after editing `.env`.
        """)
    
    # Workflow Configuration
    st.markdown("### 📁 Workflow Settings")
    
    workflow_config = get_workflow_config()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Output Directory", value=workflow_config.output_dir, disabled=True)
        st.number_input("Max Retries", value=workflow_config.max_retries, disabled=True)
    
    with col2:
        st.text_input("Results Directory", value=workflow_config.results_dir, disabled=True)
        st.checkbox("Verbose Logging", value=workflow_config.verbose, disabled=True)
    
    # System Info
    st.markdown("### 📊 System Information")

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


if __name__ == "__main__":
    main()
