"""
Streamlit Dashboard for CrewAI Astronomy Workflows
Production-ready UI with clean design
"""
import streamlit as st
import os
import sys
from datetime import datetime
from pathlib import Path

from workflow import WorkflowState, AstronomyWorkflow
from config import get_llm_config, get_workflow_config, init_directories

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


def main():
    """Main dashboard"""
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
        st.success(f"✅ LLM: Ready")
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
    
    # Quick start examples
    with st.expander("💡 Example Research Questions", expanded=False):
        examples = [
            "Analyze the color-magnitude distribution of red giant stars in the Galactic bulge",
            "Study the proper motion distribution of stars in the solar neighborhood",
            "Investigate the relationship between stellar metallicity and kinematics",
            "Create an HR diagram for open cluster NGC 2516"
        ]
        for i, example in enumerate(examples):
            if st.button(f"Use Example {i+1}", key=f"ex_{i}"):
                st.session_state.example_question = example
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
        data_source = st.selectbox(
            "Data Source",
            ["gaia_dr3", "gaia_dr2", "sdss", "2mass"],
            help="Select the astronomical data catalog to use"
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
            plan_result = workflow.planning_phase(init_result)
            
            with plan_container:
                st.markdown("**Analysis Plan:**")
                st.markdown(state.analysis_plan)
            
            # Phase 2: Analysis
            status_text.markdown("**Status:** 📊 Designing statistical approach...")
            progress_bar.progress(50)
            
            analysis_result = workflow.analysis_phase(plan_result)
            
            with analysis_container:
                st.markdown("**Statistical Approach:**")
                st.markdown(state.statistical_approach)
            
            # Phase 3: Coding
            status_text.markdown("**Status:** 💻 Generating Python code...")
            progress_bar.progress(75)
            
            code_result = workflow.coding_phase(analysis_result)
            
            with code_container:
                st.markdown("**Generated Code:**")
                st.code(state.generated_code, language='python')
            
            # Phase 4: Review
            status_text.markdown("**Status:** ✅ Reviewing code quality...")
            progress_bar.progress(90)
            
            review_result = workflow.review_phase(code_result)
            
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
                'analysis_plan': state.analysis_plan,
                'statistical_approach': state.statistical_approach,
                'generated_code': state.generated_code,
                'review_report': state.review_report,
                'code_file': workflow._get_code_path(),
                'readme_file': workflow._get_readme_path()
            }
            st.session_state.workflows.append(workflow_data)
            
            # Success message
            st.success(f"✅ Workflow {state.workflow_id} completed successfully!")
            
            # Show file locations
            st.info(f"""
            **Generated Files:**
            - 📄 Python Script: `{workflow._get_code_path()}`
            - 📖 Documentation: `{workflow._get_readme_path()}`
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
                readme_content = workflow._generate_readme()
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
                st.markdown(workflow.get('analysis_plan', 'N/A'))
            
            with tab2:
                st.markdown(workflow.get('statistical_approach', 'N/A'))
            
            with tab3:
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
                st.markdown(workflow.get('review_report', 'N/A'))


def render_config_page():
    """Render configuration page"""
    st.markdown("## ⚙️ Configuration")
    
    # LLM Configuration
    st.markdown("### 🤖 LLM Endpoint")
    
    llm_config = get_llm_config()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Endpoint URL", value=llm_config.base_url, disabled=True)
        st.text_input("Model", value=llm_config.model, disabled=True)
    
    with col2:
        st.number_input("Temperature", value=llm_config.temperature, disabled=True)
        st.number_input("Max Tokens", value=llm_config.max_tokens, disabled=True)
    
    with st.expander("🔐 API Key Configuration"):
        st.markdown("""
        To configure your AIP API key, set the environment variable:
        
        ```bash
        export AIP_API_KEY="your-api-key-here"
        ```
        
        Or create a `.env` file:
        
        ```
        AIP_LLM_ENDPOINT=https://ai.aip.de/api
        AIP_API_KEY=your-api-key-here
        AIP_MODEL=llama-3-70b
        ```
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
    
    st.markdown(f"""
    - **Python Version:** {sys.version.split()[0]}
    - **Streamlit Version:** {st.__version__}
    - **Working Directory:** `{os.getcwd()}`
    """)


if __name__ == "__main__":
    main()
