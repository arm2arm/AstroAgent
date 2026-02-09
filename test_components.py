#!/usr/bin/env python3
"""
Test script for AstroAgent components (CrewAI 1.9x YAML config)
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all module imports"""
    print("Testing imports...")
    try:
        import config
        import agents
        import workflow
        print("✅ All modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
    try:
        from config import get_llm_config, get_workflow_config, get_storage_config, init_directories

        llm_config = get_llm_config()
        print(f"  LLM Endpoint: {llm_config.base_url}")
        print(f"  Model: {llm_config.model}")

        workflow_config = get_workflow_config()
        print(f"  Output Directory: {workflow_config.output_dir}")
        print(f"  Results Directory: {workflow_config.results_dir}")

        storage_config = get_storage_config()
        print(f"  Memory Enabled: {storage_config.enabled}")
        print(f"  Memory DB Path: {storage_config.db_path}")

        init_directories()
        print("  Directories initialized")

        print("✅ Configuration works correctly")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_yaml_configs():
    """Test YAML config loading"""
    print("\nTesting YAML configs...")
    try:
        from agents import load_agents_config, load_tasks_config

        agents_cfg = load_agents_config()
        expected_agents = {"planner", "analyst", "coder", "executor", "reviewer", "summarizer"}
        found_agents = set(agents_cfg.keys())
        missing = expected_agents - found_agents
        if missing:
            print(f"❌ Missing agents in YAML: {missing}")
            return False
        print(f"  Agents defined: {', '.join(sorted(found_agents))}")

        for key in expected_agents:
            cfg = agents_cfg[key]
            for field in ("role", "goal", "backstory"):
                if field not in cfg:
                    print(f"❌ Agent '{key}' missing field '{field}'")
                    return False

        tasks_cfg = load_tasks_config()
        expected_tasks = {"planning_task", "analysis_task", "coding_task", "review_task", "summarization_task"}
        found_tasks = set(tasks_cfg.keys())
        missing_tasks = expected_tasks - found_tasks
        if missing_tasks:
            print(f"❌ Missing tasks in YAML: {missing_tasks}")
            return False
        print(f"  Tasks defined: {', '.join(sorted(found_tasks))}")

        print("✅ YAML configs are valid")
        return True
    except Exception as e:
        print(f"❌ YAML config test failed: {e}")
        return False


def test_agents():
    """Test agent factory"""
    print("\nTesting agent factory...")
    try:
        from agents import AgentFactory, get_task_template

        factory = AgentFactory()
        print("  AgentFactory created: OK")

        # Verify factory methods exist
        for name in ("planner", "analyst", "coder", "executor", "reviewer", "summarizer"):
            assert hasattr(factory, name), f"Missing method: {name}"
            print(f"  {name}() method: OK")

        # Verify task template helper
        tmpl = get_task_template("coding_task")
        assert "description" in tmpl
        assert "expected_output" in tmpl
        print("  get_task_template(): OK")

        print("✅ Agent factory definitions are correct")
        return True
    except Exception as e:
        print(f"❌ Agent factory test failed: {e}")
        return False


def test_workflow_state():
    """Test workflow state"""
    print("\nTesting workflow state...")
    try:
        from workflow import WorkflowState

        state = WorkflowState(
            research_question="Test question"
        )

        print(f"  Workflow ID: {state.workflow_id}")
        print(f"  Research Question: {state.research_question}")
        print(f"  Status: {state.status}")

        print("✅ Workflow state works correctly")
        return True
    except Exception as e:
        print(f"❌ Workflow state test failed: {e}")
        return False


def test_directory_structure():
    """Test directory structure"""
    print("\nTesting directory structure...")
    try:
        required_dirs = [
            "outputs/workflows",
            "outputs/results",
        ]
        required_files = [
            "config/agents.yaml",
            "config/tasks.yaml",
        ]

        for dir_path in required_dirs:
            if os.path.exists(dir_path):
                print(f"  ✓ {dir_path}")
            else:
                print(f"  ✗ {dir_path} (missing)")
                return False

        for file_path in required_files:
            if os.path.isfile(file_path):
                print(f"  ✓ {file_path}")
            else:
                print(f"  ✗ {file_path} (missing)")
                return False

        print("✅ Directory structure is correct")
        return True
    except Exception as e:
        print(f"❌ Directory structure test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("AstroAgent Component Tests (CrewAI 1.9x YAML)")
    print("=" * 60)

    tests = [
        test_imports,
        test_config,
        test_yaml_configs,
        test_agents,
        test_workflow_state,
        test_directory_structure,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("\n🎉 All tests passed! AstroAgent is ready to use.")
        print("\nTo start the application:")
        print("  1. Configure your LLM in .env")
        print("  2. Run: streamlit run app.py")
        print("\nTo customize agents: edit config/agents.yaml")
        print("To customize tasks:  edit config/tasks.yaml")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
