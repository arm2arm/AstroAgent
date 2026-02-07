#!/usr/bin/env python3
"""
Simple test script to verify AstroAgent components
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
        from config import get_llm_config, get_workflow_config, init_directories
        
        llm_config = get_llm_config()
        print(f"  LLM Endpoint: {llm_config.base_url}")
        print(f"  Model: {llm_config.model}")
        
        workflow_config = get_workflow_config()
        print(f"  Output Directory: {workflow_config.output_dir}")
        print(f"  Results Directory: {workflow_config.results_dir}")
        
        init_directories()
        print("  Directories initialized")
        
        print("✅ Configuration works correctly")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_agents():
    """Test agent creation"""
    print("\nTesting agents...")
    try:
        from agents import (
            create_planner_agent,
            create_analyst_agent,
            create_coder_agent,
            create_executor_agent,
            create_reviewer_agent,
        )
        
        # Note: We don't actually instantiate agents here as they need LLM connection
        # Just verify the functions exist and are callable
        print("  Planner agent function: OK")
        print("  Analyst agent function: OK")
        print("  Coder agent function: OK")
        print("  Executor agent function: OK")
        print("  Reviewer agent function: OK")
        
        print("✅ Agent definitions are correct")
        return True
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
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
        import os
        
        required_dirs = [
            "outputs/workflows",
            "outputs/results"
        ]
        
        for dir_path in required_dirs:
            if os.path.exists(dir_path):
                print(f"  ✓ {dir_path}")
            else:
                print(f"  ✗ {dir_path} (missing)")
                return False
        
        print("✅ Directory structure is correct")
        return True
    except Exception as e:
        print(f"❌ Directory structure test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("AstroAgent Component Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config,
        test_agents,
        test_workflow_state,
        test_directory_structure
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
        print("  1. Configure your API key in .env")
        print("  2. Run: streamlit run app.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
