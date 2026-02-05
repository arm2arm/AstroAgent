"""
CrewAI Workflow for Astronomy Data Analysis
"""
from crewai import Crew, Task, Process
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import datetime
import uuid
import os

from agents import (
    create_planner_agent,
    create_analyst_agent,
    create_coder_agent,
    create_reviewer_agent
)
from config import get_workflow_config


class WorkflowState(BaseModel):
    """State management for astronomy workflow"""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # User inputs
    research_question: str = ""
    data_source: str = "gaia_dr3"
    
    # Agent outputs
    analysis_plan: Optional[str] = None
    statistical_approach: Optional[str] = None
    generated_code: Optional[str] = None
    review_report: Optional[str] = None
    
    # Status
    status: str = "initialized"
    error: Optional[str] = None


class AstronomyWorkflow(Flow[WorkflowState]):
    """
    Main workflow for astronomy data analysis
    Simple 4-step process: Plan → Analyze → Code → Review
    """
    
    def __init__(self, state: WorkflowState):
        super().__init__(state)
        self.config = get_workflow_config()
        
        # Initialize agents
        self.planner = create_planner_agent()
        self.analyst = create_analyst_agent()
        self.coder = create_coder_agent()
        self.reviewer = create_reviewer_agent()

    @start()
    def initialize(self) -> Dict:
        """Entry point: Initialize workflow"""
        print(f"\n🚀 Starting workflow {self.state.workflow_id}")
        print(f"Research Question: {self.state.research_question}")
        
        self.state.status = "planning"
        
        return {
            "research_question": self.state.research_question,
            "data_source": self.state.data_source
        }

    @listen(initialize)
    def planning_phase(self, context: Dict) -> Dict:
        """Phase 1: Create analysis plan"""
        print("\n📋 Phase 1: Planning Analysis...")
        
        planning_task = Task(
            description=f"""
            Create a detailed analysis plan for this research question:
            "{context['research_question']}"
            
            Data source: {context['data_source']}
            
            Your plan should include:
            1. Data selection criteria (which columns, filters)
            2. Sample size considerations
            3. Key analysis steps
            4. Expected outputs
            
            Be specific and practical.
            """,
            expected_output="Detailed analysis plan with clear steps",
            agent=self.planner
        )
        
        crew = Crew(
            agents=[self.planner],
            tasks=[planning_task],
            process=Process.sequential,
            verbose=self.config.verbose
        )
        
        result = crew.kickoff(inputs=context)
        self.state.analysis_plan = result.raw
        self.state.status = "analyzing"
        
        return {"analysis_plan": self.state.analysis_plan}

    @listen(planning_phase)
    def analysis_phase(self, planning_result: Dict) -> Dict:
        """Phase 2: Design statistical approach"""
        print("\n📊 Phase 2: Designing Statistical Analysis...")
        
        analysis_task = Task(
            description=f"""
            Design the statistical analysis approach for this plan:
            
            {planning_result['analysis_plan']}
            
            Specify:
            1. Statistical methods to use
            2. Visualization strategies (plots, diagrams)
            3. Quality checks and validation
            4. Expected insights
            
            Focus on practical astronomy analysis.
            """,
            expected_output="Statistical analysis strategy with methods",
            agent=self.analyst
        )
        
        crew = Crew(
            agents=[self.analyst],
            tasks=[analysis_task],
            process=Process.sequential,
            verbose=self.config.verbose
        )
        
        result = crew.kickoff(inputs=planning_result)
        self.state.statistical_approach = result.raw
        self.state.status = "coding"
        
        return {"statistical_approach": self.state.statistical_approach}

    @listen(analysis_phase)
    def coding_phase(self, analysis_result: Dict) -> Dict:
        """Phase 3: Generate Python code"""
        print("\n💻 Phase 3: Generating Python Code...")
        
        coding_task = Task(
            description=f"""
            Generate complete Python code implementing this analysis:
            
            Plan: {self.state.analysis_plan}
            
            Methods: {analysis_result['statistical_approach']}
            
            Requirements:
            - Use astropy, pandas, matplotlib, numpy
            - Include proper error handling
            - Add clear comments
            - Save plots and results
            - Make it executable as a script
            
            Generate ONLY the Python code, ready to run.
            """,
            expected_output="Complete executable Python script",
            agent=self.coder
        )
        
        crew = Crew(
            agents=[self.coder],
            tasks=[coding_task],
            process=Process.sequential,
            verbose=self.config.verbose
        )
        
        result = crew.kickoff(inputs=analysis_result)
        self.state.generated_code = result.raw
        self.state.status = "reviewing"
        
        # Save code to file
        self._save_code()
        
        return {"generated_code": self.state.generated_code}

    @listen(coding_phase)
    def review_phase(self, coding_result: Dict) -> Dict:
        """Phase 4: Review code quality"""
        print("\n✅ Phase 4: Reviewing Code...")
        
        review_task = Task(
            description=f"""
            Review this Python code for quality and correctness:
            
            {coding_result['generated_code']}
            
            Check:
            1. Syntax and logic errors
            2. Error handling
            3. Code quality and style
            4. Scientific correctness
            5. Potential issues
            
            Provide constructive feedback.
            """,
            expected_output="Code review report with feedback",
            agent=self.reviewer
        )
        
        crew = Crew(
            agents=[self.reviewer],
            tasks=[review_task],
            process=Process.sequential,
            verbose=self.config.verbose
        )
        
        result = crew.kickoff(inputs=coding_result)
        self.state.review_report = result.raw
        self.state.status = "completed"
        
        print(f"\n✨ Workflow {self.state.workflow_id} completed!")
        
        return {
            "workflow_id": self.state.workflow_id,
            "status": "completed",
            "code_file": self.get_code_path()
        }

    def _save_code(self):
        """Save generated code and README to files"""
        # Save Python script
        code_filepath = self.get_code_path()
        os.makedirs(os.path.dirname(code_filepath), exist_ok=True)
        
        with open(code_filepath, 'w') as f:
            f.write(self.state.generated_code)
        
        print(f"💾 Code saved to: {code_filepath}")
        
        # Save README.md
        readme_filepath = self.get_readme_path()
        readme_content = self.generate_readme()
        
        with open(readme_filepath, 'w') as f:
            f.write(readme_content)
        
        print(f"📄 README saved to: {readme_filepath}")

    def get_code_path(self) -> str:
        """Get path for code file"""
        config = get_workflow_config()
        return os.path.join(
            config.output_dir,
            f"workflow_{self.state.workflow_id}.py"
        )

    def get_readme_path(self) -> str:
        """Get path for README file"""
        config = get_workflow_config()
        return os.path.join(
            config.output_dir,
            f"README_{self.state.workflow_id}.md"
        )

    def generate_readme(self) -> str:
        """Generate README.md for workflow"""
        return f"""# Astronomy Workflow {self.state.workflow_id}

Generated: {self.state.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Data Source: {self.state.data_source}

________________________________________

## Research Question

{self.state.research_question}

________________________________________

## Analysis Plan

{self.state.analysis_plan}

________________________________________

## Statistical Approach

{self.state.statistical_approach}

________________________________________

## Generated Code

The analysis is implemented in `workflow_{self.state.workflow_id}.py`

### Requirements

Install dependencies:

```bash
pip install astropy pandas matplotlib numpy
```

### Usage

Run the analysis:

```bash
python workflow_{self.state.workflow_id}.py
```

### Expected Outputs

The script will generate:
- Data analysis results
- Visualization plots
- Statistical summaries

________________________________________

## Code Review

{self.state.review_report}

________________________________________

## File Structure

```
workflow_{self.state.workflow_id}.py     # Main analysis script
README_{self.state.workflow_id}.md       # This documentation
```

________________________________________

## Notes

- Ensure you have access to {self.state.data_source} data
- Results will be saved in the results/ directory
- Check the code review section above for any recommendations

________________________________________

Generated by CrewAI Astronomy Workflow System
"""


def run_workflow(research_question: str, data_source: str = "gaia_dr3") -> Dict:
    """
    Convenience function to run complete workflow
    
    Args:
        research_question: Research question to analyze
        data_source: Data source (default: gaia_dr3)
        
    Returns:
        Dictionary with workflow results
    """
    state = WorkflowState(
        research_question=research_question,
        data_source=data_source
    )
    
    workflow = AstronomyWorkflow(state=state)
    result = workflow.kickoff()
    
    return {
        "workflow_id": state.workflow_id,
        "status": state.status,
        "analysis_plan": state.analysis_plan,
        "statistical_approach": state.statistical_approach,
        "generated_code": state.generated_code,
        "review_report": state.review_report,
        "code_file": workflow.get_code_path()
    }
