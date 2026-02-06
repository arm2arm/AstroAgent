# AstroAgent 🔭

CrewAI-based Multi-Agent System for Astronomical Data Analysis

## Overview

AstroAgent is a production-ready AI-powered workflow system that automatically designs, implements, and reviews astronomical data analysis workflows. Built with CrewAI and Streamlit, it features 4 specialized agents working together to transform research questions into executable Python code.

### Key Features

- **4 Specialized AI Agents**: Planner, Analyst, Coder, and Reviewer working in sequence
- **Interactive Streamlit Dashboard**: Clean UI for workflow creation and monitoring
- **Local LLM Support**: Uses OpenAI-compatible endpoints (configured for AIP)
- **Astronomy-Focused**: Built for Gaia DR3, DR2, SDSS, and 2MASS data analysis
- **Complete Outputs**: Generates executable Python scripts with documentation

## Architecture

```
┌─────────────────────────────────────────────────┐
│         Streamlit Web Interface                 │
│  (Create Workflows, Monitor, View Results)      │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│         Workflow Orchestrator                   │
│         (CrewAI Flow with 4 Agents)             │
│                                                  │
│  Planner → Analyst → Coder → Reviewer           │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│         Local LLM Endpoint                      │
│         https://ai.aip.de/api                   │
│         (OpenAI-compatible API)                 │
└──────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.12 or higher
- Access to an OpenAI-compatible LLM endpoint
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/arm2arm/AstroAgent.git
   cd AstroAgent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and set your API credentials:
   ```bash
   AIP_LLM_ENDPOINT=https://ai.aip.de/api
   AIP_API_KEY=your-api-key-here
   AIP_MODEL=llama-3-70b
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```
   
   The dashboard will open at `http://localhost:8501`

## Usage

### Creating a Workflow

1. Navigate to the **🚀 New Workflow** page
2. Enter your research question (or select an example)
3. Choose your data source (Gaia DR3, DR2, SDSS, or 2MASS)
4. Click **🚀 Launch Workflow**
5. Watch as the 4 agents process your request:
   - **Planning**: Creates detailed analysis plan
   - **Analysis**: Designs statistical approach
   - **Coding**: Generates Python code
   - **Review**: Validates code quality

### Example Research Questions

- "Analyze the color-magnitude distribution of red giant stars in the Galactic bulge"
- "Study the proper motion distribution of stars in the solar neighborhood"
- "Investigate the relationship between stellar metallicity and kinematics"
- "Create an HR diagram for open cluster NGC 2516"

### Viewing Results

Generated workflows are saved in:
- **Python Scripts**: `outputs/workflows/workflow_[ID].py`
- **Documentation**: `outputs/workflows/README_[ID].md`

You can download files directly from the dashboard or access them from the file system.

### Workflow History

Navigate to **📊 Workflow History** to:
- View all previous workflows
- See success metrics
- Access generated code and documentation
- Review analysis plans and code reviews

## Project Structure

```
AstroAgent/
├── app.py                  # Streamlit dashboard
├── workflow.py             # CrewAI workflow orchestration
├── agents.py               # AI agent definitions
├── config.py               # Configuration management
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
├── outputs/
│   ├── workflows/          # Generated Python scripts
│   └── results/            # Analysis results
└── project.md              # Detailed specification
```

## Configuration

### LLM Settings

Configure in `.env`:

```bash
# Endpoint
AIP_LLM_ENDPOINT=https://ai.aip.de/api

# Authentication
AIP_API_KEY=your-key

# Model
AIP_MODEL=llama-3-70b

# Parameters
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=4000
LLM_TIMEOUT=120
```

### Agent Temperature Settings

Different agents use optimized temperatures:
- **Planner**: 0.4 (more creative)
- **Analyst**: 0.3 (balanced)
- **Coder**: 0.2 (precise)
- **Reviewer**: 0.3 (balanced)

### Workflow Settings

```bash
OUTPUT_DIR=outputs/workflows
RESULTS_DIR=outputs/results
MAX_RETRIES=3
VERBOSE=true
```

## Development

### Make Commands

Common tasks are available via `make`:

```bash
make build   # Install dependencies
make up      # Start Streamlit in the background
make down    # Stop Streamlit started by make up
make status  # Check Streamlit status
make clean   # Remove outputs and local run artifacts
```

`make up` writes logs to `.streamlit.log` and stores the PID in `.streamlit.pid`.

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/
```

### Code Style

This project follows PEP 8 guidelines. Format code with:

```bash
black .
isort .
```

## Troubleshooting

### LLM Connection Issues

**Problem**: Cannot connect to LLM endpoint

**Solution**: 
- Verify `AIP_LLM_ENDPOINT` and `AIP_API_KEY` in `.env`
- Test connection: `curl -H "Authorization: Bearer $AIP_API_KEY" https://ai.aip.de/api/models`

### Agent Timeouts

**Problem**: Tasks are timing out

**Solution**: Increase timeout in `.env`:
```bash
LLM_TIMEOUT=300
```

### Port Already in Use

**Problem**: Streamlit port 8501 is busy

**Solution**: Run on different port:
```bash
streamlit run app.py --server.port 8502
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- [CrewAI Documentation](https://docs.crewai.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Gaia DR3 Documentation](https://www.cosmos.esa.int/web/gaia/dr3)
- [Astropy Documentation](https://docs.astropy.org/)

## Support

For questions or issues:
- Open an issue on GitHub
- Check the [project.md](project.md) file for detailed specifications

---

**Built with ❤️ for the astronomy community**