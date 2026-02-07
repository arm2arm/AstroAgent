# Quick Start Guide

## WorkflowAgent - CrewAI Multi-Agent Workflows

### Prerequisites

1. **Python 3.12+** installed
2. **Access to any OpenAI-compatible /v1 LLM endpoint** (Ollama, vLLM, LiteLLM, etc.)
3. **pip** package manager

### Installation Steps

#### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/arm2arm/AstroAgent.git
cd AstroAgent

# Install dependencies
pip install -r requirements.txt
```

#### 2. Configure Environment

Copy the example environment file and edit it:

```bash
cp .env.example .env
```

Edit `.env` with your endpoint:

```env
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=
LLM_MODEL=qwen3-coder:latest

# Optional: override model for the Executor agent
EXECUTOR_LLM_MODEL=

# Optional: embeddings endpoint (separate from LLM)
EMBED_MODEL=nomic-embed-text:latest
EMBED_PROVIDER=ollama
EMBED_BASE_URL=
EMBED_API_KEY=
```

Leave `LLM_API_KEY` empty for endpoints that do not require authentication (e.g. local Ollama).

#### Optional: Use Makefile Shortcuts

Run `make` with no arguments to see all available commands:

```bash
make           # Show help with all available commands
make build     # Install Python dependencies
make up        # Start Streamlit server in the background
make down      # Stop Streamlit server
make logs      # Tail the Streamlit log file
make status    # Show Streamlit process status
make restart   # Stop + start Streamlit server
make clean     # Remove outputs and local artifacts
```

`make up` writes logs to `.streamlit.log` and stores the PID in `.streamlit.pid`.

#### 3. Test Installation

Run the component test:

```bash
python3 test_components.py
```

You should see: ✅ All tests passed!

#### 4. Start the Application

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

### Usage

#### Creating Your First Workflow

1. **Navigate** to 🚀 New Workflow page (default)

2. **Enter a task request**, for example:
   - "Plot sin(x) and save the figure"
   - Or click one of the example tasks

4. **Click** "🚀 Launch Workflow"

5. **Watch** the 4-phase process:
   - 📋 **Planning**: Creates analysis plan
   - 📊 **Analysis**: Designs statistical approach
   - 💻 **Coding**: Generates Python code
   - ✅ **Review**: Validates code quality

6. **Download** generated files:
   - Python script (`.py`)
   - Documentation (`.md`)

#### Viewing Workflow History

1. Navigate to 📊 **Workflow History**
2. View all past workflows with success metrics
3. Expand any workflow to see:
   - Analysis plan
   - Statistical approach
   - Generated code
   - Code review

#### Checking Configuration

1. Navigate to ⚙️ **Configuration**
2. View current settings:
   - LLM endpoint and model (including discovered models)
   - Workflow directories
   - System information

### Example Tasks

Examples are loaded from `example_tasks/*.yaml`.

1. **Sine Plot**:
   ```
   Plot sin(x) in Python and save the figure
   ```

2. **CSV Summary**:
   ```
   Load a CSV and summarize columns with basic stats
   ```

3. **API Data**:
   ```
   Fetch JSON from an API and visualize a key metric
   ```

### Output Files

Workflows generate two files in `outputs/workflows/`:

1. **`workflow_[ID].py`**: Executable Python script
   - Complete analysis implementation
   - Uses pandas, matplotlib (and any libraries you request)
   - Includes error handling

2. **`README_[ID].md`**: Documentation
   - Task request
   - Analysis plan
   - Statistical approach
   - Usage instructions
   - Code review feedback

### Troubleshooting

#### Connection Issues

If you see LLM connection errors:

```bash
# Test your API endpoint
curl $LLM_BASE_URL/models
```

Verify your `.env` file has the correct credentials.

#### Port Already in Use

If port 8501 is busy:

```bash
streamlit run app.py --server.port 8502
```

#### Timeout Issues

For complex queries, increase timeout in `.env`:

```env
LLM_TIMEOUT=300
```

### Advanced Options

When creating a workflow, expand **⚙️ Advanced Options** to adjust:

- **LLM Temperature**: 0.0-1.0 (default: 0.3)
  - Lower = more focused/deterministic
  - Higher = more creative/varied

- **Max Retries**: 1-5 (default: 3)
  - Number of retry attempts for failed tasks

### Project Structure

```
AstroAgent/
├── app.py              # Streamlit dashboard
├── workflow.py         # CrewAI workflow orchestration
├── agents.py           # 4 specialized AI agents
├── config.py           # Configuration management
├── requirements.txt    # Dependencies
├── .env               # Your configuration (create from .env.example)
├── example_tasks/      # YAML example tasks
├── .streamlit/         # Streamlit config
├── test_components.py  # Component tests
└── outputs/
    ├── workflows/      # Generated Python scripts
    └── results/        # Analysis results
```

### Getting Help

- **GitHub Issues**: Report bugs or request features
- **Documentation**: See [README.md](README.md) for details
- **Project Spec**: See [project.md](project.md) for architecture

### Next Steps

1. ✅ Create your first workflow
2. ✅ Review the generated code
3. ✅ Run the Python script (install any required libraries)
4. ✅ Use the outputs in your project

---

**Happy Building!**
