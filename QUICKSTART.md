# Quick Start Guide

## AstroAgent - CrewAI Astronomy Workflows

### Prerequisites

1. **Python 3.12+** installed
2. **Access to an OpenAI-compatible LLM endpoint** (e.g., AIP endpoint)
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

Edit `.env` with your credentials:

```env
AIP_LLM_ENDPOINT=https://ai.aip.de/api
AIP_API_KEY=your-actual-api-key-here
AIP_MODEL=llama-3-70b
```

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

2. **Enter a research question**, for example:
   - "Analyze the color-magnitude distribution of red giant stars in the Galactic bulge"
   - Or click one of the example questions

3. **Select data source**: Gaia DR3 (default), DR2, SDSS, or 2MASS

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
   - LLM endpoint and model
   - Workflow directories
   - System information

### Example Research Questions

Try these astronomy-focused questions:

1. **Stellar Analysis**:
   ```
   Analyze the color-magnitude distribution of red giant stars 
   in the Galactic bulge using Gaia DR3 data
   ```

2. **Kinematics**:
   ```
   Study the proper motion distribution of stars in the 
   solar neighborhood within 100 parsecs
   ```

3. **HR Diagram**:
   ```
   Create an HR diagram for open cluster NGC 2516 and 
   identify main sequence, giants, and white dwarfs
   ```

4. **Metallicity Studies**:
   ```
   Investigate the relationship between stellar metallicity 
   and kinematics for disk stars
   ```

### Output Files

Workflows generate two files in `outputs/workflows/`:

1. **`workflow_[ID].py`**: Executable Python script
   - Complete analysis implementation
   - Uses astropy, pandas, matplotlib
   - Includes error handling

2. **`README_[ID].md`**: Documentation
   - Research question
   - Analysis plan
   - Statistical approach
   - Usage instructions
   - Code review feedback

### Troubleshooting

#### Connection Issues

If you see LLM connection errors:

```bash
# Test your API endpoint
curl -H "Authorization: Bearer $AIP_API_KEY" https://ai.aip.de/api/models
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
3. ✅ Run the Python script (after installing astropy, etc.)
4. ✅ Analyze real astronomical data!

---

**Happy Analyzing! 🔭**
