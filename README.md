# LLM Jailbreak Evaluation Pipeline

A comprehensive toolkit for evaluating jailbreaking methods on large language models (LLMs). This pipeline allows researchers to systematically test how various jailbreaking techniques affect model safety, utility, and computational efficiency.

## 🎯 What This Does

This pipeline helps you:
- **Test Model Safety**: Evaluate how resistant models are to harmful prompts (refusal rate)
- **Measure Utility**: Assess model performance on standard benchmarks (MMLU, HellaSwag)
- **Track Costs**: Monitor computational expenses for both training and inference
- **Compare Methods**: Benchmark multiple jailbreaking techniques individually or in combination

## 📋 Prerequisites

Before you start, make sure you have:
- Python 3.8 or higher
- macOS or Linux (Windows support may vary)
- GPU with 40GB+ VRAM (optional, for faster local inference)

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd jailbreaks

# Install UV package manager (faster than pip)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Configure Access Tokens

You'll need accounts and API keys for:

**Hugging Face (Required)**:
```bash
# Get your token from https://huggingface.co/settings/tokens
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

**Weights & Biases (Required for tracking)**:
```bash
# You'll be prompted to login when you first run the pipeline
# Get your API key from https://wandb.ai/authorize
```

**External API Keys (Optional)**:
```bash
# Only needed if using external LLM judges for evaluation
export GOOGLE_API_KEY="your_google_api_key"
export GROQ_API_KEY="your_groq_api_key"
```

### 3. Request Model Access

Visit Hugging Face and request access to the models you want to test:
- [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- Or any other models you plan to use

### 4. Download Models

```bash
# Configure which models to download in main.py, then:
uv run main.py --mode download
```

## 📖 Complete Workflow

### Step 1: Fit Jailbreaking Methods
Prepare and train the jailbreaking methods on your selected models:

```bash
uv run main.py --mode fit
```

**Options**:
- Add `--refit` to retrain already fitted methods

### Step 2: Generate Responses
Test the jailbreaking methods against your chosen datasets:

```bash
# Basic generation
uv run main.py --mode generate --project-name my_experiment --benchmark advbench
```

**Available Benchmarks**:
- `advbench`
- `malicious`

**Where Results Go**:
- Local files: `results/` directory
- Online tracking: Weights & Biases dashboard (link provided in terminal)

### Step 3: Evaluate Responses
Assess the quality and safety of generated responses:

```bash
uv run main.py --mode evaluate \
  --project-name advbench_experiment_v1 \
  --eval-run-id deepseek \
  --use-local
```

**Configuration**: Set up evaluators in `main.py`:
```python
QualityEvaluator(
    judge=GroqLLMJudge(model="deepseek-r1-distill-llama-70b"), 
    name="deepseek-v1"
)
```

**Options**:
- `--use-local`: Use local data instead of downloading from Weights & Biases
- `--eval-run-id`: Continue from a previous evaluation run

### Step 4: Analyze Results
Generate comprehensive reports using the provided notebooks:

1. Open `report_utils/aggregate_safety.ipynb`
2. Configure your datasets:
```python
datasets = [
    {"project_name": "advbench_v1", "eval_run_id": "deepseek-v1"},
    {"project_name": "malicious-instruct-v2", "eval_run_id": "deepseek-v1"},
]
```
3. Run the notebook to generate aggregated analysis

## 🔧 Configuration

### Customizing Experiments

The main configuration happens in `main.py`. Key settings include:

- **Models**: Which LLMs to test
- **Methods**: Which jailbreaking techniques to apply
- **Method Combos**: Sequences of jailbreaking methods to chain together
- **Generation Parameters**: Token limits, temperature, etc.
- **Evaluators**: How to judge response quality and safety

### Running Multiple Experiments

```bash
# Run the same experiment 5 times for statistical significance
seq 5 | xargs -n1 -I{} uv run main.py --mode generate --project-name my_experiment

# Auto-shutdown Lambda instances after completion (if using https://cloud.lambda.ai/)
seq 5 | xargs -n1 -I{} uv run main.py --mode generate --project-name my_experiment; 
uv run lambda.py
```

## 🛠️ Extending the Pipeline

### Adding New Models
1. Add tokenization templates in `utils/tokenization.py`
2. Ensure the model is supported by `transformerlens` (for certain manipulations)
3. Update model configurations in `main.py`

### Adding New Jailbreaking Methods
1. Create new method classes in `methods/`
2. Extend the appropriate `base_method` class
3. Implement required methods for your technique

### Adding New Evaluators
1. Create evaluator classes in `evaluators/`
2. Extend the `ResponseEvaluator` base class
3. Configure different LLM providers and evaluation prompts

## 🔍 Troubleshooting

**Common Issues**:

- **"Model not found"**: Ensure you have access to the model on Hugging Face, your token is set and model and tokenizer is downloaded. 
- **Out of memory**: Reduce batch size in `main.py` or use smaller models
- **Slow generation**: Consider using GPU acceleration or external APIs
- **Wandb login issues**: Run `wandb login` manually with your API key

## 📊 Understanding Results

**Metrics Tracked**:
- **Refusal Rate**: Percentage of harmful prompts the model refuses to answer
- **Utility Scores**: Performance on standard benchmarks (MMLU, HellaSwag)
- **Cost Metrics**: Computational time and resource usage
- **Jailbreak Success Rate**: How often jailbreaking methods succeed

**Output Locations**:
- Raw responses: `results/` directory
- Evaluation scores: Weights & Biases dashboard
- Aggregated reports: Generated notebooks in `report_utils/`

## 🗺️ Roadmap

- [ ] Hydra configuration system
- [ ] Automatic config upload to Weights & Biases
- [ ] Additional jailbreaking methods
- [ ] Support for more model architectures
- [ ] Docker containerization
- [ ] Distributed evaluation support

## 📄 License

[MIT License](LICENSE)
