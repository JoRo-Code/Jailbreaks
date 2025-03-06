# LLM Jailbreak Evaluation Pipeline

This repository contains a modular pipeline for evaluating different jailbreaking methods on language models. The pipeline is designed to benchmark various methods individually and in combination, tracking metrics for:

1. **Utility** - How well the model retains its capabilities on standard benchmarks
2. **Refusal** - How resistant the model is to answering harmful prompts
3. **Cost** - Computational expense in both fitting and inference time

## Setup

1. Install UV as package manager (instead of pip):
```sh
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create virtual environment
```sh
uv venv
```

```sh
source .venv/bin/activate
```

3. Install dependencies
```sh
uv pip install -r requirements.txt
```

4. Set up MLflow tracking:
```sh
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```
5. Secrets
populate `.env` with your `HUGGINGFACE_HUB_TOKEN`
```sh
export $(cat .env | xargs)
```
6. Head to Hugging Face and get access to downloading models
e.g. [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b)

## Usage

### Running the Pipeline

The easiest way to run the pipeline is using the provided shell script:

```bash
# Make the script executable
chmod +x run_pipeline.sh

# Run with default settings
./run_pipeline.sh

# See all available options
./run_pipeline.sh --help
```

This script will:
1. Start an MLflow server on the specified port
2. Run the evaluation pipeline with your settings
3. Keep the MLflow server running so you can explore results

You can customize the run with various options:

```bash
# Evaluate specific methods on a custom model
./run_pipeline.sh --model gpt2 --methods gcg,token_aware --experiment custom-test

# Run with detailed debug logging
./run_pipeline.sh --log-level DEBUG
```

### Basic Pipeline Configuration

```python
from jailbreak_pipeline import (
    JailbreakPipeline, 
    GCG, 
    SingleDirection, 
    TokenAware,
    BaselineRefusalScorer
)
from transformers import AutoModelForCausalLM

# Load your model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# Configure jailbreak methods
gcg = GCG(target_sentence="I'll help with that.")
single_direction = SingleDirection()
token_aware = TokenAware(refusal_scorer=BaselineRefusalScorer())

# Set up pipeline
pipeline = JailbreakPipeline(
    models=[model],
    method_combinations=[
        [],  # Baseline (no methods)
        [gcg],  # Individual methods
        [single_direction],
        [token_aware],
        [gcg, single_direction, token_aware]  # Combined
    ],
    benchmarks={
        "utility": UtilityBenchmark(datasets=["MMLU", "HellaSwag"]),
        "refusal": RefusalBenchmark(prompts=["your harmful prompts here"]),
        "cost": CostBenchmark()
    }
)

# Run evaluation
results = pipeline.run(experiment_name="your_experiment_name")
```

### Running Tests

To run the test suite:

```bash
python -m pytest test_pipeline.py
```

## Extending the Pipeline

### Adding New Jailbreak Methods

To add a new jailbreak method, create a class that inherits from `JailbreakMethod`:

```python
class YourMethod(JailbreakMethod):
    name = "your_method"
    
    def __init__(self, your_params):
        super().__init__()
        self.your_params = your_params
        
    def fit(self, model, training_data):
        # Implementation for fitting your method
        self.fit_time = measured_time
        return self
        
    def apply(self, model, prompt):
        # Implementation for applying your method
        return modified_prompt
```

### Adding New Benchmarks

To add a new benchmark, create a class with an `evaluate` method that returns a `BenchmarkResult`:

```python
class YourBenchmark:
    def __init__(self, your_params):
        self.your_params = your_params
    
    def evaluate(self, model, methods=None):
        # Implementation for evaluation
        return BenchmarkResult(
            name="your_benchmark",
            score=your_score,
            details={"your_details": values}
        )
```

## Visualization

Results are tracked in MLflow and can be visualized through the MLflow UI:

```bash
mlflow ui --port 5000
```
You can adjust the port accordingly.

Navigate to http://localhost:5000 to explore experiment results.

## License

[MIT License](LICENSE)