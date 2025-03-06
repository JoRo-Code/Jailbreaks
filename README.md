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
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

## Usage

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
mlflow ui
```

Navigate to http://localhost:5000 to explore experiment results.

## License

[MIT License](LICENSE)