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


## Visualization and Reporting

### MLflow Tracking UI

Results are tracked in MLflow and can be viewed through the MLflow UI:

```bash
mlflow ui --port 5001
```

Navigate to http://localhost:5001 to explore experiment results. Each run includes:
- All configuration parameters
- Metrics for utility and refusal
- Detailed refusal benchmark results in CSV format
- HTML summary tables of refusal data

## License

[MIT License](LICENSE)