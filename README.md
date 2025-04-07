# LLM Jailbreak Evaluation Pipeline

This repository contains a modular pipeline for evaluating different jailbreaking methods on language models. The pipeline is designed to benchmark various methods individually and in combination, tracking metrics for:

1. **Refusal** - How resistant the model is to answering harmful prompts
2. **Utility** - Capabilities on standard benchmarks like MMLU and HellaSwag
<!-- 3. **Cost** - Computational expense in both fitting and inference time -->

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

4. 
```sh
cd jailbreak
```

5. Set up MLflow tracking:
```sh
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```
5. Secrets
populate `.env` with your `HUGGINGFACE_HUB_TOKEN`
```sh
export $(cat .env | xargs)
```
6. Head to Hugging Face and get access to downloading models
e.g. [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) or [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

## Usage
```sh
uv run jailbreaks/main.py
```

## Visualization

### MLflow Tracking UI

Results are tracked in MLflow and can be viewed through the MLflow UI:

```bash
mlflow ui --port 5001
```

Navigate to http://localhost:5001 to explore experiment results. 

### TODO:
- Utility benchmark
    - [x] MMLU
    - [x] hellaswag
    - Should prompts and answers be passed in a chat interface? Applying the chat-interface also adds system prompts. It is probably alright cause all this benchmark is trying to do is to evaluate the change when using methods
    - [ ] handle when no answer is found, should be quantified. Would help debug the extraction and quantify the error rate. 
- Methods
    - [x] output aware generation exploit
    - [x] GCG fitting
        - Alternative: fit on model with hooks (ablated model)
    - [x] diff-in-means single direction
- Refusal Benchmark 
    - [x] simple "include"-classifier
    - [ ] Refusal scorer (LLM/classifier):
        - [ ] Refusal score
        - [ ] Response Quality
    - [x] logging to unique runs (model & method combinations)
- Models
    - [x] mistralai/Mistral-7B-Instruct-v0.1
    - [x] Qwen/Qwen2.5-7B-Instruct
    - [x] meta-llama/Llama-3.1-8B-Instruct

- [ ] easy extraction of results
- [ ] fitting time

## Issues
- GCG: Qwen/Qwen2.0-0.5B-Instruct crashes midway fitting, on iteration 358/500. 

## License

[MIT License](LICENSE)
