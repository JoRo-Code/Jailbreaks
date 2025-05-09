# LLM Jailbreak Evaluation Pipeline

This repository contains a modular pipeline for evaluating different jailbreaking methods on language models. The pipeline is designed to benchmark various methods individually and in combination, tracking metrics for:

1. **Refusal** - How resistant the model is to answering harmful prompts
<!-- 2. **Utility** - Capabilities on standard benchmarks like MMLU and HellaSwag -->
<!-- 3. **Cost** - Computational expense in both fitting and inference time -->

## Setup

### 1. Install UV as package manager (instead of pip):
```sh
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create virtual environment
```sh
uv venv
```

```sh
source .venv/bin/activate
```

### 3. Install dependencies
```sh
uv pip install -r requirements.txt
```

### 4. Setup Model eligibilty via hugginface
- Head to Hugging Face and get access to downloading models
e.g. [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) or [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

- export `HUGGINGFACE_HUB_TOKEN`

### 5. Login to weights and biases (will prompt you when you run the pipeline)

### 6. (Optional) Inference APIs
- if utilizing non-local LLM-judges for evaluation. Specify in `main.py`
    - export `GOOGLE_API_KEY` and/or `GROQ_API_KEY`

## Usage

### Run
```sh
uv run main.py
```
Run pipeline with new models, downloads models and tokenizers (first time)
```sh
uv run main.py --download
```

### Specific steps

```sh
uv run main.py --mode fit

uv run main.py --mode generate
uv run main.py --mode generate --project-name jailbreaks

seq 5 | xargs -n1 -I{} uv run main.py --mode generate project-name jailbreaks

uv run main.py --mode evaluate
uv run main.py --mode evaluate --eval-run-id baseline --project-name jailbreaks --use-local

uv run main.py --mode aggregate
uv run main.py --mode aggregate --eval-run-id baseline --project-name jailbreaks --use-local
```

### TODO:
- Utility
- Methods
    - [x] baseline
    - [x] output-aware
    - [x] prefix/suffix injections
    - [x] ablation (diff-in-means)
        - [x] dynamic fitting for new models (finds best direction)
- Refusal:
    - [x] AdvBench
    - [ ] LLM evaluator
        - Quality
        - Refusal
- Models:
    - [x] qwen2.5
    - [x] phi-4
    - [x] gemma-2
    - [x] llama-3

## License

[MIT License](LICENSE)
