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

uv run main.py --mode evaluate
uv run main.py --mode evaluate --eval-run-id abcd0123 --project-name jailbreaks 

uv run main.py --mode aggregate
uv run main.py --mode aggregate --eval-run-id abcd0123 --project-name jailbreaks
```

### TODO:
- Utility
- Methods
    - [x] baseline
    - [x] output-aware - takes lots of time
    - [x] prefix/suffix injections
    - [x] diff-in-means
        - [x] dynamic fitting for new models (finds best direction)
    - [ ] GCG - errors
- Refusal:
    - [x] AdvBench
    - [ ] LLM evaluator
        - Quality
        - Refusal
- Models:
    - [x] Qwen/Qwen2-0.5B-Instruct
    - [x] Qwen/Qwen2.5-1.5B-Instruct
    - [x] Qwen/Qwen2.5-3B-Instruct
    - [x] Qwen/Qwen2.5-7B-Instruct
    - [x] mistralai/Mistral-7B-Instruct-v0.1
    - [x] meta-llama/Llama-3.1-8B-Instruct
    (works for bigger models too but runs out of VRAM on 40Gb A100. diff-in-means requires 30GB on 7B models, while baseline 14B requires 30GB)

## Issues
- GCG

## License

[MIT License](LICENSE)
