model_map = {
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "gemma-2-9b-it": "Gemma-2-9B",
    "phi-4": "Phi-4-14B",
}

method_map = {
    "diffinmeans": "Single Refusal Direction Ablation",
    "prefixinjection-3": "Prefix Injection",
    "baseline": "Baseline",
    "outputaware-temperature": "Output-Aware Temperature Sampling",
    "hiddenlayerinjection": "Config Prompt Injection",

}

safety_metric_map = {
    "ASR"        : {"name":"Attack Success Rate", "unit":"%", "ylim":(0, 100)},
    "unethical"  : {"name":"Unethical Percentage", "unit":"%", "ylim":(0, 100)},
    "harmful"    : {"name":"Harmful Response Quality (≥ 0)", "unit":"%", "ylim":(0, 100)},
    "harmful_1"  : {"name":"Harmful Response Quality (≥ 1)", "unit":"%", "ylim":(0, 100)},
    "harmful_2"  : {"name":"Harmful Response Quality (≥ 2)", "unit":"%", "ylim":(0, 100)},
    "harmful_3"  : {"name":"Harmful Response Quality", "unit":"%", "ylim":(0, 100)},
    "refusal"    : {"name":"Refusal Rate", "unit":"%", "ylim":(0, 100)},
    "ethical"    : {"name":"Ethical Rate", "unit":"%", "ylim":(0, 100)},
    "depth"      : {"name":"Quality Score", "unit":"Score", "ylim":(0, 3)},
}

heat_map_font_sizes = {
    "xlabel": 16,
    "ylabel": 16,
    "xtick": 14,
    "ytick": 14,
    "title": 16,
    "legend": 14,
}
