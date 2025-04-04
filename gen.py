from jailbreaks.methods.model import DiffInMeans

import requests
import pandas as pd
import io
from jailbreaks.data import get_advbench_instructions, get_harmless_instructions
import torch

import logging

import os

from colorama import Fore
import textwrap

import time
start = time.time()

HF_TOKEN = os.getenv('HUGGINGFACE_HUB_TOKEN')
if not HF_TOKEN:
    raise ValueError("Please set the HUGGINGFACE_HUB_TOKEN environment variable.")

logger = logging.getLogger("Ablation-TEST")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Instantiate DiffInMeans - Assuming it handles internal device placement correctly
d = DiffInMeans(use_cache=True, hf_token=HF_TOKEN)

#model_path = "meta-llama/Llama-3.1-8B-Instruct"

#model_path = "mistralai/Mistral-7B-Instruct-v0.1"
#model_path = "Qwen/Qwen-1_8B-chat" #works 120s

#model_path = "Qwen/Qwen2-0.5B-Instruct" #works
#model_path = "Qwen/Qwen2.5-1.5B-Instruct" #works 160s
#model_path = "Qwen/Qwen2.5-3B-Instruct" #works 330s
model_path = "Qwen/Qwen2.5-7B-Instruct" #works 224s
#model_path = "Qwen/Qwen2.5-14B-Instruct" # out of memory

harmful_prompts = get_advbench_instructions()
harmless_prompts = get_harmless_instructions()

N = 32

prompts = harmful_prompts[N:N*2]

result = d.fit_and_generate(model_path, harmful_prompts[:N], harmless_prompts[:N], prompts[:N], layer=15)

for r in result:
    prompt = r['prompt']
    ablated_response = r['ablated_response']
    baseline_response = r['baseline_response']
    print("Prompt:")
    print(textwrap.fill(repr(prompt), width=100, initial_indent='\t', subsequent_indent='\t'))
    print("Intervention:")
    print(Fore.RED)
    print(textwrap.fill(repr(ablated_response), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.RESET)
    print("Baseline:")
    print(Fore.GREEN)
    print(textwrap.fill(repr(baseline_response), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.RESET)
    print("-"*100)
#d.fit(model_path, harmful_prompts[:N], harmless_prompts[:N], refit=True)
logger.info(f"Running time: {time.time() - start}")