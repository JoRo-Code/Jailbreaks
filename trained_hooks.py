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

logger = logging.getLogger("hooks-test")

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

model = d.load_model(model_path)
tokenizer = d.load_tokenizer(model_path)

hooks = d.get_hooks(model)
logger.info(hooks[0])

from jailbreaks.methods.model.diff_in_means import format_prompts, tokenize_prompts

formatted_prompts = format_prompts(prompts, tokenizer)
ids = tokenize_prompts(formatted_prompts, tokenizer).to(device)

with model.hooks(hooks):
    output_ids = model.generate(ids)

output = [tokenizer.decode(o[ids.shape[1]:], skip_special_tokens=True) for o in output_ids]

for prompt, response in zip(prompts, output):
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Response: {response}")
    logger.info("-" * 20) 

logger.info(f"Running time: {time.time() - start}")




