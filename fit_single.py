from jailbreaks.methods.model import DiffInMeans
from jailbreaks.utils.model_loading import load_hooked_transformer, load_tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

import requests
import pandas as pd
import io
from jailbreaks.data import get_advbench_instructions, get_harmless_instructions
import torch

import logging

logger = logging.getLogger("Ablation-TEST")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

# Determine device
import time
start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


model_path = "meta-llama/Llama-3.1-8B-Instruct"

model_path = "mistralai/Mistral-7B-Instruct-v0.1"
#model_path = "Qwen/Qwen-1_8B-chat" #works 120s

#model_path = "Qwen/Qwen2-0.5B-Instruct" #works
#model_path = "Qwen/Qwen2.5-1.5B-Instruct" #works 160s
#model_path = "Qwen/Qwen2.5-3B-Instruct" #works 330s
#model_path = "Qwen/Qwen2.5-7B-Instruct" #works 224s
#model_path = "Qwen/Qwen2.5-14B-Instruct" # out of memory


tokenizer = load_tokenizer(model_path)
# Instantiate DiffInMeans - Assuming it handles internal device placement correctly
d = DiffInMeans(use_cache=True)



harmful_prompts = get_advbench_instructions()
harmless_prompts = get_harmless_instructions()

N = 32
d.fit(model_path, harmful_prompts[:N], harmless_prompts[:N], refit=True)
logger.info(f"Fitting time: {time.time() - start}")
d.save()