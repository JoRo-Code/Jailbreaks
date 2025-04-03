from jailbreaks.methods.model import DiffInMeans

import requests
import pandas as pd
import io
from jailbreaks.data import get_advbench_instructions, get_harmless_instructions
import torch

import logging

import os
HF_TOKEN = os.getenv('HUGGINGFACE_HUB_TOKEN')
if not HF_TOKEN:
    raise ValueError("Please set the HUGGINGFACE_HUB_TOKEN environment variable.")

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

# Instantiate DiffInMeans - Assuming it handles internal device placement correctly
d = DiffInMeans(use_cache=True, hf_token=HF_TOKEN)

model_path = "meta-llama/Llama-3.1-8B-Instruct"

"mistralai/Mistral-7B-Instruct-v0.3"
#model_path = "Qwen/Qwen-1_8B-chat" #works 120s

#model_path = "Qwen/Qwen2-0.5B-Instruct" #works
#model_path = "Qwen/Qwen2.5-1.5B-Instruct" #works 160s
#model_path = "Qwen/Qwen2.5-3B-Instruct" #works 330s
#model_path = "Qwen/Qwen2.5-7B-Instruct" #works 224s
#model_path = "Qwen/Qwen2.5-14B-Instruct" # out of memory

harmful_prompts = get_advbench_instructions()
harmless_prompts = get_harmless_instructions()

N = 32
d.fit(model_path, harmful_prompts[:N], harmless_prompts[:N], refit=True)
logger.info(f"Fitting time: {time.time() - start}")
d.save()

# Removed redundant import torch here
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model using device_map="auto"
# Add torch_dtype for potential performance improvement on CUDA
model_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto", # Automatically uses CUDA if available
    torch_dtype=model_dtype,
    trust_remote_code=True,
    use_auth_token=HF_TOKEN,
)
input_device = model.get_input_embeddings().weight.device
logger.info(f"Model input layer device: {input_device}")

model = d.apply(model)
hooks = d.get_hooks(model)

tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto", use_auth_token=HF_TOKEN)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
toks = tokenizer(harmful_prompts[0], return_tensors="pt").to(device)
# Original code for reference (commented out)
with model.hooks(hooks):
    output_toks = model.generate(
         toks.input_ids,
         max_new_tokens=100,
         attention_mask=toks.attention_mask
     )
response = tokenizer.decode(output_toks[0], skip_special_tokens=True)
logger.info(response)
logger.info(f"Generation time: {time.time() - start}")

