from jailbreaks.methods.model import DiffInMeans

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Instantiate DiffInMeans - Assuming it handles internal device placement correctly
d = DiffInMeans(use_cache=True)

model_path = "Qwen/Qwen-1_8B-chat"
model_path = "Qwen/Qwen2-0.5B-Instruct"

harmful_prompts = get_advbench_instructions()
harmless_prompts = get_harmless_instructions()

N = 32
d.fit(model_path, harmful_prompts[:N], harmless_prompts[:N], refit=True)

# Removed redundant import torch here
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model using device_map="auto"
# Add torch_dtype for potential performance improvement on CUDA
model_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto", # Automatically uses CUDA if available
    torch_dtype=model_dtype
)
input_device = model.get_input_embeddings().weight.device
logger.info(f"Model input layer device: {input_device}")

model = d.apply(model)
hooks = d.get_hooks(model)
tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")

# Original code for reference (commented out)
# with model.hooks(hooks):
#     output_toks = model.generate(
#         toks.input_ids,
#         max_new_tokens=100,
#         attention_mask=toks.attention_mask
#     )
# response = tokenizer.decode(output_toks[0], skip_special_tokens=True)
# logger.info(response)