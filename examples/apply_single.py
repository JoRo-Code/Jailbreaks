import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch

from jailbreaks.methods.model import DiffInMeans
from jailbreaks.data import get_advbench_instructions, get_harmless_instructions
from jailbreaks.utils.model_loading import load_tokenizer
from jailbreaks.utils.tokenization import format_prompts, tokenize_prompts


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

harmful_prompts = get_advbench_instructions()
harmless_prompts = get_harmless_instructions()

N = 32
d = DiffInMeans(use_cache=True, harmful_prompts=harmful_prompts[:N], harmless_prompts=harmless_prompts[:N]) 


prompts = harmful_prompts[N:N*2]

model_path = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = load_tokenizer(model_path) 
model = d.apply(model_path)

formatted_prompts = format_prompts(prompts, tokenizer)
toks = tokenize_prompts(formatted_prompts, tokenizer).to(device)
ids = toks.input_ids

output_ids = model.generate(ids, max_new_tokens=100)

output = [tokenizer.decode(o[ids.shape[1]:], skip_special_tokens=True) for o in output_ids]

for prompt, response in zip(prompts, output):
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("-" * 20) 




