from jailbreaks.methods.model import DiffInMeans

d = DiffInMeans()

model_path = "Qwen/Qwen-1_8B-chat"

import requests
import pandas as pd
import io
from datasets import load_dataset

def get_advbench_instructions():
    url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
    response = requests.get(url)

    dataset = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    instructions = dataset['goal'].tolist()
    return instructions

def get_harmless_instructions():
    hf_path = 'tatsu-lab/alpaca'
    dataset = load_dataset(hf_path)

    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset['train'])):
        if dataset['train'][i]['input'].strip() == '':
            instructions.append(dataset['train'][i]['instruction'])

    return instructions

harmful_prompts = get_advbench_instructions()
harmless_prompts = get_harmless_instructions()

d.fit(model_path, harmful_prompts[:5], harmless_prompts[:5])

