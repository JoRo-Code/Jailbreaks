import os 
import json
import logging

import requests
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

DATASET_DIR = "data"
HARMFUL_CSV_PATH = os.path.join(DATASET_DIR, "harmful_behaviors.csv")
HARMLESS_JSONL_PATH = os.path.join(DATASET_DIR, "harmless_instructions.jsonl")

def ensure_dataset_dir():
    os.makedirs(DATASET_DIR, exist_ok=True)

def get_advbench_instructions(test_split=0.0):
    ensure_dataset_dir()
    if not os.path.exists(HARMFUL_CSV_PATH):
        logger.info(f"Downloading harmful instructions to {HARMFUL_CSV_PATH}...")
        url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        with open(HARMFUL_CSV_PATH, 'w', encoding='utf-8') as f:
            f.write(response.content.decode('utf-8'))
        logger.info("Download complete.")
    else:
        logger.info(f"Loading harmful instructions from {HARMFUL_CSV_PATH}...")

    dataset = pd.read_csv(HARMFUL_CSV_PATH)
    instructions = dataset['goal'].tolist()
    logger.info(f"Loaded {len(instructions)} harmful instructions.")
    
    if test_split > 0:
        train, test = train_test_split(instructions, test_size=test_split, random_state=42)
        return train, test
    
    return instructions

def get_harmless_instructions():
    """Loads harmless instructions, downloading if necessary."""
    ensure_dataset_dir()
    if not os.path.exists(HARMLESS_JSONL_PATH):
        logger.info("Downloading harmless instructions...")
        hf_path = 'tatsu-lab/alpaca'
        dataset = load_dataset(hf_path)
        logger.info("Download complete. Filtering and saving...")

        instructions = []
        for i in range(len(dataset['train'])):
            if dataset['train'][i]['input'].strip() == '':
                instructions.append(dataset['train'][i]['instruction'])

        # Save as JSON Lines
        with open(HARMLESS_JSONL_PATH, 'w', encoding='utf-8') as f:
            for instruction in instructions:
                f.write(json.dumps({"instruction": instruction}) + '\n')
        logger.info(f"Saved {len(instructions)} harmless instructions to {HARMLESS_JSONL_PATH}.")

    else:
        logger.info(f"Loading harmless instructions from {HARMLESS_JSONL_PATH}...")
        instructions = []
        with open(HARMLESS_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                instructions.append(data['instruction'])
        logger.info(f"Loaded {len(instructions)} harmless instructions.")

    return instructions

if __name__ == "__main__":
    get_advbench_instructions()
    get_harmless_instructions()
