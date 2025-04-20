from typing import List

from jailbreaks.utils.model_loading import (
    load_model, 
    load_tokenizer
)

def download(model_paths: List[str]):
    for model_path in model_paths:
        load_model(model_path, device="cpu")
        load_tokenizer(model_path, device="cpu")