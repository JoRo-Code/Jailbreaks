import os
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_auth_token(provided_token=None):
    return provided_token if provided_token else os.environ.get("HUGGINGFACE_HUB_TOKEN", None)

def load_hooked_transformer(model_path, device=None, **kwargs):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    return HookedTransformer.from_pretrained_no_processing(
        model_path,
        device=device,
        dtype=torch.float16,
        default_padding_side='left',
        trust_remote_code=True,
        use_auth_token=get_auth_token(),
        **kwargs
    )

def load_model(model_path, device=None, **kwargs):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        trust_remote_code=True,
        token=get_auth_token(),
        **kwargs
    ).to(device)
    return model

def load_tokenizer(model_path, **kwargs):
    return AutoTokenizer.from_pretrained(
        model_path, 
        device_map="auto", 
        token=get_auth_token(),
        **kwargs
    )