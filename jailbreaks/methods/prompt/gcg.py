from jailbreaks.methods.base_method import PromptInjection
import nanogcg
from nanogcg import GCGConfig
from jailbreaks.utils.model_loading import load_model, load_tokenizer
import torch
import os

import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GCG")

class PastKeyValuesWrapper:
    """
    A simple wrapper around the list of past key values that implements get_seq_length() and update().
    Assumes that each element in the list is a tuple (key, value) with key tensor shape:
    (batch_size, num_heads, seq_length, head_dim).
    """
    def __init__(self, pkv):
        self.pkv = pkv
        if self.pkv and len(self.pkv) > 0:
            # Get the sequence length from the first key tensor.
            self.seq_length = self.pkv[0][0].size(2)
        else:
            self.seq_length = 0

    def get_seq_length(self):
        return self.seq_length

    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        """
        Update the cached key and value states for the given layer.
        This method is called by Qwen2 during cache update. We simply replace the cached tuple for the specified
        layer with the new key and value states. Depending on the requirements of caching, you might instead 
        need to concatenate along the sequence dimension.
        """
        try:
            self.pkv[layer_idx] = (key_states, value_states)
        except IndexError:
            # If layer_idx is out of range, ignore the update.
            pass
        # Optionally update the seq_length if updating layer 0.
        if layer_idx == 0 and key_states is not None:
            self.seq_length = key_states.size(2)
        return key_states, value_states

    def __iter__(self):
        return iter(self.pkv)

    def __getitem__(self, idx):
        return self.pkv[idx]

    def __len__(self):
        return len(self.pkv)

import json
class GCG(PromptInjection):
    def __init__(self, message: str, target: str, config: GCGConfig = None, path: str = None):
        super().__init__()
        self.message = message
        self.target = target
        self.suffixes = {}
        self.config = config or GCGConfig(
            num_steps=500,
            search_width=64,
            topk=64,
            seed=42,
            verbosity="WARNING"
        )
        self.path = path
        if self.path:
            self.load(self.path)
        
    def save(self, path: str = None):        
        with open(path or self.path, "w") as f:
            json.dump(self.suffixes, f)
    
    def load(self, path: str = None):
        logger.info(f"Loading GCG from {path or self.path}")
        # check if path exists
        if not os.path.exists(path or self.path):
            logger.warning(f"Path {path or self.path} does not exist")
            return
        with open(path or self.path, "r") as f:
            self.suffixes = json.load(f)
    
    def fit(self, model_names: list[str], refit: bool = True):
        for model_name in model_names:
            if refit or model_name not in self.suffixes:
                self.fit_model(model_name)
    
    def fit_model(self, model_name: str):
        logger.info(f"Fitting GCG for {model_name}")
        
        model = load_model(model_name)
        tokenizer = load_tokenizer(model_name)
        original_forward = model.forward
        
        def patched_forward(*args, **kwargs):
            pkv = kwargs.get("past_key_values", None)
            if pkv is not None and isinstance(pkv, list):
                kwargs["past_key_values"] = PastKeyValuesWrapper(pkv)
            return original_forward(*args, **kwargs)
        
        model.forward = patched_forward
        result = nanogcg.run(model, tokenizer, self.message, self.target, self.config)
                
        self.suffixes[model_name] = result.best_string
        return result

    def preprocess(self, prompt: str, model_name: str, refit: bool = False) -> str:
        if model_name not in self.suffixes or refit:
            self.fit([model_name], refit)
            
        return f"{prompt} {self.suffixes[model_name]}"


