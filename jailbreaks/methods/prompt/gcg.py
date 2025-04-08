from jailbreaks.methods.base_method import PromptInjection
import nanogcg
from nanogcg import GCGConfig
from jailbreaks.utils.model_loading import load_model, load_tokenizer
import os
import time

import logging

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
    
from dataclasses import dataclass
@dataclass
class CustomGCGResult:
    suffix: str
    loss: float
    message: str
    target: str
    # loss_history: list[float]
    # strings: list[str]
    config: GCGConfig
    time: float
import json
class GCG(PromptInjection):
    def __init__(self, message: str, target: str, config: GCGConfig = None, path: str = None):
        super().__init__()
        self.message = message
        self.target = target
        self.results = {}
        self.config = config or GCGConfig(
            num_steps=500,
            search_width=64,
            topk=64,
            seed=42,
            verbosity="WARNING"
        )
        self.path = path or "checkpoints/gcg.json"
        if self.path:
            self.load(self.path)
        
    def save(self, path: str = None):
        logger.info(f"Saving GCG results to {path or self.path}")
        serializable_results = {
            model_name: {
                "suffix": result.suffix,
                "loss": result.loss,
                "message": result.message,
                "target": result.target,
                "config": result.config.__dict__,
                "time": result.time,
                # "loss_history": result.loss_history,
                # "strings": result.strings
            }
            for model_name, result in self.results.items()
        }
        with open(path or self.path, "w") as f:
            json.dump(serializable_results, f)
    
    def load(self, path: str = None):
        logger.info(f"Loading GCG from {path or self.path}")
        # check if path exists
        if not os.path.exists(path or self.path):
            logger.warning(f"Path {path or self.path} does not exist")
            return
        with open(path or self.path, "r") as f:
            serialized_results = json.load(f)
            # Convert dictionaries back to GCGResult objects
            self.results = {
                model_name: CustomGCGResult(
                    message=data["message"],
                    target=data["target"],
                    config=GCGConfig(**data["config"]),
                    suffix=data["suffix"],
                    loss=data["loss"],
                    time=data["time"],
                    # loss_history=data["loss_history"],
                    # strings=data["strings"]
                )
                for model_name, data in serialized_results.items()
            }
    
    def clear(self):
        self.results = {}
        if self.path and os.path.exists(self.path):
            os.remove(self.path)
            logger.info(f"Deleted GCG results file: {self.path}")
    
    def fit_models(self, model_names: list[str], refit: bool = True):
        for model_name in model_names:
            self.fit(model_name, refit)
    
    def fit(self, model_name: str, refit: bool = True):
        if not refit and model_name in self.results:
            logger.info(f"Skipping fitting for {model_name} because it already exists")
            return

        logger.info(f"Fitting GCG for {model_name}")
        
        self.load()

        model = load_model(model_name)
        tokenizer = load_tokenizer(model_name)
        original_forward = model.forward
        
        def patched_forward(*args, **kwargs):
            pkv = kwargs.get("past_key_values", None)
            if pkv is not None and isinstance(pkv, list):
                kwargs["past_key_values"] = PastKeyValuesWrapper(pkv)
            return original_forward(*args, **kwargs)
        
        model.forward = patched_forward
        start_time = time.time()
        result = nanogcg.run(model, tokenizer, self.message, self.target, self.config)
        end_time = time.time()
        
        result = CustomGCGResult(
            message=self.message,
            target=self.target,
            config=self.config,
            suffix=result.best_string,
            loss=result.best_loss,
            time=end_time - start_time,
            # loss_history=result.losses,
            # strings=result.strings
        )
        logger.info(f"Time taken to fit {model_name}: {end_time - start_time} seconds")
        self.results[model_name] = result
        self.save()
        
        return result

    def preprocess(self, prompt: str, model_name: str, refit: bool = False) -> str:
        if model_name not in self.results or refit:
            self.fit([model_name], refit)
            
        return f"{prompt} {self.results[model_name].suffix}"


