from jailbreaks.methods.base_method import ModelManipulation
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from typing import Dict, Any, List, Tuple, Callable
from jaxtyping import Float, Int
from torch import Tensor
import functools

from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import einops

import logging
import pickle
import os

logger = logging.getLogger("DiffInMeans")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

def format_prompts(prompts: List[str], tokenizer: AutoTokenizer) -> List[str]:
    if tokenizer.name_or_path and "Qwen" in tokenizer.name_or_path:
        return [QWEN_CHAT_TEMPLATE.format(instruction=prompt) for prompt in prompts]
    else:
        raise Exception(f"Tokenizer {tokenizer.name_or_path} not supported. Add chat template to tokenizer.")

def tokenize_prompts(prompts: List[str], tokenizer: AutoTokenizer) -> List[str]:
    return tokenizer(prompts, padding=True,truncation=False, return_tensors="pt").input_ids

def hooked_transformer_path(model: HookedTransformer) -> str:
    return model.cfg.tokenizer_name

def direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"]
):
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj

def generate_hooks(model: HookedTransformer, direction: Float[Tensor, "d_act"]):
    intervention_layers = list(range(model.cfg.n_layers)) # all layers
    hook_fn = functools.partial(direction_ablation_hook,direction=direction)
    fwd_hooks = [(utils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]
    return fwd_hooks

class DiffInMeans(ModelManipulation):
    def __init__(self, path: str = None, use_cache: bool = True):
        super().__init__()
        self.name = "diff-in-means"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.path = path or "jailbreaks/checkpoints/diff-in-means.pkl"
        self.directions = self.load_directions(self.path) if use_cache else {}

    def apply(self, model: AutoModelForCausalLM) -> str:
        original_generate = model.generate
        if not isinstance(model, HookedTransformer):
            model = self._convert_to_hooked_transformer(model)
        else:
            logger.warning("Model is already a HookedTransformer")

        model_path = hooked_transformer_path(model)
        direction = self.get_direction(model_path)
        hooks = generate_hooks(model, direction)
        
        def new_generate(*args, **kwargs):
            with model.hooks(hooks):
                logger.debug(f"Generating with '{len(hooks)}' hooks")
                return original_generate(*args, **kwargs)
                
        model.generate = new_generate
                
        return model
    
    def save(self, path: str = None):
        path = path or self.path
        self.save_directions(path)

    def save_directions(self, path: str = None):
        path = path or self.path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.directions, f)
    
    def load_directions(self, path: str = None) -> Dict[str, Float[Tensor, "d_act"]]:
        path = path or self.path
        if not os.path.exists(path):
            logger.warning(f"No saved directions found at {path}")
            return {}
        with open(path, "rb") as f:
            directions = pickle.load(f)
        return directions
    
    def _convert_to_hooked_transformer(self, model: AutoModelForCausalLM) -> HookedTransformer:
        return self.load_model(model.name_or_path)
    
    def load_model(self, model_path: str) -> HookedTransformer:
        logger.info(f"Loading {model_path} as HookedTransformer")
        model = HookedTransformer.from_pretrained_no_processing(
            model_path,
            device=self.device,
            dtype=torch.float16,
            default_padding_side='left',
            #fp16=True,
            trust_remote_code=True
        )
        return model
    
    def fit(self, model_path: str, harmful_prompts: List[str], harmless_prompts: List[str], refit=False) -> str:
        if model_path in self.directions and not refit:
            logger.info(f"Skipping fitting for model {model_path} because it already exists")
            return self.get_direction(model_path)
        
        logger.info(f"Loading model {model_path}")
        model = self.load_model(model_path)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenize_instructions_fn = functools.partial(tokenize_prompts, tokenizer=model.tokenizer)
        
        harmful_toks = tokenize_instructions_fn(harmful_prompts)
        harmless_toks = tokenize_instructions_fn(harmless_prompts)
        
        logger.info(f"Running harmful prompts with cache")
        harmful_logits, harmful_cache = model.run_with_cache(harmful_toks, names_filter=lambda hook_name: 'resid' in hook_name)
        logger.info(f"Running harmless prompts with cache")
        harmless_logits, harmless_cache = model.run_with_cache(harmless_toks, names_filter=lambda hook_name: 'resid' in hook_name)
        
        pos = -1
        layer = 14

        harmful_mean_act = harmful_cache['resid_pre', layer][:, pos, :].mean(dim=0)
        harmless_mean_act = harmless_cache['resid_pre', layer][:, pos, :].mean(dim=0)

        refusal_dir = harmful_mean_act - harmless_mean_act
        refusal_dir = refusal_dir / refusal_dir.norm()
        
        intervention_dir = refusal_dir
        
        self.directions[model_path] = intervention_dir
        
        logger.info(f"Saved direction for model {model_path}")
        return self.directions[model_path]
        
    # def generate(self, model_path: str, inputs: Dict[str, torch.Tensor], **kwargs) -> str:
    #     model = self.load_model(model_path)
    #     hooks = self.model2hooks[model_path]
    #     with model.hooks(hooks):
    #         output = model.generate(inputs, **kwargs)
    #     return model.tokenizer.decode(output[0])
    
    # def generate_completion(self, model_path: str, prompt: str, **kwargs) -> str:
    #     model = self.load_model(model_path)
    #     hooks = self.model2hooks[model_path]
    #     toks = tokenize_prompts([prompt], model.tokenizer)
    #     with model.hooks(hooks):
    #         output = model.generate(toks, **kwargs)
    #     return model.tokenizer.decode(output[0])
    
    def get_direction(self, model_path: str) -> Float[Tensor, "d_act"]:
        if model_path not in self.directions:
            raise ValueError(f"No direction found for model {model_path}")
        return self.directions[model_path]
    
    def get_hooks(self, model: HookedTransformer) -> List[Tuple[str, Callable]]:
        model_path = hooked_transformer_path(model)
        if model_path not in self.directions:
            raise ValueError(f"No direction found for model {model_path}")
        return generate_hooks(model, self.directions[model_path])