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
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)



QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

# TODO: move to utils
# think about how to include prefix injections in the assistant 
def tokenize_prompts(prompts: List[str], tokenizer: AutoTokenizer) -> List[str]:
    # Check if it's a Qwen model
    if tokenizer.name_or_path and "Qwen" in tokenizer.name_or_path:
        # Use Qwen-specific template
        prompts = [QWEN_CHAT_TEMPLATE.format(instruction=prompt) for prompt in prompts]
    
    else:
        raise Exception(f"Tokenizer {tokenizer.name_or_path} not supported. Add chat template to tokenizer.")
    
    return tokenizer(prompts, padding=True,truncation=False, return_tensors="pt").input_ids

def direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"]
):
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj

class DiffInMeans(ModelManipulation):
    def __init__(self, path: str = None):
        super().__init__()
        self.name = "diff-in-means"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.path = path or "jailbreaks/checkpoints/diff-in-means.pkl"
        self.model2hooks = self.load(self.path) or {}        
    
    def save(self, path: str = None):
        path = path or self.path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "version": "1.0",
                    "model2hooks": self.model2hooks
                }, f)
            logger.info(f"Successfully saved model hooks to {path}")
        except Exception as e:
            logger.error(f"Failed to save model hooks: {e}")
            
    def load(self, path: str = None):
        path = path or self.path
        if not os.path.exists(path):
            logger.warning(f"No saved hooks found at {path}")
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
                
            # Handle versioning if needed
            if isinstance(data, dict) and "version" in data:
                hooks = data["model2hooks"]
            else:
                # Legacy format
                hooks = data
                
            logger.info(f"Successfully loaded model hooks from {path}")
            return hooks
        except Exception as e:
            logger.warning(f"No hooks found at {path}, initializing empty model2hooks")
            return {}
    
    def load_model(self, model_path: str) -> HookedTransformer:
        model = HookedTransformer.from_pretrained_no_processing(
            model_path,
            device=self.device,
            dtype=torch.float16,
            default_padding_side='left',
            fp16=True,
            trust_remote_code=True
        )
        return model
    
    def fit(self, model_path: str, harmful_prompts: List[str], harmless_prompts: List[str], refit=False) -> str:
        if model_path in self.model2hooks and not refit:
            logger.info(f"Skipping fitting for model {model_path} because it already exists")
            return self.hooks_for_model(model_path)
        
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
        
        intervention_layers = list(range(model.cfg.n_layers)) # all layers
        hook_fn = functools.partial(direction_ablation_hook,direction=intervention_dir)
        fwd_hooks = [(utils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]
        self.model2hooks[model_path] = fwd_hooks
        
        logger.info(f"Saved hooks for model {model_path}")
        
        return fwd_hooks
        
    def generate(self, model_path: str, inputs: Dict[str, torch.Tensor], **kwargs) -> str:
        model = self.load_model(model_path)
        hooks = self.model2hooks[model_path]
        return self._generate_with_hooks(model, inputs, fwd_hooks=hooks, **kwargs)
    
    def generate_completion(self, model_path: str, prompt: str, **kwargs) -> str:
        model = self.load_model(model_path)
        hooks = self.model2hooks[model_path]
        toks = tokenize_prompts([prompt], model.tokenizer)
        return self._generate_with_hooks(model, toks, fwd_hooks=hooks, **kwargs)
    
    def hooks_for_model(self, model_path: str) -> List[Tuple[str, Callable]]:
        return self.model2hooks[model_path]
    
    
    def _generate_with_hooks(
        self,
        model: HookedTransformer,
        toks: Int[Tensor, 'batch_size seq_len'], # TODO: fix type
        max_new_tokens: int = 64,
        fwd_hooks = [],
        **kwargs,
    ) -> List[str]:

        all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_new_tokens), dtype=torch.long, device=toks.device)
        all_toks[:, :toks.shape[1]] = toks

        for i in range(max_new_tokens):
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(all_toks[:, :-max_new_tokens + i])
                next_tokens = logits[:, -1, :].argmax(dim=-1) # greedy sampling (temperature=0) # TODO: flexible sampling
                all_toks[:,-max_new_tokens+i] = next_tokens

        return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=True)