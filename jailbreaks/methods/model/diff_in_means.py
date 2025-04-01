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
import contextlib
from tqdm import tqdm

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
    
    def get_last_position_logits(self, model, harmful_toks, harmless_toks, hooks=None) -> torch.Tensor:
        """Get logits at last position for harmful and harmless inputs."""
        with model.hooks(hooks) if hooks else contextlib.nullcontext():
            harmful_logits = model(harmful_toks)
            harmless_logits = model(harmless_toks)
        
        return harmful_logits[:, -1, :], harmless_logits[:, -1, :]

    def kl_div(self, probs_a, probs_b, epsilon=1e-8):
        """Calculate KL divergence between two probability distributions."""
        return torch.sum(probs_a * (torch.log(probs_a + epsilon) - torch.log(probs_b + epsilon)), dim=-1).mean()

    def refusal_score(self, logits, refusal_indices, epsilon=1e-8):
        """Measure tendency to refuse based on refusal token probabilities."""
        probs = torch.nn.functional.softmax(logits, dim=-1)
        refusal_probs = torch.sum(probs[:, refusal_indices], dim=-1)
        nonrefusal_probs = 1.0 - refusal_probs
        return torch.log(refusal_probs + epsilon) - torch.log(nonrefusal_probs + epsilon)

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
        
        # Get baseline refusal scores
        baseline_harmful_logits = harmful_logits[:, -1, :]
        baseline_harmless_logits = harmless_logits[:, -1, :]
        
        # Define refusal tokens (could be customized based on tokenizer)
        refusal_tokens = ["I'm sorry", "I apologize", "I cannot", "I can't"]
        refusal_indices = []
        for token in refusal_tokens:
            ids = tokenizer.encode(token, add_special_tokens=False)
            refusal_indices.extend(ids)
        refusal_indices = list(set(refusal_indices))  # Remove duplicates
        
        baseline_refusal_harmful = self.refusal_score(baseline_harmful_logits, refusal_indices).mean()
        baseline_refusal_harmless = self.refusal_score(baseline_harmless_logits, refusal_indices).mean()
        
        logger.info(f"Baseline refusal score (harmful): {baseline_refusal_harmful:.4f}")
        logger.info(f"Baseline refusal score (harmless): {baseline_refusal_harmless:.4f}")
        
        # Collect candidate directions from multiple positions and layers
        n_layers = model.cfg.n_layers
        n_positions = min(5, harmful_toks.shape[1])  # Consider last 5 positions or fewer
        candidate_positions = list(range(-n_positions, 0))
        candidate_directions = {}
        
        for pos in candidate_positions:
            for layer in range(n_layers):
                harmful_act = harmful_cache['resid_pre', layer][:, pos, :].mean(dim=0)
                harmless_act = harmless_cache['resid_pre', layer][:, pos, :].mean(dim=0)
                
                direction = harmful_act - harmless_act
                direction = direction / (direction.norm() + 1e-8)
                candidate_directions[(pos, layer)] = direction
        
        logger.info(f"Evaluating {len(candidate_directions)} candidate directions")
        
        # Evaluate each direction
        best_direction = None
        best_score = float('inf')
        best_pos_layer = None
        
        results = []
        
        for (pos, layer), direction in tqdm(candidate_directions.items()[:5], total=len(candidate_directions)):
            hooks = generate_hooks(model, direction)
            
            # Evaluate refusal with this direction
            with model.hooks(hooks):
                ablated_harmful_logits, _ = model.run_with_cache(harmful_toks, names_filter=lambda hook_name: 'resid' in hook_name)
                ablated_harmless_logits, _ = model.run_with_cache(harmless_toks, names_filter=lambda hook_name: 'resid' in hook_name)
            
            # Calculate refusal scores with direction ablation
            ablated_refusal_harmful = self.refusal_score(ablated_harmful_logits[:, -1, :], refusal_indices).mean()
            ablated_refusal_harmless = self.refusal_score(ablated_harmless_logits[:, -1, :], refusal_indices).mean()
            
            # Calculate KL divergence for harmless prompts (measures distribution change)
            baseline_probs = torch.nn.functional.softmax(baseline_harmless_logits, dim=-1)
            ablated_probs = torch.nn.functional.softmax(ablated_harmless_logits[:, -1, :], dim=-1)
            kl_divergence = self.kl_div(baseline_probs, ablated_probs)
            
            # Compute overall score (lower is better)
            # We want: low refusal on harmful, unchanged refusal on harmless, low KL divergence
            overall_score = ablated_refusal_harmful + abs(ablated_refusal_harmless - baseline_refusal_harmless) + kl_divergence
            
            results.append({
                'position': pos,
                'layer': layer,
                'harmful_refusal': ablated_refusal_harmful.item(),
                'harmless_refusal': ablated_refusal_harmless.item(),
                'kl_divergence': kl_divergence.item(),
                'overall_score': overall_score.item()
            })
            
            if overall_score < best_score:
                best_score = overall_score
                best_direction = direction
                best_pos_layer = (pos, layer)
        
        # Sort results by overall score (lower is better)
        results.sort(key=lambda x: x['overall_score'])
        
        # Log the top 5 directions
        logger.info("Top 5 directions:")
        for i, result in enumerate(results[:5]):
            logger.info(f"#{i+1}: pos={result['position']}, layer={result['layer']}, "
                       f"score={result['overall_score']:.4f}, "
                       f"harmful_refusal={result['harmful_refusal']:.4f}, "
                       f"harmless_refusal={result['harmless_refusal']:.4f}, "
                       f"kl_div={result['kl_divergence']:.4f}")
        
        # Save the best direction
        pos, layer = best_pos_layer
        logger.info(f"Selected best direction from position {pos}, layer {layer}")
        self.directions[model_path] = best_direction
        
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