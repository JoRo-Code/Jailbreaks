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

    def kl_div(self, probs_a, probs_b, epsilon=1e-6):
        """Calculate KL divergence between two probability distributions."""
        # Ensure inputs are probabilities (sum to 1) and non-negative
        probs_a = torch.clamp(probs_a, min=0)
        probs_b = torch.clamp(probs_b, min=0)
        # Add epsilon and re-normalize slightly to avoid log(0) issues more robustly
        probs_a = (probs_a + epsilon) / (1.0 + probs_a.shape[-1] * epsilon)
        probs_b = (probs_b + epsilon) / (1.0 + probs_b.shape[-1] * epsilon)
        
        # Use log_softmax for potentially better stability
        log_probs_a = torch.log(probs_a) # log(p)
        log_probs_b = torch.log(probs_b) # log(q)

        # KL(P || Q) = sum(P * (log P - log Q))
        kl_term = probs_a * (log_probs_a - log_probs_b)
        
        # Check for NaNs/Infs in the KL term itself before summing
        if torch.isnan(kl_term).any() or torch.isinf(kl_term).any():
            logger.warning("NaN or Inf detected in KL divergence term calculation. Clamping.")
            # Replace NaNs/Infs with 0, assuming they result from 0 * log(0) or similar issues
            # A large penalty might be better if this happens often.
            kl_term = torch.nan_to_num(kl_term, nan=0.0, posinf=0.0, neginf=0.0) 
            
        return torch.sum(kl_term, dim=-1).mean()

    def refusal_score(self, logits, refusal_indices, epsilon=1e-6):
        """Measure tendency to refuse based on refusal token probabilities."""
        probs = torch.nn.functional.softmax(logits.float(), dim=-1) # Use float32 for softmax stability
        
        # Ensure refusal_indices are valid
        if not refusal_indices:
             logger.warning("Refusal indices list is empty.")
             return torch.tensor(0.0, device=logits.device) # Return neutral score if no refusal tokens
        
        # Clamp indices to be within vocab size
        vocab_size = probs.shape[-1]
        valid_indices = [idx for idx in refusal_indices if 0 <= idx < vocab_size]
        if len(valid_indices) < len(refusal_indices):
            logger.warning(f"Some refusal indices were out of bounds. Original: {len(refusal_indices)}, Valid: {len(valid_indices)}")
        if not valid_indices:
             logger.warning("No valid refusal indices found after bounds check.")
             return torch.tensor(0.0, device=logits.device)

        refusal_probs = torch.sum(probs[:, valid_indices], dim=-1)
        nonrefusal_probs = 1.0 - refusal_probs
        
        # Add epsilon before log
        log_refusal = torch.log(refusal_probs + epsilon)
        log_nonrefusal = torch.log(nonrefusal_probs + epsilon)

        score = log_refusal - log_nonrefusal

        # Check for NaNs/Infs in the final score
        if torch.isnan(score).any() or torch.isinf(score).any():
            logger.warning("NaN or Inf detected in refusal score calculation. Clamping.")
            score = torch.nan_to_num(score, nan=0.0, posinf=10.0, neginf=-10.0) # Clamp infs to large values

        return score

    def fit(self, model_path: str, harmful_prompts: List[str], harmless_prompts: List[str], refit=False) -> str:
        if model_path in self.directions and not refit:
            logger.info(f"Skipping fitting for model {model_path} because it already exists")
            return self.get_direction(model_path)
        
        # Add check for empty prompts
        if not harmful_prompts or not harmless_prompts:
            raise ValueError("Harmful and harmless prompts lists cannot be empty for fitting.")

        logger.info(f"Loading model {model_path}")
        model = self.load_model(model_path)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Ensure tokenizer has a padding token if padding is enabled
        if tokenizer.pad_token is None:
            logger.warning("Tokenizer does not have a pad token. Setting to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            model.tokenizer.pad_token = model.tokenizer.eos_token # Also update the model's tokenizer instance if separate

        # Ensure the model's tokenizer config matches the loaded tokenizer regarding padding side
        # HookedTransformer defaults to left padding, ensure consistency
        if model.tokenizer.padding_side != 'left':
             logger.warning(f"Model tokenizer padding side is '{model.tokenizer.padding_side}'. Forcing to 'left'.")
             model.tokenizer.padding_side = 'left'
        if tokenizer.padding_side != 'left':
             logger.warning(f"Loaded tokenizer padding side is '{tokenizer.padding_side}'. Forcing to 'left'.")
             tokenizer.padding_side = 'left'

        tokenize_instructions_fn = functools.partial(tokenize_prompts, tokenizer=tokenizer) # Use the loaded tokenizer consistently
        
        harmful_toks = tokenize_instructions_fn(harmful_prompts)
        harmless_toks = tokenize_instructions_fn(harmless_prompts)

        # Add check for empty tokenized tensors (e.g., if prompts were just padding tokens)
        if harmful_toks.numel() == 0 or harmless_toks.numel() == 0:
             raise ValueError("Tokenization resulted in empty tensors. Check input prompts.")
        if harmful_toks.shape[1] == 0 or harmless_toks.shape[1] == 0:
             raise ValueError(f"Tokenization resulted in tensors with sequence length 0. Harmful shape: {harmful_toks.shape}, Harmless shape: {harmless_toks.shape}. Check input prompts and tokenizer padding.")

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
        # Ensure n_positions is at least 1 if harmful_toks has sequences
        n_positions = min(5, harmful_toks.shape[1]) if harmful_toks.shape[1] > 0 else 0
        candidate_positions = list(range(-n_positions, 0))
        candidate_directions = {}
        
        # Add check if candidate_positions is empty
        if not candidate_positions:
             raise ValueError(f"Could not determine candidate positions. Harmful tokens sequence length might be too short ({harmful_toks.shape[1]}).")

        for pos in candidate_positions:
            for layer in range(n_layers):
                # Ensure tensors are on the same device
                harmful_act = harmful_cache['resid_pre', layer][:, pos, :].mean(dim=0).to(self.device)
                harmless_act = harmless_cache['resid_pre', layer][:, pos, :].mean(dim=0).to(self.device)
                
                direction = harmful_act - harmless_act
                direction_norm = direction.norm()
                # Avoid division by zero or very small norms
                if direction_norm > 1e-8:
                    direction = direction / direction_norm
                else:
                    logger.warning(f"Direction norm is close to zero at pos={pos}, layer={layer}. Skipping this candidate.")
                    continue # Skip this candidate direction if norm is too small
                candidate_directions[(pos, layer)] = direction
        
        # Add check if candidate_directions is empty after the loop
        if not candidate_directions:
            raise ValueError("No valid candidate directions found. Check model activations or prompt differences.")

        logger.info(f"Evaluating {len(candidate_directions)} candidate directions")
        
        # Evaluate each direction
        best_direction = None
        # Initialize best_score to a very large float, ensuring it's the same type as overall_score might become
        best_score = torch.finfo(torch.float32).max 
        best_pos_layer = None
        
        results = []
        
        # Use float32 for baseline logits/probs for stability in comparisons
        baseline_harmful_logits = harmful_logits[:, -1, :].float()
        baseline_harmless_logits = harmless_logits[:, -1, :].float()
        baseline_probs_harmless = torch.nn.functional.softmax(baseline_harmless_logits, dim=-1)
        baseline_refusal_harmful = self.refusal_score(baseline_harmful_logits, refusal_indices).mean()
        baseline_refusal_harmless = self.refusal_score(baseline_harmless_logits, refusal_indices).mean()

        for (pos, layer), direction in tqdm(candidate_directions.items(), total=len(candidate_directions)):
            hooks = generate_hooks(model, direction.to(self.device)) # Ensure direction is on device
            
            # Evaluate refusal with this direction
            ablated_harmful_logits = None
            ablated_harmless_logits = None
            try:
                with model.hooks(hooks):
                    # Run with float32 context potentially? Or just cast outputs
                    ablated_harmful_logits, _ = model.run_with_cache(harmful_toks, names_filter=lambda hook_name: 'resid' in hook_name)
                    ablated_harmless_logits, _ = model.run_with_cache(harmless_toks, names_filter=lambda hook_name: 'resid' in hook_name)
            except Exception as e:
                logger.error(f"Error during hooked run for pos={pos}, layer={layer}: {e}")
                continue # Skip this direction if the hooked run fails

            # Ensure logits were obtained
            if ablated_harmful_logits is None or ablated_harmless_logits is None:
                 logger.warning(f"Failed to get ablated logits for pos={pos}, layer={layer}. Skipping.")
                 continue

            # Calculate refusal scores with direction ablation, use float32
            ablated_harmful_logits_last = ablated_harmful_logits[:, -1, :].float()
            ablated_harmless_logits_last = ablated_harmless_logits[:, -1, :].float()
            
            ablated_refusal_harmful = self.refusal_score(ablated_harmful_logits_last, refusal_indices).mean()
            ablated_refusal_harmless = self.refusal_score(ablated_harmless_logits_last, refusal_indices).mean()
            
            # Calculate KL divergence for harmless prompts (measures distribution change)
            ablated_probs_harmless = torch.nn.functional.softmax(ablated_harmless_logits_last, dim=-1)
            kl_divergence = self.kl_div(baseline_probs_harmless, ablated_probs_harmless)
            
            # Check for NaNs/Infs in components before calculating overall score
            score_components = [ablated_refusal_harmful, ablated_refusal_harmless, baseline_refusal_harmless, kl_divergence]
            if any(torch.isnan(c).any() or torch.isinf(c).any() for c in score_components):
                logger.warning(f"NaN or Inf detected in score components for pos={pos}, layer={layer}. "
                               f"Harmful Refusal: {ablated_refusal_harmful.item():.4f}, "
                               f"Harmless Refusal: {ablated_refusal_harmless.item():.4f}, "
                               f"KL Div: {kl_divergence.item():.4f}. Assigning high penalty score.")
                # Assign a large penalty score instead of nan/inf
                overall_score = torch.tensor(1e6, device=self.device, dtype=torch.float32) 
            else:
                # Compute overall score (lower is better)
                # We want: low refusal on harmful, unchanged refusal on harmless, low KL divergence
                refusal_diff = torch.abs(ablated_refusal_harmless - baseline_refusal_harmless)
                overall_score = ablated_refusal_harmful + refusal_diff + kl_divergence

            # Ensure overall_score is a scalar tensor
            if not isinstance(overall_score, torch.Tensor) or overall_score.numel() != 1:
                 logger.error(f"overall_score is not a scalar tensor for pos={pos}, layer={layer}. Type: {type(overall_score)}, Value: {overall_score}. Skipping.")
                 continue # Skip if score calculation failed unexpectedly

            # Ensure score is float for comparison
            current_score_float = overall_score.item() 

            results.append({
                'position': pos,
                'layer': layer,
                'harmful_refusal': ablated_refusal_harmful.item() if isinstance(ablated_refusal_harmful, torch.Tensor) else ablated_refusal_harmful,
                'harmless_refusal': ablated_refusal_harmless.item() if isinstance(ablated_refusal_harmless, torch.Tensor) else ablated_refusal_harmless,
                'kl_divergence': kl_divergence.item() if isinstance(kl_divergence, torch.Tensor) else kl_divergence,
                'overall_score': current_score_float
            })
            
            # Compare using float values
            if current_score_float < best_score:
                # Check again for NaN/Inf just before assignment
                if not (torch.isnan(overall_score).any() or torch.isinf(overall_score).any()):
                    best_score = current_score_float
                    best_direction = direction # Keep original precision direction
                    best_pos_layer = (pos, layer)
                else:
                     logger.warning(f"Skipping update for best_score due to NaN/Inf in overall_score for pos={pos}, layer={layer}")

        # Sort results by overall score (lower is better), handle potential NaNs by putting them last
        results.sort(key=lambda x: float('inf') if x['overall_score'] is None or torch.isnan(torch.tensor(x['overall_score'])).item() or torch.isinf(torch.tensor(x['overall_score'])).item() else x['overall_score'])
        
        # Log the top 5 directions
        logger.info("Top 5 directions:")
        for i, result in enumerate(results[:5]):
            logger.info(f"#{i+1}: pos={result['position']}, layer={result['layer']}, "
                       f"score={result['overall_score']:.4f}, "
                       f"harmful_refusal={result['harmful_refusal']:.4f}, "
                       f"harmless_refusal={result['harmless_refusal']:.4f}, "
                       f"kl_div={result['kl_divergence']:.4f}")
        
        # Add check before unpacking best_pos_layer
        if best_pos_layer is None:
             # Log results again to see why nothing was chosen
             logger.error("Failed to find a best direction. Evaluation loop did not update the best score. Final top results:")
             for i, result in enumerate(results[:10]): # Log more results
                 logger.error(f"#{i+1}: pos={result['position']}, layer={result['layer']}, "
                            f"score={result['overall_score']:.4f}, "
                            f"harmful_refusal={result['harmful_refusal']:.4f}, "
                            f"harmless_refusal={result['harmless_refusal']:.4f}, "
                            f"kl_div={result['kl_divergence']:.4f}")
             raise RuntimeError("Failed to find a best direction. Evaluation loop did not update the best score.")
        
        # Save the best direction
        pos, layer = best_pos_layer
        logger.info(f"Selected best direction from position {pos}, layer {layer} with score {best_score:.4f}")
        # Ensure best_direction is valid before saving
        if best_direction is None or torch.isnan(best_direction).any() or torch.isinf(best_direction).any():
             raise RuntimeError(f"Selected best_direction is invalid (None, NaN, or Inf) for pos={pos}, layer={layer}")
             
        self.directions[model_path] = best_direction.to('cpu') # Store direction on CPU to save GPU memory
        
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
        # Ensure direction is moved to the correct device when retrieved
        return self.directions[model_path].to(self.device)
    
    def get_hooks(self, model: HookedTransformer) -> List[Tuple[str, Callable]]:
        model_path = hooked_transformer_path(model)
        if model_path not in self.directions:
            raise ValueError(f"No direction found for model {model_path}")
        return generate_hooks(model, self.directions[model_path])