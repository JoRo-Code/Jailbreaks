from jailbreaks.methods.base_method import ModelManipulation
from transformers import AutoModelForCausalLM

import torch
from typing import Dict, Any, List, Tuple, Callable
from jaxtyping import Float, Int
from torch import Tensor
import functools

from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import einops

import logging
import os
from tqdm import tqdm
import pandas as pd
import time
import gc

from jailbreaks.utils.refusal import is_refusal
from jailbreaks.utils.tokenization import format_prompts, tokenize_prompts
from jailbreaks.utils.model_loading import load_hooked_transformer, load_tokenizer

from jailbreaks.methods.utils import format_name

logger = logging.getLogger(__name__)


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

def kl_div(probs_a, probs_b, epsilon=1e-6):
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

class DiffInMeans(ModelManipulation):
    def __init__(self, harmful_prompts: List[str], harmless_prompts: List[str], generation_kwargs: Dict = None, use_cache: bool = True, path: str = None, description: str = ""):
        super().__init__()
        self.name = format_name(self.__class__.__name__, description)
        self.description = description
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.path = path or "checkpoints/diff-in-means.pt"
        self.directions = self.load_directions(self.path) if use_cache else {}
        self.harmful_prompts = harmful_prompts
        self.harmless_prompts = harmless_prompts
        self.generation_kwargs = generation_kwargs
    
    def apply(self, model_path: str) -> str:
        model = load_hooked_transformer(model_path)
        hooks = self.get_hooks(model)
        
        original_generate = model.generate
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
        torch.save(self.directions, path)
    
    def load_directions(self, path: str = None) -> Dict[str, Float[Tensor, "d_act"]]:
        path = path or self.path
        if not os.path.exists(path):
            logger.warning(f"No saved directions found at {path}")
            return {}
        directions = torch.load(path, map_location=self.device)
        for key in directions:
            if isinstance(directions[key], torch.Tensor):
                directions[key] = directions[key].to(self.device)
        return directions
    
    def get_direction(self, model_path: str) -> Float[Tensor, "d_act"]:
        if model_path not in self.directions:
            raise ValueError(f"No direction found for model {model_path}")
        return self.directions[model_path].to(self.device)
    
    def get_hooks(self, model: HookedTransformer) -> List[Tuple[str, Callable]]:
        model_path = hooked_transformer_path(model)
        direction = self.get_direction(model_path)
        return generate_hooks(model, direction)
    

    def fit(self, model_path: str, refit: bool = True) -> str:
        """
        generate candidate hook directions
        evaluate directions
        select best direction
        save direction
        
        logging and cache cleanups
        """
        
        start_time_include_loading = time.time()
        
        harmful_prompts = self.harmful_prompts
        harmless_prompts = self.harmless_prompts
        generation_kwargs = self.generation_kwargs
        
        if model_path in self.directions and not refit:
            logger.info(f"Skipping fitting for model {model_path} because it already exists")
            return self.get_direction(model_path)
        
        if not harmful_prompts or not harmless_prompts:
            raise ValueError("Harmful and harmless prompts lists cannot be empty for fitting.")

        logger.info(f"Loading model {model_path}")
        model = load_hooked_transformer(model_path)
        tokenizer = load_tokenizer(model_path)

        fitting_start_time = time.time()

        formatted_harmful_prompts = format_prompts(harmful_prompts, tokenizer)
        formatted_harmless_prompts = format_prompts(harmless_prompts, tokenizer)

        harmful_toks = tokenize_prompts(formatted_harmful_prompts, tokenizer).input_ids.to(self.device)
        harmless_toks = tokenize_prompts(formatted_harmless_prompts, tokenizer).input_ids.to(self.device)

        if harmful_toks.numel() == 0 or harmless_toks.numel() == 0:
             raise ValueError("Tokenization resulted in empty tensors. Check input prompts.")
        if harmful_toks.shape[1] == 0 or harmless_toks.shape[1] == 0:
             raise ValueError(f"Tokenization resulted in tensors with sequence length 0. Check input prompts and tokenizer padding.")

        logger.info("Generating baseline responses...")
        default_generation_kwargs = {
            "max_new_tokens": 50,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": False,
        }
        if generation_kwargs:
            default_generation_kwargs.update(generation_kwargs)

        logger.info("Running baseline prompts with cache for KL div")

        resid_pre_hook_name_filter = lambda name: name.endswith('resid_pre')

        baseline_harmful_logits_full, harmful_cache = model.run_with_cache(
            harmful_toks,
            names_filter=resid_pre_hook_name_filter
        )
        del baseline_harmful_logits_full
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

        baseline_harmless_logits_full, harmless_cache = model.run_with_cache(
            harmless_toks,
            names_filter=resid_pre_hook_name_filter
        )
        baseline_harmless_logits_last = baseline_harmless_logits_full[:, -1, :].float().detach()
        del baseline_harmless_logits_full
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

        baseline_probs_harmless = torch.nn.functional.softmax(baseline_harmless_logits_last, dim=-1)

        logger.info("Generating baseline text outputs...")
        baseline_harmful_outputs_ids = model.generate(harmful_toks, **default_generation_kwargs)
        baseline_harmless_outputs_ids = model.generate(harmless_toks, **default_generation_kwargs)

        # decode baseline outputs, remove prompt
        baseline_harmful_texts = [tokenizer.decode(o[harmful_toks.shape[1]:], skip_special_tokens=True) for o in baseline_harmful_outputs_ids]
        baseline_harmless_texts = [tokenizer.decode(o[harmless_toks.shape[1]:], skip_special_tokens=True) for o in baseline_harmless_outputs_ids]

        # baseline refusal rates
        baseline_harmful_refusal_rate = torch.tensor([is_refusal(t) for t in baseline_harmful_texts], dtype=torch.float).mean().item()
        baseline_harmless_refusal_rate = torch.tensor([is_refusal(t) for t in baseline_harmless_texts], dtype=torch.float).mean().item()

        baseline_harmful_refusals = [is_refusal(t) for t in baseline_harmful_texts]
        baseline_harmless_refusals = [is_refusal(t) for t in baseline_harmless_texts]

        logger.info(f"Baseline refusal rate (harmful): {baseline_harmful_refusal_rate:.4f}")
        logger.info(f"Baseline refusal rate (harmless): {baseline_harmless_refusal_rate:.4f}")
        
        # candidate direction generation
        n_layers = model.cfg.n_layers
        n_positions = 1
        candidate_positions = list(range(-n_positions, 0))
        candidate_directions = {}

        if not candidate_positions:
             del harmful_cache
             del harmless_cache
             if torch.cuda.is_available(): torch.cuda.empty_cache()
             gc.collect()
             raise ValueError(f"Could not determine candidate positions. Harmful tokens sequence length might be too short ({harmful_toks.shape[1]}).")

        logger.info("Calculating candidate directions...")
        for pos in candidate_positions:
            for layer in range(n_layers):
                harmful_act = harmful_cache['resid_pre', layer][:, pos, :].mean(dim=0).to(self.device)
                harmless_act = harmless_cache['resid_pre', layer][:, pos, :].mean(dim=0).to(self.device)
                direction = harmful_act - harmless_act
                direction_norm = direction.norm()
                if direction_norm > 1e-8:
                    direction = direction / direction_norm
                    candidate_directions[(pos, layer)] = direction.detach()

        logger.info("Deleting initial activation caches...")
        del harmful_cache
        del harmless_cache
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()


        if not candidate_directions:
            raise ValueError("No valid candidate directions found. Check model activations or prompt differences.")

        logger.info(f"Evaluating {len(candidate_directions)} candidate directions by generating text...")

        # evaluate candidate directions with KL divergence
        best_direction = None
        best_score = torch.finfo(torch.float32).max
        best_pos_layer = None
        overall_results_log = []
        detailed_results_log = []
        kl_weight = 1.0

        for (pos, layer), direction in tqdm(
            candidate_directions.items(),
            total=len(candidate_directions),
            desc="Evaluating candidate directions"
        ):
            hooks = generate_hooks(model, direction.to(self.device))

            ablated_harmful_texts = []
            ablated_harmless_texts = []
            ablated_harmless_logits_last = None
            kl_divergence = torch.tensor(torch.finfo(torch.float32).max, device=self.device)
            current_harmful_refusals = [True] * len(harmful_prompts)
            current_harmless_refusals = [True] * len(harmless_prompts)
            error_msg = None

            try:
                with model.hooks(hooks):
                    # generate ablated text
                    ablated_harmful_outputs_ids = model.generate(harmful_toks, **default_generation_kwargs)
                    ablated_harmless_outputs_ids = model.generate(harmless_toks, **default_generation_kwargs)

                    # decode ablated outputs
                    ablated_harmful_texts = [tokenizer.decode(o[harmful_toks.shape[1]:], skip_special_tokens=True) for o in ablated_harmful_outputs_ids]
                    ablated_harmless_texts = [tokenizer.decode(o[harmless_toks.shape[1]:], skip_special_tokens=True) for o in ablated_harmless_outputs_ids]

                    # Get ablated logits for KL divergence using a standard forward pass
                    ablated_harmless_logits_full = model(harmless_toks) # No caching here
                    ablated_harmless_logits_last = ablated_harmless_logits_full[:, -1, :].float().detach() # Detach
                    del ablated_harmless_logits_full 

            except Exception as e:
                logger.error(f"Error during hooked generation/logit calculation for pos={pos}, layer={layer}: {e}")
                error_msg = str(e)
                overall_score = torch.tensor(best_score + 1.0, device=self.device) 
                ablated_harmless_logits_last = None
                
                if 'ablated_harmless_logits_full' in locals(): del ablated_harmless_logits_full
                if 'ablated_harmful_outputs_ids' in locals(): del ablated_harmful_outputs_ids
                if 'ablated_harmless_outputs_ids' in locals(): del ablated_harmless_outputs_ids

            if 'ablated_harmful_outputs_ids' in locals(): del ablated_harmful_outputs_ids
            if 'ablated_harmless_outputs_ids' in locals(): del ablated_harmless_outputs_ids


            # calculate refusals for ablated outputs
            current_harmful_refusals = [is_refusal(t) for t in ablated_harmful_texts]
            current_harmless_refusals = [is_refusal(t) for t in ablated_harmless_texts]

            current_harmful_refusal_rate = torch.tensor(current_harmful_refusals, dtype=torch.float).mean()
            current_harmless_refusal_rate = torch.tensor(current_harmless_refusals, dtype=torch.float).mean()

            # calculate KL divergence for harmless prompts
            if ablated_harmless_logits_last is not None:
                try:
                    ablated_probs_harmless = torch.nn.functional.softmax(ablated_harmless_logits_last, dim=-1)
                    kl_divergence = kl_div(baseline_probs_harmless.to(ablated_probs_harmless.device), ablated_probs_harmless)
                except Exception as kl_e:
                    logger.error(f"Error calculating KL divergence for pos={pos}, layer={layer}: {kl_e}")
                    kl_divergence = torch.tensor(torch.finfo(torch.float32).max, device=self.device)
                finally:
                    del ablated_harmless_logits_last
                    if 'ablated_probs_harmless' in locals(): del ablated_probs_harmless
            else:
                 logger.warning(f"Could not calculate KL divergence for pos={pos}, layer={layer} due to missing ablated logits.")

            # compute overall score
            score_components = [current_harmful_refusal_rate, current_harmless_refusal_rate, kl_divergence]
            if any(torch.isnan(c).any() or torch.isinf(c).any() for c in score_components):
                logger.warning(f"NaN or Inf detected in score components for pos={pos}, layer={layer}. Assigning high penalty score.")
                overall_score = torch.tensor(1e6, device=self.device, dtype=torch.float32)
            else:
                refusal_diff_harmless = torch.abs(current_harmless_refusal_rate - baseline_harmless_refusal_rate)
                overall_score = current_harmful_refusal_rate + refusal_diff_harmless + kl_weight * kl_divergence

            current_score_float = overall_score.item() if not (torch.isnan(overall_score).any() or torch.isinf(overall_score).any()) else float('inf')

            # log overall pos/layer scores
            overall_results_log.append({
                'position': pos,
                'layer': layer,
                'baseline_harmless_refusal_rate': baseline_harmless_refusal_rate,
                'harmful_refusal_rate': current_harmful_refusal_rate.item() if not error_msg else 1.0,
                'harmless_refusal_rate': current_harmless_refusal_rate.item() if not error_msg else 1.0,
                'kl_divergence': kl_divergence.item() if not error_msg else float('inf'),
                'overall_score': current_score_float,
                'error': error_msg,
                'fitting_time': time.time() - fitting_start_time,
                'total_time': time.time() - start_time_include_loading
            })

            # log prompts and scores
            for i, prompt in enumerate(formatted_harmful_prompts):
                detailed_results_log.append({
                    'position': pos,
                    'layer': layer,
                    'prompt_type': 'harmful',
                    'prompt_index': i,
                    'prompt': prompt,
                    'baseline_response': baseline_harmful_texts[i],
                    'baseline_refusal': baseline_harmful_refusals[i],
                    'ablated_response': ablated_harmful_texts[i] if not error_msg else "GENERATION_ERROR",
                    'ablated_refusal': current_harmful_refusals[i],
                    'error': error_msg
                })
            for i, prompt in enumerate(formatted_harmless_prompts):
                 detailed_results_log.append({
                    'position': pos,
                    'layer': layer,
                    'prompt_type': 'harmless',
                    'prompt_index': i,
                    'prompt': prompt,
                    'baseline_response': baseline_harmless_texts[i],
                    'baseline_refusal': baseline_harmless_refusals[i],
                    'ablated_response': ablated_harmless_texts[i] if not error_msg else "GENERATION_ERROR",
                    'ablated_refusal': current_harmless_refusals[i],
                    'error': error_msg
                })

            # update best direction
            if current_score_float < best_score:
                 if not (torch.isnan(overall_score).any() or torch.isinf(overall_score).any()):
                    best_score = current_score_float
                    best_direction = direction.detach().cpu()
                    best_pos_layer = (pos, layer)
                 else:
                     logger.warning(f"Skipping update for best_score due to NaN/Inf in overall_score for pos={pos}, layer={layer}")

        # sort overall results for logging top directions
        overall_results_log.sort(key=lambda x: float('inf') if x['overall_score'] is None or pd.isna(x['overall_score']) or x['overall_score'] == float('inf') else x['overall_score'])

        logger.info("Top 5 directions based on generation evaluation:")
        for i, result in enumerate(overall_results_log[:5]):
             logger.info(f"#{i+1}: pos={result['position']}, layer={result['layer']}, "
                        f"score={result['overall_score']:.4f}, "
                        f"harmful_refusal={result['harmful_refusal_rate']:.4f}, "
                        f"harmless_refusal={result['harmless_refusal_rate']:.4f}, "
                        f"kl_div={result['kl_divergence']:.4f}" +
                        (f", error={result['error']}" if result['error'] else ""))

        # save logs to csv
        log_dir = "logs/diff_in_means_fit"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_name_safe = model_path.replace("/", "_") # Make model path safe for filename
        overall_log_path = os.path.join(log_dir, f"{model_name_safe}_overall_results_{timestamp}.csv")
        detailed_log_path = os.path.join(log_dir, f"{model_name_safe}_detailed_results_{timestamp}.csv")

        try:
            overall_df = pd.DataFrame(overall_results_log)
            overall_df.to_csv(overall_log_path, index=False)
            logger.info(f"Saved overall direction results to {overall_log_path}")

            detailed_df = pd.DataFrame(detailed_results_log)
            detailed_df.to_csv(detailed_log_path, index=False)
            logger.info(f"Saved detailed prompt results to {detailed_log_path}")
        except Exception as e:
            logger.error(f"Failed to save logs to CSV: {e}")


        if best_pos_layer is None:
             del model
             if torch.cuda.is_available(): torch.cuda.empty_cache()
             gc.collect()
             raise RuntimeError("Failed to find a best direction. Evaluation loop did not update the best score.")

        pos, layer = best_pos_layer
        logger.info(f"Selected best direction from position {pos}, layer {layer} with score {best_score:.4f}")

        if best_direction is None or torch.isnan(best_direction).any() or torch.isinf(best_direction).any():
             del model
             if torch.cuda.is_available(): torch.cuda.empty_cache()
             gc.collect()
             raise RuntimeError(f"Selected best_direction is invalid (None, NaN, or Inf) for pos={pos}, layer={layer}")

        self.directions[model_path] = best_direction
        self.save_directions()

        logger.info("Cleaning up model and final cache...")
        del model
        del best_direction
        del candidate_directions
        del baseline_probs_harmless
        # del baseline_harmful_logits_last, TODO: find if something else is not deleted
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.directions[model_path]
        
