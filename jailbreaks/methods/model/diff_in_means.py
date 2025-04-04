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
import pandas as pd
import time
import gc # Import garbage collector

from jailbreaks.utils import is_refusal

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

LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

def format_prompts(prompts: List[str], tokenizer: AutoTokenizer) -> List[str]:
    # Check if the tokenizer has the apply_chat_template method
    # if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
    #     logger.debug(f"Using tokenizer.apply_chat_template for {tokenizer.name_or_path}")
    #     # Format prompts using the standard method
    #     # Each prompt is treated as a single user message in a conversation
    #     formatted_prompts = [
    #         tokenizer.apply_chat_template(
    #             [{"role": "user", "content": prompt}],
    #             tokenize=False,
    #             add_generation_prompt=True 
    #         )
    #         for prompt in prompts
    #     ]
    #     return formatted_prompts
    # Fallback for Qwen or models without a standard chat template configured
    if tokenizer.name_or_path and "Qwen" in tokenizer.name_or_path:
        logger.warning(f"Using hardcoded Qwen chat template for {tokenizer.name_or_path}.")
        return [QWEN_CHAT_TEMPLATE.format(instruction=prompt) for prompt in prompts]
    else:
        # If no method and not Qwen, raise an error
        raise ValueError(
            f"Tokenizer {tokenizer.name_or_path} does not have a chat_template configured "
            f"and is not the hardcoded 'Qwen' type. Cannot format prompts."
        )

def tokenize_prompts(prompts: List[str], tokenizer: AutoTokenizer) -> List[str]:
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad token. Setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # No need to update model.tokenizer separately if using model.tokenizer directly
    if tokenizer.padding_side != 'left':
            logger.warning(f"Tokenizer padding side is '{tokenizer.padding_side}'. Forcing to 'left'.")
            tokenizer.padding_side = 'left'
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
    def __init__(self, path: str = None, use_cache: bool = True, hf_token: str = None):
        super().__init__()
        self.name = "diff-in-means"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.path = path or "jailbreaks/checkpoints/diff-in-means.pt"
        self.directions = self.load_directions(self.path) if use_cache else {}
        self.hf_token = hf_token
        
    def apply(self, model: AutoModelForCausalLM) -> str:
        original_generate = model.generate
        if not isinstance(model, HookedTransformer):
            model = self._convert_to_hooked_transformer(model)
        else:
            logger.warning("Model is already a HookedTransformer")

        model_path = hooked_transformer_path(model)
        direction = self.get_direction(model_path)
        hooks = generate_hooks(model, direction.to(self.device))
        
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
            trust_remote_code=True,
            use_auth_token=self.hf_token
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

    def fit(self, model_path: str, harmful_prompts: List[str], harmless_prompts: List[str], refit=False, generation_kwargs: Dict = None) -> str:
        if model_path in self.directions and not refit:
            logger.info(f"Skipping fitting for model {model_path} because it already exists")
            return self.get_direction(model_path)
        
        if not harmful_prompts or not harmless_prompts:
            raise ValueError("Harmful and harmless prompts lists cannot be empty for fitting.")

        logger.info(f"Loading model {model_path}")
        model = self.load_model(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto", use_auth_token=self.hf_token)

        # Ensure tokenizer has a padding token and correct padding side (as before)
        if tokenizer.pad_token is None:
            logger.warning("Tokenizer does not have a pad token. Setting to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            # No need to update model.tokenizer separately if using model.tokenizer directly
        if tokenizer.padding_side != 'left':
             logger.warning(f"Tokenizer padding side is '{tokenizer.padding_side}'. Forcing to 'left'.")
             tokenizer.padding_side = 'left'

        # --- Prepare Prompts and Tokens ---
        # Format prompts using the appropriate chat template
        formatted_harmful_prompts = format_prompts(harmful_prompts, tokenizer)
        formatted_harmless_prompts = format_prompts(harmless_prompts, tokenizer)

        # Tokenize prompts
        harmful_toks = tokenize_prompts(formatted_harmful_prompts, tokenizer).to(self.device)
        harmless_toks = tokenize_prompts(formatted_harmless_prompts, tokenizer).to(self.device)

        if harmful_toks.numel() == 0 or harmless_toks.numel() == 0:
             raise ValueError("Tokenization resulted in empty tensors. Check input prompts.")
        if harmful_toks.shape[1] == 0 or harmless_toks.shape[1] == 0:
             raise ValueError(f"Tokenization resulted in tensors with sequence length 0. Check input prompts and tokenizer padding.")

        # --- Baseline Evaluation (No Hooks) ---
        logger.info("Generating baseline responses...")
        default_generation_kwargs = {
            "max_new_tokens": 50,
            "eos_token_id": tokenizer.eos_token_id,
            #"pad_token_id": tokenizer.pad_token_id,
            "do_sample": False, # Use greedy decoding for reproducibility
        }
        if generation_kwargs:
            default_generation_kwargs.update(generation_kwargs)

        # --- Edit Start: Optimize initial caching and logit storage ---
        # Get baseline logits for KL divergence calculation later
        logger.info("Running baseline prompts with cache for KL div (optimizing memory)")

        # Only cache 'resid_pre' needed for direction calculation
        resid_pre_hook_name_filter = lambda name: name.endswith('resid_pre')

        # Run harmful prompts
        baseline_harmful_logits_full, harmful_cache = model.run_with_cache(
            harmful_toks,
            names_filter=resid_pre_hook_name_filter
        )
        # We don't need harmful logits, delete the full tensor
        del baseline_harmful_logits_full
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

        # Run harmless prompts
        baseline_harmless_logits_full, harmless_cache = model.run_with_cache(
            harmless_toks,
            names_filter=resid_pre_hook_name_filter
        )
        # Extract only the last token logits needed for KL div and delete the full tensor
        baseline_harmless_logits_last = baseline_harmless_logits_full[:, -1, :].float().detach() # Detach to be safe
        del baseline_harmless_logits_full
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

        baseline_probs_harmless = torch.nn.functional.softmax(baseline_harmless_logits_last, dim=-1)
        # --- Edit End ---


        # Generate baseline text for refusal checking (requires separate forward pass without cache)
        # This might still use significant memory during generation itself (KV cache)
        logger.info("Generating baseline text outputs...")
        baseline_harmful_outputs_ids = model.generate(harmful_toks, **default_generation_kwargs)
        baseline_harmless_outputs_ids = model.generate(harmless_toks, **default_generation_kwargs)

        # Decode baseline outputs (skip special tokens and prompt)
        baseline_harmful_texts = [tokenizer.decode(o[harmful_toks.shape[1]:], skip_special_tokens=True) for o in baseline_harmful_outputs_ids]
        baseline_harmless_texts = [tokenizer.decode(o[harmless_toks.shape[1]:], skip_special_tokens=True) for o in baseline_harmless_outputs_ids]

        # Calculate baseline refusal rates
        baseline_harmful_refusal_rate = torch.tensor([is_refusal(t) for t in baseline_harmful_texts], dtype=torch.float).mean().item()
        baseline_harmless_refusal_rate = torch.tensor([is_refusal(t) for t in baseline_harmless_texts], dtype=torch.float).mean().item()

        # Store baseline refusals for detailed logging
        baseline_harmful_refusals = [is_refusal(t) for t in baseline_harmful_texts]
        baseline_harmless_refusals = [is_refusal(t) for t in baseline_harmless_texts]

        logger.info(f"Baseline refusal rate (harmful): {baseline_harmful_refusal_rate:.4f}")
        logger.info(f"Baseline refusal rate (harmless): {baseline_harmless_refusal_rate:.4f}")

        # --- Candidate Direction Generation ---
        n_layers = model.cfg.n_layers
        n_positions = 1
        candidate_positions = list(range(-n_positions, 0))
        candidate_directions = {}

        if not candidate_positions:
             # --- Edit Start: Delete caches before raising error ---
             del harmful_cache
             del harmless_cache
             if torch.cuda.is_available(): torch.cuda.empty_cache()
             gc.collect()
             # --- Edit End ---
             raise ValueError(f"Could not determine candidate positions. Harmful tokens sequence length might be too short ({harmful_toks.shape[1]}).")

        logger.info("Calculating candidate directions...")
        for pos in candidate_positions:
            for layer in range(n_layers):
                # Ensure activations are on the correct device if they aren't already
                harmful_act = harmful_cache['resid_pre', layer][:, pos, :].mean(dim=0).to(self.device)
                harmless_act = harmless_cache['resid_pre', layer][:, pos, :].mean(dim=0).to(self.device)
                direction = harmful_act - harmless_act
                direction_norm = direction.norm()
                if direction_norm > 1e-8:
                    direction = direction / direction_norm
                    candidate_directions[(pos, layer)] = direction.detach() # Detach to potentially save memory if original acts are kept
                # else: # No need to store zero/small directions
                #    logger.warning(f"Direction norm is close to zero at pos={pos}, layer={layer}. Skipping.")

        # --- Edit Start: Delete caches after calculating directions ---
        logger.info("Deleting initial activation caches...")
        del harmful_cache
        del harmless_cache
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        # --- Edit End ---


        if not candidate_directions:
            raise ValueError("No valid candidate directions found. Check model activations or prompt differences.")

        logger.info(f"Evaluating {len(candidate_directions)} candidate directions by generating text...")

        # --- Evaluate Candidate Directions ---
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
            hooks = generate_hooks(model, direction.to(self.device)) # Ensure direction is on device

            ablated_harmful_texts = []
            ablated_harmless_texts = []
            ablated_harmless_logits_last = None
            kl_divergence = torch.tensor(torch.finfo(torch.float32).max, device=self.device)
            current_harmful_refusals = [True] * len(harmful_prompts)
            current_harmless_refusals = [True] * len(harmless_prompts)
            error_msg = None

            try:
                with model.hooks(hooks):
                    # Generate ablated text
                    ablated_harmful_outputs_ids = model.generate(harmful_toks, **default_generation_kwargs)
                    ablated_harmless_outputs_ids = model.generate(harmless_toks, **default_generation_kwargs)

                    # Decode ablated outputs
                    ablated_harmful_texts = [tokenizer.decode(o[harmful_toks.shape[1]:], skip_special_tokens=True) for o in ablated_harmful_outputs_ids]
                    ablated_harmless_texts = [tokenizer.decode(o[harmless_toks.shape[1]:], skip_special_tokens=True) for o in ablated_harmless_outputs_ids]

                    # --- Edit Start: Get ablated logits with standard forward pass and delete full tensor ---
                    # Get ablated logits for KL divergence using a standard forward pass
                    ablated_harmless_logits_full = model(harmless_toks) # No caching here
                    ablated_harmless_logits_last = ablated_harmless_logits_full[:, -1, :].float().detach() # Detach
                    del ablated_harmless_logits_full # Delete the large tensor immediately
                    # --- Edit End ---

            except Exception as e:
                logger.error(f"Error during hooked generation/logit calculation for pos={pos}, layer={layer}: {e}")
                error_msg = str(e)
                overall_score = torch.tensor(best_score + 1.0, device=self.device) # Ensure it's worse than current best
                # Explicitly clear potentially large tensors from the failed try block
                ablated_harmless_logits_last = None
                # --- Edit Start: Add cleanup in except block ---
                if 'ablated_harmless_logits_full' in locals(): del ablated_harmless_logits_full
                if 'ablated_harmful_outputs_ids' in locals(): del ablated_harmful_outputs_ids
                if 'ablated_harmless_outputs_ids' in locals(): del ablated_harmless_outputs_ids
                # --- Edit End ---

            # --- Edit Start: Add potential cleanup after try-except ---
            # Explicitly delete generation outputs if they exist to free memory before KL calc
            if 'ablated_harmful_outputs_ids' in locals(): del ablated_harmful_outputs_ids
            if 'ablated_harmless_outputs_ids' in locals(): del ablated_harmless_outputs_ids
            # --- Edit End ---


            # Calculate refusal status for ablated outputs
            current_harmful_refusals = [is_refusal(t) for t in ablated_harmful_texts]
            current_harmless_refusals = [is_refusal(t) for t in ablated_harmless_texts]

            current_harmful_refusal_rate = torch.tensor(current_harmful_refusals, dtype=torch.float).mean()
            current_harmless_refusal_rate = torch.tensor(current_harmless_refusals, dtype=torch.float).mean()

            # Calculate KL divergence for harmless prompts
            if ablated_harmless_logits_last is not None:
                try:
                    ablated_probs_harmless = torch.nn.functional.softmax(ablated_harmless_logits_last, dim=-1)
                    # Ensure baseline_probs_harmless is on the same device
                    kl_divergence = self.kl_div(baseline_probs_harmless.to(ablated_probs_harmless.device), ablated_probs_harmless)
                except Exception as kl_e:
                    logger.error(f"Error calculating KL divergence for pos={pos}, layer={layer}: {kl_e}")
                    kl_divergence = torch.tensor(torch.finfo(torch.float32).max, device=self.device) # Assign max KL on error
                finally:
                    # --- Edit Start: Delete ablated logits after use ---
                    del ablated_harmless_logits_last # Delete after KL calculation
                    if 'ablated_probs_harmless' in locals(): del ablated_probs_harmless
                    # --- Edit End ---
            else:
                 logger.warning(f"Could not calculate KL divergence for pos={pos}, layer={layer} due to missing ablated logits.")
                 # kl_divergence remains high

            # Compute overall score
            score_components = [current_harmful_refusal_rate, current_harmless_refusal_rate, kl_divergence]
            if any(torch.isnan(c).any() or torch.isinf(c).any() for c in score_components):
                logger.warning(f"NaN or Inf detected in score components for pos={pos}, layer={layer}. Assigning high penalty score.")
                overall_score = torch.tensor(1e6, device=self.device, dtype=torch.float32)
            else:
                refusal_diff_harmless = torch.abs(current_harmless_refusal_rate - baseline_harmless_refusal_rate)
                overall_score = current_harmful_refusal_rate + refusal_diff_harmless + kl_weight * kl_divergence

            current_score_float = overall_score.item() if not (torch.isnan(overall_score).any() or torch.isinf(overall_score).any()) else float('inf')

            # --- Log Overall Results ---
            overall_results_log.append({
                'position': pos,
                'layer': layer,
                'baseline_harmless_refusal_rate': baseline_harmless_refusal_rate,
                'harmful_refusal_rate': current_harmful_refusal_rate.item() if not error_msg else 1.0,
                'harmless_refusal_rate': current_harmless_refusal_rate.item() if not error_msg else 1.0,
                'kl_divergence': kl_divergence.item() if not error_msg else float('inf'),
                'overall_score': current_score_float,
                'error': error_msg
            })

            # --- Log Detailed Results ---
            # Log harmful prompts/results
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
            # Log harmless prompts/results
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


            # Update best direction logic
            if current_score_float < best_score:
                 if not (torch.isnan(overall_score).any() or torch.isinf(overall_score).any()):
                    best_score = current_score_float
                    # Store the best direction on CPU to save VRAM, move back to device when needed in apply()
                    best_direction = direction.detach().cpu()
                    best_pos_layer = (pos, layer)
                 else:
                     logger.warning(f"Skipping update for best_score due to NaN/Inf in overall_score for pos={pos}, layer={layer}")

            # --- Edit Start: Add periodic cleanup inside loop ---
            # Optional: Force garbage collection periodically if the loop is very long
            # and memory seems to creep up. This can add overhead.
            # if (i + 1) % 10 == 0: # Example: every 10 directions
            #    gc.collect()
            #    if torch.cuda.is_available(): torch.cuda.empty_cache()
            # --- Edit End ---


        # --- Final Selection, Logging, and Saving ---
        # Sort overall results for logging top directions
        overall_results_log.sort(key=lambda x: float('inf') if x['overall_score'] is None or pd.isna(x['overall_score']) or x['overall_score'] == float('inf') else x['overall_score'])

        logger.info("Top 5 directions based on generation evaluation:")
        for i, result in enumerate(overall_results_log[:5]):
             logger.info(f"#{i+1}: pos={result['position']}, layer={result['layer']}, "
                        f"score={result['overall_score']:.4f}, "
                        f"harmful_refusal={result['harmful_refusal_rate']:.4f}, "
                        f"harmless_refusal={result['harmless_refusal_rate']:.4f}, "
                        f"kl_div={result['kl_divergence']:.4f}" +
                        (f", error={result['error']}" if result['error'] else ""))

        # --- Save Logs to CSV ---
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
             # ... (error handling remains the same) ...
             # --- Edit Start: Ensure model deletion on error path ---
             del model
             if torch.cuda.is_available(): torch.cuda.empty_cache()
             gc.collect()
             # --- Edit End ---
             raise RuntimeError("Failed to find a best direction. Evaluation loop did not update the best score.")

        pos, layer = best_pos_layer
        logger.info(f"Selected best direction from position {pos}, layer {layer} with score {best_score:.4f}")

        if best_direction is None or torch.isnan(best_direction).any() or torch.isinf(best_direction).any():
             # --- Edit Start: Ensure model deletion on error path ---
             del model
             if torch.cuda.is_available(): torch.cuda.empty_cache()
             gc.collect()
             # --- Edit End ---
             raise RuntimeError(f"Selected best_direction is invalid (None, NaN, or Inf) for pos={pos}, layer={layer}")

        # Best direction is already on CPU from the loop optimization
        self.directions[model_path] = best_direction
        self.save_directions() # Save immediately after finding

        # Clean up model to free memory
        logger.info("Cleaning up model and final cache...")
        del model
        del best_direction # Remove reference
        del candidate_directions # Remove reference
        del baseline_probs_harmless # Remove reference
        # baseline_harmful_logits_last was never stored
        del baseline_harmless_logits_last # Remove reference
        gc.collect() # Python garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # PyTorch CUDA cache clear

        return self.directions[model_path] # Return the CPU tensor
    
    def fit_and_generate(self, model_path: str, harmful_prompts: List[str], harmless_prompts: List[str], prompts: List[str], layer: int = 14, pos: int = -1, **kwargs) -> str:
        
        model = self.load_model(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto", use_auth_token=self.hf_token)

        formatted_harmful_prompts = format_prompts(harmful_prompts, tokenizer)
        formatted_harmless_prompts = format_prompts(harmless_prompts, tokenizer)

        # Tokenize prompts
        harmful_toks = tokenize_prompts(formatted_harmful_prompts, tokenizer).to(self.device)
        harmless_toks = tokenize_prompts(formatted_harmless_prompts, tokenizer).to(self.device)
        default_generation_kwargs = {
            "max_new_tokens": 50,
            "eos_token_id": tokenizer.eos_token_id,
            #"pad_token_id": tokenizer.pad_token_id,
            "do_sample": False, # Use greedy decoding for reproducibility
        }

        resid_pre_hook_name_filter = lambda name: name.endswith('resid_pre')

        baseline_harmful_logits_full, harmful_cache = model.run_with_cache(
            harmful_toks,
            names_filter=resid_pre_hook_name_filter
        )
        # We don't need harmful logits, delete the full tensor
        del baseline_harmful_logits_full
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

        # Run harmless prompts
        baseline_harmful_logits_full, harmless_cache = model.run_with_cache(
            harmless_toks,
            names_filter=resid_pre_hook_name_filter
        )
        del baseline_harmful_logits_full
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

        harmful_act = harmful_cache['resid_pre', layer][:, pos, :].mean(dim=0).to(self.device)
        harmless_act = harmless_cache['resid_pre', layer][:, pos, :].mean(dim=0).to(self.device)
        direction = harmful_act - harmless_act
        direction_norm = direction.norm()
        if direction_norm > 1e-8:
            direction = direction / direction_norm

        hooks = generate_hooks(model, direction.to(self.device))
        del harmful_cache
        del harmless_cache
        del harmful_toks
        del harmless_toks
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        
        formatted_prompts = format_prompts(prompts, tokenizer)
        toks = tokenize_prompts(formatted_prompts, tokenizer).to(self.device)
        
        baseline_outputs_ids = model.generate(toks, **default_generation_kwargs)
        with model.hooks(hooks):
            ablated_outputs_ids = model.generate(toks, **default_generation_kwargs)
        
        
        ablated_responses = [tokenizer.decode(o[toks.shape[1]:], skip_special_tokens=True) for o in ablated_outputs_ids]
        baseline_responses = [tokenizer.decode(o[toks.shape[1]:], skip_special_tokens=True) for o in baseline_outputs_ids]
        prompt_response_pairs = [
            {"prompt": original_prompt, "ablated_response": ablated_response, "baseline_response": baseline_response}
            for original_prompt, ablated_response, baseline_response in zip(prompts, ablated_responses, baseline_responses)
        ]
        
        del model
        del direction
        del toks
        del ablated_outputs_ids
        del ablated_responses # Delete the intermediate list too
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        
        return prompt_response_pairs
        
        
    def generate(self, model_path: str, inputs: Dict[str, torch.Tensor], **kwargs) -> str:
        model = self.load_model(model_path)
        hooks = self.get_hooks(model)
        with model.hooks(hooks):
            output = model.generate(inputs, **kwargs)
        return model.tokenizer.decode(output[0])
    
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
        # The stored direction is on CPU, move it to self.device here
        return self.directions[model_path].to(self.device)
    
    def get_hooks(self, model: HookedTransformer) -> List[Tuple[str, Callable]]:
        model_path = hooked_transformer_path(model)
        if model_path not in self.directions:
            raise ValueError(f"No direction found for model {model_path}")
        return generate_hooks(model, self.directions[model_path])