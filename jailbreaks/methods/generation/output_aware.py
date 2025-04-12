from jailbreaks.methods.base_method import GenerationExploit
from jailbreaks.utils.refusal import is_refusal
from jailbreaks.methods.utils import format_name
from typing import Dict, Any, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
import logging

from tqdm import tqdm
import itertools
import pandas as pd
import os
import hashlib

logger = logging.getLogger(__name__)

class OutputAware(GenerationExploit):
    def __init__(self, params: Dict[str, Any] = None, output_scorer=None, path: str = None, save: bool = False, description: str = "", verbose: bool = False):
        super().__init__()
        self.name = format_name(self.__class__.__name__,description)
        self.params = params or {
            "top_k": [1, 3, 5, 7],
            "top_p": [0.2, 0.4, 0.6, 0.8],
            "temperature": [0.1, 0.3, 0.7, 1.0]
        }
        self.output_scorer = lambda output: 0 if is_refusal(output) else 1
        self.path = path or self.name
        self.save = save
        self.verbose = verbose    
        
    def slow_batch(self, model: Union[AutoModelForCausalLM, HookedTransformer], tokenizer: AutoTokenizer, inputs: Dict[str, torch.Tensor], generation_id:str, **kwargs) -> str:
        outputs = []
        batch_size = inputs["input_ids"].size(0)
        for i in range(batch_size):
            single_input = {k: v[i:i+1] for k, v in inputs.items()}
            outputs.append(self.generate(model, tokenizer, single_input, generation_id, **kwargs))
        
        # Find the maximum length among all output tensors
        max_length = max(output.size(1) for output in outputs)
        
        # Pad each tensor to the maximum length
        padded_outputs = []
        for output in outputs:
            pad_length = max_length - output.size(1)
            if pad_length > 0:
                padding = torch.full((1, pad_length), tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0, 
                                    dtype=output.dtype, device=output.device)
                padded_output = torch.cat([output, padding], dim=1)
            else:
                padded_output = output
            padded_outputs.append(padded_output)
        
        combined_outputs = torch.cat(padded_outputs, dim=0)
        return combined_outputs
    
    
    
    def generate(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, inputs: Dict[str, torch.Tensor], generation_id:str, **kwargs) -> str:
        best_output_tokens = None
        best_output = ""
        best_score = -1
        
        toks = inputs["input_ids"]
        original_prompt = tokenizer.decode(toks[0], skip_special_tokens=True)
        
        param_value_pairs = [(param_name, value) 
                            for param_name, param_values in self.params.items() 
                            for value in param_values]
        generations = []
        param_iterator = tqdm(param_value_pairs, desc="Testing parameter values") if self.verbose else param_value_pairs

        for param_name, value in param_iterator:
            generation_kwargs = kwargs.copy()
            generation_kwargs["do_sample"] = True
            generation_kwargs[param_name] = value
            
            logger.debug(f"Generating with {param_name}={value}")
            
            output_tokens = model.generate(**inputs, **generation_kwargs)
            output_text = tokenizer.decode(output_tokens[0][toks.shape[1]:], skip_special_tokens=True)
            
            score = self.output_scorer(output_text)
            logger.debug(f"Score for {param_name}={value}: {score:.4f}")
            
            generations.append({
                "prompt": original_prompt,
                "response": output_text,
                "score": score,
                "generation_kwargs": generation_kwargs
            })
            
            if score > best_score:
                best_score = score
                best_output_tokens = output_tokens
                best_output = output_text
                logger.debug(f"New best output (score: {score:.4f})")
        
        if self.save:
            df = pd.DataFrame(generations)
            if not os.path.exists(f"{self.path}/{generation_id}"):
                os.makedirs(f"{self.path}/{generation_id}")
            df.to_csv(f"{self.path}/{generation_id}/{hashlib.sha256(original_prompt.encode()).hexdigest()}.csv", index=False)
                
        logger.debug(f"{self.name}: Best output had score {best_score:.4f}, output: {best_output}")
        return best_output_tokens
    
    def generate_batch(self, model: Union[AutoModelForCausalLM, HookedTransformer], tokenizer: AutoTokenizer, inputs: Dict[str, torch.Tensor], generation_id:str, **kwargs) -> str:
        return self.speedy_batch(model, tokenizer, inputs, generation_id, **kwargs)
    
    def speedy_batch(self, model: Union[AutoModelForCausalLM, HookedTransformer], tokenizer: AutoTokenizer, inputs: Dict[str, torch.Tensor], generation_id:str, **kwargs) -> str:

        batch_size = inputs["input_ids"].size(0)
        
        # Set up storage for best scores and corresponding tokens (one per prompt).
        best_scores = [-float("inf")] * batch_size
        best_output_tokens = [None] * batch_size
        generations = []  # To store details about all generation attempts.
        
        # Flatten parameters into a list of (param, value) pairs.
        param_value_pairs = [(param_name, value) 
                            for param_name, param_values in self.params.items() 
                            for value in param_values]
        
        # Optionally add a progress bar if verbose mode is enabled.
        iterator = tqdm(param_value_pairs, desc="Testing parameter values") if self.verbose else param_value_pairs
        
        # Iterate over each parameter-value combination.
        for param_name, value in iterator:
            generation_kwargs = kwargs.copy()
            generation_kwargs[param_name] = value
            logger.debug(f"Generating for batch with {param_name}={value}")
            
            # Generate outputs for the entire batch at once.
            batch_outputs = model.generate(**inputs, **generation_kwargs)
            # Assume batch_outputs is a tensor of shape (batch_size, seq_length).
            
            # For each prompt in the batch, decode only the generated continuation (after the prompt).
            for i in range(batch_size):
                prompt_ids = inputs["input_ids"][i]
                prompt_length = prompt_ids.shape[0]
                # Extract tokens beyond the prompt.
                gen_tokens = batch_outputs[i][prompt_length:]
                output_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                score = self.output_scorer(output_text)
                
                logger.debug(f"Score for batch index {i} with {param_name}={value}: {score:.4f}")
                generations.append({
                    "prompt": tokenizer.decode(prompt_ids, skip_special_tokens=True),
                    "response": output_text,
                    "score": score,
                    "generation_kwargs": generation_kwargs
                })
                
                # Update best generation for this prompt if this score is higher.
                if score > best_scores[i]:
                    best_scores[i] = score
                    # Keep as a (1, seq_length) tensor slice.
                    best_output_tokens[i] = batch_outputs[i:i+1]
                    logger.debug(f"New best output for index {i} (score: {score:.4f})")
        
        # Optionally save all the generation details.
        if self.save:
            df = pd.DataFrame(generations)
            out_dir = os.path.join(self.path, generation_id)
            os.makedirs(out_dir, exist_ok=True)
            # Combine all prompts into a single string to compute a hash key.
            all_prompts = " ".join(tokenizer.decode(inputs["input_ids"][i], skip_special_tokens=True)
                                    for i in range(batch_size))
            filename = os.path.join(out_dir, f"{hashlib.sha256(all_prompts.encode()).hexdigest()}.csv")
            df.to_csv(filename, index=False)
        
        logger.debug(f"{self.name}: Best scores for batch: {best_scores}")
        
        # Pad each best output if their sequence lengths differ.
        max_length = max(token_tensor.size(1) for token_tensor in best_output_tokens if token_tensor is not None)
        padded_outputs = []
        for token_tensor in best_output_tokens:
            if token_tensor is None:
                # In the rare case that no generation was produced,
                # create an empty tensor padded with the pad token.
                padded = torch.full((1, max_length),
                                    tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
            else:
                seq_length = token_tensor.size(1)
                if seq_length < max_length:
                    pad_length = max_length - seq_length
                    padding = torch.full((1, pad_length),
                                        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
                                        dtype=token_tensor.dtype,
                                        device=token_tensor.device)
                    padded = torch.cat([token_tensor, padding], dim=1)
                else:
                    padded = token_tensor
            padded_outputs.append(padded)
        
        # Concatenate best outputs from each example into a single tensor.
        combined_outputs = torch.cat(padded_outputs, dim=0)
        return combined_outputs
