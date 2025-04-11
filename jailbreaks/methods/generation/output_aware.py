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

    def generate_batch(self, model: Union[AutoModelForCausalLM, HookedTransformer], tokenizer: AutoTokenizer, inputs: Dict[str, torch.Tensor], generation_id:str, **kwargs) -> str:
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