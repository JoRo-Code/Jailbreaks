from jailbreaks.methods.base_method import GenerationExploit
from jailbreaks.utils.refusal import is_refusal
from typing import Dict, Any, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
import logging

from tqdm import tqdm
import itertools

logger = logging.getLogger("OutputAware")

class OutputAware(GenerationExploit):
    def __init__(self, params: Dict[str, Any] = None, output_scorer=None, ):
        super().__init__()
        self.name = "output_aware"
        self.params = params or {
            "top_k": [1, 3, 5, 7],
            "top_p": [0.2, 0.4, 0.6, 0.8],
            "temperature": [0.1, 0.3, 0.7, 1.0]
        }
        self.output_scorer = lambda output: 0 if is_refusal(output) else 1
    
    def generate_batch(self, model: Union[AutoModelForCausalLM, HookedTransformer], tokenizer: AutoTokenizer, inputs: Dict[str, torch.Tensor], **kwargs) -> str:
        outputs = []
        print(inputs)
        batch_size = inputs["input_ids"].size(0)
        for i in range(batch_size):
            single_input = {k: v[i:i+1] for k, v in inputs.items()}
            outputs.append(self.generate(model, tokenizer, single_input, **kwargs))
        
        return torch.cat(outputs, dim=0)
    
    def generate(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, inputs: Dict[str, torch.Tensor], **kwargs) -> str:
        best_output_tokens = None
        best_output = ""
        best_score = -1
        
        toks = inputs["input_ids"]
        
        param_value_pairs = [(param_name, value) 
                            for param_name, param_values in self.params.items() 
                            for value in param_values]
        
        for param_name, value in tqdm(param_value_pairs, desc="Testing parameter values"):
            generation_kwargs = kwargs.copy()
            generation_kwargs[param_name] = value
            
            logger.debug(f"Generating with {param_name}={value}")
            
            output_tokens = model.generate(**inputs, **generation_kwargs)
            output_text = tokenizer.decode(output_tokens[0][toks.shape[1]:], skip_special_tokens=True)
            
            score = self.output_scorer(output_text)
            logger.debug(f"Score for {param_name}={value}: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_output_tokens = output_tokens
                best_output = output_text
                logger.debug(f"New best output (score: {score:.4f})")
                
        logger.debug(f"{self.name}: Best output had score {best_score:.4f}, output: {best_output}")
        return best_output_tokens