
from jailbreaks.methods.base_method import GenerationExploit
from jailbreaks.utils.refusal import is_refusal
from typing import Dict, Any, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

import pandas as pd
import hashlib
import os

import logging

logger = logging.getLogger("OutputAware")

class OutputAware(GenerationExploit):
    def __init__(self, params: Dict[str, Any] = None, output_scorer=None, path: str = None):
        super().__init__()
        self.name = "output_aware"
        self.params = params or {
            "top_k": [1, 3, 5, 7],
            "top_p": [0.2, 0.4, 0.6, 0.8],
            "temperature": [0.1, 0.3, 0.7, 1.0]
        }
        self.output_scorer = lambda output: 0 if is_refusal(output) else 1
        self.path = path or self.name
        
    
    def generate_single(self, model: Union[AutoModelForCausalLM, HookedTransformer], tokenizer: AutoTokenizer, inputs, save_generations: bool = False, **kwargs) -> str:
        best_output = ""
        best_score = -1
        
        # if isinstance(inputs, str):
        #     original_prompt = inputs
        #     tokenized_inputs = tokenize_prompts([inputs], tokenizer)
        # else:
        tokenized_inputs = inputs
        original_prompt = tokenizer.decode(tokenized_inputs["input_ids"][0], skip_special_tokens=True)
        
        logger.debug(f"{self.name}: Trying parameters with {sum(len(values) for values in self.params.values())} combinations")

        generations = []
        
        for param_name, param_values in self.params.items():
            for value in param_values:
                generation_kwargs = kwargs.copy()
                generation_kwargs[param_name] = value
                
                logger.debug(f"Generating with {param_name}={value}")
                output_tokens = model.generate(**inputs, **generation_kwargs)
                output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                output_text = output_text[len(original_prompt):].strip()
                
                score = self.output_scorer(output_text)
                logger.debug(f"Score for {param_name}={value}: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_output = output_text
                    logger.debug(f"New best output (score: {score:.4f})")
                generations.append({
                    "prompt": original_prompt,
                    "response": output_text,
                    "score": score,
                    "generation_kwargs": generation_kwargs
                })
        if save_generations:
            df = pd.DataFrame(generations)
            if not os.path.exists(f"{self.path}/generations"):
                os.makedirs(f"{self.path}/generations")
            df.to_csv(f"{self.path}/generations/{hashlib.sha256(original_prompt.encode()).hexdigest()}.csv", index=False)
                    
        logger.debug(f"{self.name}: Best output had score {best_score:.4f}")
        return original_prompt + best_output