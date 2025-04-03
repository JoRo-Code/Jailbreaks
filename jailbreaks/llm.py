
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict

from jailbreaks.methods.base_method import JailBreakMethod, PromptInjection, GenerationExploit

import logging
logger = logging.getLogger("LLM")

class LLM:
    def __init__(self, model: AutoModelForCausalLM, methods: List[JailBreakMethod], device: str = "cpu"):
        self.model = model.to(device)
        self.name = getattr(model.config, "name_or_path", None) or model.config._name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.methods = methods
        self.device = device
    
    def to(self, device):
        # Only move if needed
        if self.device != device:
            self.device = device
            if not hasattr(self.model, "device_map") or self.model.device_map is None:
                self.model = self.model.to(device)
        return self

    def _process_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        
        logger.debug(f"modified prompt: {prompt}")
        # tokenized and batch prepared inputs
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048
        )
        # move each batch to the correct device
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def _remove_prompt(self, prompt: str, response: str) -> str:
        return response[len(prompt):].strip()

    def generate(self, prompt, decode:bool=True, return_input:bool=False, only_new_tokens:bool=True, **kwargs):
        for method in self.methods:
            if isinstance(method, PromptInjection):
                prompt = method.preprocess(prompt, self.model.config.name_or_path)
    
        inputs = self._process_prompt(prompt)
        
        output_text = None
        for method in self.methods:
            if isinstance(method, GenerationExploit):
                output_text = method.generate(
                    model=self.model,
                    inputs=inputs,
                    tokenizer=self.tokenizer,
                    **kwargs
                )
        
        if not output_text:
            output_tokens = self.model.generate(**inputs, **kwargs)
            output_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        if only_new_tokens:
            output_text = self._remove_prompt(prompt, output_text)
        
        output = output_text
        if not decode:
            output = self.tokenizer.encode(output_text, add_special_tokens=True)
        
        logger.debug(f"output: {output_text}\n\n")
        
        if return_input:
            return output, prompt
        else:
            return output
