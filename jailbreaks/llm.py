
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict

from jailbreaks.methods.base_method import JailBreakMethod, PromptInjection, GenerationExploit, ModelManipulation

import logging
logger = logging.getLogger("LLM")

from jailbreaks.utils.model_loading import load_model, load_tokenizer

from jailbreaks.utils.tokenization import format_prompts, tokenize_prompts

class LLM:
    def __init__(self, model_path: str, methods: List[JailBreakMethod], device: str = "cpu"):
        self.model = load_model(model_path)
        self.name = model_path
        self.tokenizer = load_tokenizer(model_path)
        self.methods = methods
        self.device = device
    
    def to(self, device):
        if self.device != device:
            self.device = device
            if not hasattr(self.model, "device_map") or self.model.device_map is None:
                self.model = self.model.to(device)
        return self
    
    def _process_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        return self._process_prompts([prompt])[0]
    
    def _process_prompts(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        # # tokenized and batch prepared inputs
        # inputs = self.tokenizer(
        #     prompt, 
        #     return_tensors="pt", 
        #     padding=True, 
        #     truncation=True,
        #     max_length=2048
        # )
        # # move each batch to the correct device
        # return {k: v.to(self.device) for k, v in inputs.items()}
        inputs = tokenize_prompts(format_prompts(prompts, self.tokenizer), self.tokenizer)
        return inputs
    
    def _remove_prompt(self, prompt: str, response: str) -> str:
        return response[len(prompt):].strip()
    
    def prepare_prompt(self, prompt: str) -> str:
        for method in self.methods:
            if isinstance(method, PromptInjection):
                prompt = method.preprocess(prompt, self.model.config.name_or_path)
        return prompt

    def generate(self, prompt, decode:bool=True, return_input:bool=False, only_new_tokens:bool=True, **kwargs):
        prompt = self.prepare_prompt(prompt)
        inputs = tokenize_prompts(prompt, self.tokenizer)
        
        for method in self.methods:
            if isinstance(method, ModelManipulation):
                self.model = method.apply(self.name)
                # potentially add multi hook support
        
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
