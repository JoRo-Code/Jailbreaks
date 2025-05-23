from abc import ABC
from typing import Dict

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)

class JailBreakMethod(ABC):
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__

    def preprocess(self, prompt: str) -> str:
        return prompt
    
    def __str__(self):
        return self.name

class PromptInjection(JailBreakMethod):
    def __init__(self):
        super().__init__()

class ModelManipulation(JailBreakMethod):
    def __init__(self):
        super().__init__()
    
    def apply(self, model: AutoModelForCausalLM, inputs: Dict[str, torch.Tensor], **kwargs) -> str:
        pass

class GenerationExploit(JailBreakMethod):
    def __init__(self):
        super().__init__()

    def generate(self, model: AutoModelForCausalLM, inputs: Dict[str, torch.Tensor], tokenizer: AutoTokenizer, **kwargs) -> str:
        # old + new tokens
        return model.generate(**inputs, **kwargs)

