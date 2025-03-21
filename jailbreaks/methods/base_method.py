from abc import ABC
from typing import Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

class GenerationExploit(JailBreakMethod):
    def __init__(self):
        super().__init__()

    def generate(self, model: AutoModelForCausalLM, inputs: Dict[str, torch.Tensor], tokenizer: AutoTokenizer, **kwargs) -> str:
        # generate textual response
        output_tokens = model.generate(**inputs, **kwargs)
        output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return output_text

