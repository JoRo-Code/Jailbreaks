
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict

from jailbreaks.methods.base_method import JailBreakMethod, PromptInjection, GenerationExploit, ModelManipulation

import logging
logger = logging.getLogger(__name__)

from jailbreaks.utils.model_loading import load_model, load_tokenizer

from jailbreaks.utils.tokenization import format_prompts, tokenize_prompts
import gc

class LLM:
    def __init__(self, model_path: str, methods: List[JailBreakMethod]):
        self.model = load_model(model_path)
        self.name = model_path
        self.tokenizer = load_tokenizer(model_path)
        self.methods = methods
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def to(self, device):
        if self.device != device:
            self.device = device
            if not hasattr(self.model, "device_map") or self.model.device_map is None:
                self.model = self.model.to(device)
        return self
    
    def clear_cache(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
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
    
    def prepare_prompt(self, prompt: str) -> str:
        for method in self.methods:
            if isinstance(method, PromptInjection):
                prompt = method.preprocess(prompt, self.model.config.name_or_path)
        prompt = format_prompts([prompt], self.tokenizer)[0]
        # TODO: fix prefix injection with chat templates
        return prompt
    
    def generate(self, prompt, **kwargs):
        return self.generate_batch([prompt], **kwargs)[0]

    def generate_batch(self, prompts, **kwargs):
        # prepare
        prompts = [self.prepare_prompt(prompt) for prompt in prompts]
        toks = tokenize_prompts(prompts, self.tokenizer).to(self.device)
        inputs = toks.input_ids
        
        # manipulations
        for method in self.methods:
            if isinstance(method, ModelManipulation):
                self.model = method.apply(self.name)
                # TODO: only apply once
                # potentially add multi hook support
        from transformer_lens import HookedTransformer
        if isinstance(self.model, HookedTransformer):
            logger.debug(f"Using HookedTransformer")
        else:
            logger.debug(f"Using AutoModelForCausalLM")
        
        # generation
        output_tokens = None
        for method in self.methods:
            if isinstance(method, GenerationExploit):
                output_tokens = method.generate_batch(self.model, self.tokenizer, toks, self._model_method_combo(), **kwargs)
        if output_tokens is None:
            output_tokens = self.model.generate(inputs, **kwargs)
                
        # decode
        output = [self.tokenizer.decode(o[inputs.shape[1]:], skip_special_tokens=True) for o in output_tokens]
        
        return output
    
    def _model_method_combo(self):
        return self.name + "_" + "_".join([method.__class__.__name__ for method in self.methods])
        
        
        

