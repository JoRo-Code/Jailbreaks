# jailbreaks/benchmarks/utility_benchmark.py

import logging
from typing import List, Dict, Any, Optional, Union, Literal
import random
from datasets import load_dataset

from jailbreaks.benchmarks.base import Benchmark

logger = logging.getLogger(__name__)

class UtilityBenchmark(Benchmark):
    """Benchmark for utility tasks like MMLU or Hellaswag."""
    
    def __init__(
        self,
        dataset_type: Literal["mmlu", "hellaswag"],
        subject: Optional[str] = None,
        split: str = "validation",
        max_new_tokens: int = 32,
        num_samples: Optional[int] = 100,
        name: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Initialize a utility benchmark.
        
        Args:
            dataset_type: Type of dataset ("mmlu" or "hellaswag")
            subject: For MMLU, the specific subject to use (e.g., "abstract_algebra")
            split: Dataset split to use
            max_new_tokens: Maximum number of tokens to generate
            num_samples: Number of samples to use (None for all)
            name: Name of the benchmark
            seed: Random seed for sampling
        """
        self.dataset_type = dataset_type
        self.subject = subject
        self.split = split
        
        # Load the dataset
        if dataset_type == "mmlu":
            if subject is None:
                raise ValueError("Subject must be specified for MMLU")
            ds = load_dataset("cais/mmlu", subject)
        elif dataset_type == "hellaswag":
            ds = load_dataset("Rowan/hellaswag")
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Get the prompts
        dataset = ds[split]
        
        if num_samples is not None and num_samples < len(dataset):
            random.seed(seed)
            indices = random.sample(range(len(dataset)), num_samples)
            samples = [dataset[i] for i in indices]
        else:
            samples = dataset
        
        self.samples = samples
        
        prompts = []
        prompt_metadata = []
        for sample in samples:
            if dataset_type == "mmlu":
                prompt = self._create_mmlu_prompt(sample)
                idx = sample.get("idx", None)
                answer = sample.get("answer", None)
            else:  # hellaswag
                prompt = self._create_hellaswag_prompt(sample)
                idx = sample.get("source_id", None)
                answer = sample.get("label", None)
        
            prompts.append(prompt)
            prompt_metadata.append({
                "prompt": prompt,
                "id": idx,
                "answer": answer,
            })
        
        # Generate a name if not provided
        if name is None:
            if dataset_type == "mmlu":
                name = f"mmlu-{subject}-{split}"
            else:
                name = f"hellaswag-{split}"
        
        super().__init__(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            name=name
        )
        
        self.prompt_metadata = prompt_metadata
    
    def get_prompts_with_metadata(self):
        return self.prompt_metadata
    
    def is_utility_benchmark(self):
        return True
    
    def get_dataset(self):
        return self.dataset
    
    def _create_mmlu_prompt(self, example):
        """Create a prompt from an MMLU example."""
        prompt = f"Question: {example['question']}\n\n"
        prompt += "Options:\n"
        for i, choice in enumerate(example['choices']):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "\nPlease select the correct answer (A, B, C, or D). Answer with a single letter."
        return prompt
    
    def _create_hellaswag_prompt(self, example):
        """Create a prompt from a Hellaswag example."""
        prompt = f"Context: {example['ctx']}\n\n"
        prompt += "Options:\n"
        for i, ending in enumerate(example['endings']):
            prompt += f"{chr(65+i)}. {ending}\n"
        prompt += "\nWhich option is the most plausible continuation? Please select the correct answer (A, B, C, or D). Answer with a single letter."
        return prompt
