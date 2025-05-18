import logging
from typing import Dict, Any, Optional, Literal, List, Tuple
import re

from datasets import Dataset

from jailbreaks.pipeline.schemas import GeneratedResponse
from jailbreaks.evaluators.base import ResponseEvaluator

logger = logging.getLogger(__name__)

class UtilityEvaluator(ResponseEvaluator):
    """Evaluator for utility tasks like MMLU or Hellaswag."""
    
    def __init__(
        self,
        dataset_type: Literal["mmlu", "hellaswag"],
        dataset: Dataset,
        name: Optional[str] = None,
    ):
        self.dataset_type = dataset_type
        self.dataset = dataset
        
        super().__init__(name=name or f"{dataset_type}-accuracy")
    
    def evaluate(self, responses: List[GeneratedResponse]) -> List[Dict[str, Any]]:
        samples = [self.evaluate_single(resp.prompt, resp.response) for resp in responses]
        metrics = {
            "accuracy": sum(sample["is_correct"] for sample in samples) / len(samples),
        }
        return metrics, samples
    
    def evaluate_single(self, prompt: Dict[str, Any], response: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        
        original_sample = self._find_matching_sample(prompt)
        extracted_answer = self._extract_answer(response)
        
        if self.dataset_type == "mmlu":
            correct_answer_idx = original_sample.get("answer")
            correct_answer = chr(65 + correct_answer_idx) if correct_answer_idx is not None else None
            is_correct = self._evaluate_mmlu_response(original_sample, extracted_answer)
        elif self.dataset_type == "hellaswag":
            correct_answer_idx = int(original_sample.get("label", -1))
            correct_answer = chr(65 + correct_answer_idx) if correct_answer_idx >= 0 else None
            is_correct = self._evaluate_hellaswag_response(original_sample, extracted_answer)
        else:
            raise ValueError(f"Invalid dataset type: {self.dataset_type}")
        
        return {
            "prompt": prompt,
            "response": response,
            "accuracy": 1.0 if is_correct else 0.0,
            "extracted_answer": extracted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
        }
    
    def _find_matching_sample(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find the original sample in the dataset that matches the prompt.
        
        Args:
            prompt: The prompt dictionary
        
        Returns:
            The matching original sample from the dataset
        """
        prompt_text = prompt.get("text", "")
        
        for sample in self.dataset:
            if self.dataset_type == "mmlu":
                question = sample.get("question", "")
                if question in prompt_text:
                    return sample
            if self.dataset_type == "hellaswag":
                context = sample.get("ctx", "")
                if context in prompt_text:
                    return sample
        
        return None
    
    def _extract_answer(self, response: str) -> Optional[str]:
        """
        Extract the answer (A, B, C, or D) from the response.
        
        Args:
            response: The model's response
        
        Returns:
            The extracted answer or None if no answer was found
        """
        patterns = [
            r"(?:answer|option|select|choose|choice)(?:\s+is)?(?:\s*:)?\s*([A-Da-d])",
            r"([A-Da-d])(?:\s*\.|$)",
            r"^([A-Da-d])$",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return None
    
    def _evaluate_mmlu_response(self, example, response):
        """Evaluate if a response is correct for MMLU."""
        if response is None:
            return False
            
        response = response.strip().upper()
        if response == 'A':
            selected_idx = 0
        elif response == 'B':
            selected_idx = 1
        elif response == 'C':
            selected_idx = 2
        elif response == 'D':
            selected_idx = 3
        else:
            return False
        
        return selected_idx == int(example.get('answer'))
    
    def _evaluate_hellaswag_response(self, example, response):
        """Evaluate if a response is correct for Hellaswag."""
        if response is None:
            return False

        response = response.strip().upper()
        if response == 'A':
            selected_idx = 0
        elif response == 'B':
            selected_idx = 1
        elif response == 'C':
            selected_idx = 2
        elif response == 'D':
            selected_idx = 3
        else:
            return False
        
        return selected_idx == int(example.get('label', -1))