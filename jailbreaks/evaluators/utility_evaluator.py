import logging
from typing import Dict, Any, Optional, Literal, List
import re

from jailbreaks.pipeline.schemas import GeneratedResponse
from jailbreaks.evaluators.base import ResponseEvaluator

logger = logging.getLogger(__name__)

class UtilityEvaluator(ResponseEvaluator):
    """Evaluator for utility tasks like MMLU or Hellaswag."""
    
    def __init__(
        self,
        dataset_type: Literal["mmlu", "hellaswag"],
        name: Optional[str] = None,
    ):
        """
        Initialize a utility evaluator.
        
        Args:
            dataset_type: Type of dataset ("mmlu" or "hellaswag")
            name: Name of the evaluator
        """
        self.dataset_type = dataset_type
        
        # Generate a name if not provided
        if name is None:
            name = f"{dataset_type}-accuracy"
        
        super().__init__(name=name)
    
    def evaluate(self, responses: List[GeneratedResponse]) -> Dict[str, Any]:
        return [self.evaluate_single(resp.prompt, resp.response) for resp in responses]
    
    def evaluate_single(self, prompt: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate a response to a utility prompt.
        
        Args:
            prompt: The prompt dictionary containing the original sample
            response: The model's response
        
        Returns:
            A dictionary with evaluation results
        """
        original_sample = prompt.get("original_sample", {})
        
        # Extract the answer from the response
        extracted_answer = self._extract_answer(response)
        
        if self.dataset_type == "mmlu":
            correct_answer_idx = original_sample.get("answer")
            correct_answer = chr(65 + correct_answer_idx) if correct_answer_idx is not None else None
            is_correct = self._evaluate_mmlu_response(original_sample, extracted_answer)
        else:  # hellaswag
            correct_answer_idx = int(original_sample.get("label", -1))
            correct_answer = chr(65 + correct_answer_idx) if correct_answer_idx >= 0 else None
            is_correct = self._evaluate_hellaswag_response(original_sample, extracted_answer)
        
        return {
            "accuracy": 1.0 if is_correct else 0.0,
            "extracted_answer": extracted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
        }
    
    def _extract_answer(self, response: str) -> Optional[str]:
        """
        Extract the answer (A, B, C, or D) from the response.
        
        Args:
            response: The model's response
        
        Returns:
            The extracted answer or None if no answer was found
        """
        # Look for patterns like "Answer: A" or "The answer is B" or just "C."
        patterns = [
            r"(?:answer|option|select|choose|choice)(?:\s+is)?(?:\s*:)?\s*([A-Da-d])",  # Answer: A, The answer is B
            r"([A-Da-d])(?:\s*\.|$)",  # A., B
            r"^([A-Da-d])$",  # Just A, B, C, or D
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
            
        # Convert model's letter response to index (A->0, B->1, etc.)
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
            return False  # Invalid response
        
        # Check if the selected index matches the answer
        return selected_idx == example.get('answer')
    
    def _evaluate_hellaswag_response(self, example, response):
        """Evaluate if a response is correct for Hellaswag."""
        if response is None:
            return False
            
        # Convert model's letter response to index (A->0, B->1, etc.)
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
            return False  # Invalid response
        
        return selected_idx == int(example.get('label', -1))