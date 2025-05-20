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
        name: Optional[str] = None,
    ):

        
        super().__init__(name=name or f"accuracy")
    
    def evaluate(self, responses: List[GeneratedResponse]) -> List[Dict[str, Any]]:
        samples = [self.evaluate_single(resp) for resp in responses]
        metrics = {
            "accuracy": sum(sample["is_correct"] for sample in samples) / len(samples),
        }
        return metrics, samples
    
    def idx_to_char(self, idx: int) -> str:
        return chr(65 + idx)
    
    def evaluate_single(self, response: GeneratedResponse) -> Dict[str, Any]:
        
        correct_answer_idx = response.metadata.get("answer")
        correct_answer = self.idx_to_char(correct_answer_idx)
        extracted_answer = self._extract_answer(response.response)
        
        is_correct = (extracted_answer is not None and correct_answer is not None and str(extracted_answer).upper() == str(correct_answer).upper())
        
        return {
            "prompt": response.prompt,
            "response": response.response,
            "extracted_answer": extracted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
        }
    
    def _extract_answer(self, response: str) -> Optional[str]:
        response = str(response)
        patterns = [
            r"(?:answer|option|select|choose|choice)(?:\s+is)?(?:\s*:)?\s*\b([A-Da-d])\b",
            r"\b([A-Da-d])\b(?:\s*\.?\s*$)",
            r"^\s*([A-Da-d])\s*$",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return None
    