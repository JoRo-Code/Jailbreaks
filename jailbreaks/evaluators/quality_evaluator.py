
from jailbreaks.pipeline.pipeline import GeneratedResponse
from jailbreaks.evaluators.base import ResponseEvaluator
from typing import List
import numpy as np

class QualityEvaluator(ResponseEvaluator):
    def __init__(self):
        pass
    
    def evaluate(self, responses: List[GeneratedResponse]):
        metrics = {}
        sample_results = []
        
        total = 0
        for resp in responses:
            score = np.random.rand()
            total += score
            sample_results.append({
                "prompt": resp.prompt,
                "response": resp.response,
                "refusal": np.random.rand(),
                "score": score
            })
        
        metrics["score"] = total / len(responses)
        metrics["total_evaluated"] = len(responses)
        
        return metrics, sample_results
        