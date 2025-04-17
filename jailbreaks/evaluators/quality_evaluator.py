from typing import List

from jailbreaks.evaluators.base import ResponseEvaluator
from jailbreaks.pipeline.pipeline import GeneratedResponse
from jailbreaks.evaluators.llm_judge.judge import LLMJudge

class QualityEvaluator(ResponseEvaluator):
    def __init__(self, n_votes:int=1):
        self.judge = LLMJudge()
        self.n_votes = n_votes
    
    def evaluate(self, responses: List[GeneratedResponse]):
        metrics = {}
        sample_results = []
        
        total = {}
        for resp in responses:
            score, _, _ = self.judge.score(resp.prompt, resp.response, n_votes=self.n_votes)
            for key, value in score.items():
                total[key] = total.get(key, 0) + value
            sample_results.append({
                "prompt": resp.prompt,
                "response": resp.response,
                **score
            })
        
        for key, value in total.items():
            metrics[key] = value / len(responses)
        
        metrics["total_evaluated"] = len(responses)
        
        return metrics, sample_results
