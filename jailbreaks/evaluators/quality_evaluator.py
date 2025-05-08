from typing import List

from jailbreaks.evaluators.base import ResponseEvaluator
from jailbreaks.pipeline.schemas import GeneratedResponse
from jailbreaks.evaluators.llm_judge.judge import (
    BaseLLMJudge, 
    LLMJudgeVote
)

class QualityEvaluator(ResponseEvaluator):
    def __init__(self, judge:BaseLLMJudge, name:str=None):
        super().__init__(name=name or "QualityEvaluator")
        self.judge = judge
    
    def evaluate(self, responses: List[GeneratedResponse]):
        votes: list[LLMJudgeVote] = self.judge.vote_batch(
            [resp.prompt for resp in responses],
            [resp.response for resp in responses],
        )
        
        sample_results = [
            {
                **vote.scores,
                **vote.metadata(),
            }
            for vote in votes
        ]
        
        metrics = {}
        vote_keys = votes[0].scores.keys()
        null_counts = {key: 0 for key in vote_keys}
        
        for vote in votes:
            for key in vote_keys:
                if hasattr(vote, key) and getattr(vote, key) is None:
                    null_counts[key] += 1
                if hasattr(vote, key) and getattr(vote, key) is not None:
                    metrics[key] = metrics.get(key, 0) + getattr(vote, key)
        
        for key in vote_keys:
            metrics[key] = metrics.get(key, 0)
        
        for key in vote_keys:
            metrics[f'{key}_avg'] = metrics.get(key, 0) / len(responses)
        
        for key in null_counts:
            metrics[f'{key}_null'] = null_counts[key]
        metrics["total_evaluated"] = len(responses)
        
        return metrics, sample_results
