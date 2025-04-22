from typing import List

from jailbreaks.pipeline.schemas import GeneratedResponse

class ResponseEvaluator:
    
    def __init__(self, name:str):
        self.name = name
    
    def __str__(self):
        return self.name
    
    def evaluate(self, responses: List[GeneratedResponse]):
        """
        Evaluate responses and return metrics and sample results
        
        Args:
            responses: List of GeneratedResponse objects
            
        Returns:
            tuple: (metrics_dict, sample_results_list)
        """
        raise NotImplementedError()
    
