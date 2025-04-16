from typing import List

from jailbreaks.pipeline.pipeline import GeneratedResponse

class ResponseEvaluator:
    
    def __str__(self):
        return self.__class__.__name__
    
    def evaluate(self, responses: List[GeneratedResponse]):
        """
        Evaluate responses and return metrics and sample results
        
        Args:
            responses: List of GeneratedResponse objects
            
        Returns:
            tuple: (metrics_dict, sample_results_list)
        """
        raise NotImplementedError()
    
