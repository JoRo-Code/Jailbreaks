from jailbreaks.models.llm import LLM

class Benchmark:
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
    
    def evaluate(self, model: LLM):
        raise NotImplementedError
    
    def __str__(self):
        return self.name
