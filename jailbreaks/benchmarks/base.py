from typing import List

class Benchmark:    
    def __init__(self, prompts: List[str] = None, max_new_tokens: int = 100, name: str = None):
        self.name = name or self.__class__.__name__ 
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens
    
    def get_prompts(self):
        return self.prompts
    
    def __str__(self):
        return self.name
