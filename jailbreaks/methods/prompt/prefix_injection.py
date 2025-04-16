from jailbreaks.methods.base_method import PromptInjection
from jailbreaks.methods.utils import format_name

class PrefixInjection(PromptInjection):
    def __init__(self, suffix: str = "", prefix: str = "", description: str = ""):
        super().__init__()
        self.suffix = suffix
        self.prefix = prefix
        self.name = format_name(self.__class__.__name__, description)

    def preprocess(self, prompt: str, model_name: str) -> str:
        return f"{self.prefix}{prompt}{self.suffix}"
