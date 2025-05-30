import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

@dataclass
class GeneratedResponse:
    """Structure to hold a generated response"""
    prompt: str
    raw_prompt: str  # The full prompt after jailbreak methods applied
    response: str
    model_id: str
    method_combo: str
    gen_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)
    
    
@dataclass
class EvaluationResult:
    """Structure to hold evaluation results"""
    model_id: str
    method_config: Dict[str, Any]
    evaluator_name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    sample_results: List[Dict[str, Any]] = field(default_factory=list)
    runtime_seconds: float = 0.0
    
    def add_sample_result(self, prompt, response, success=None, metadata=None):
        result = {
            "prompt": prompt,
            "response": response,
            "success": success,
        }
        if metadata:
            result.update(metadata)
        self.sample_results.append(result)
    
    def add_metric(self, name, value):
        self.metrics[name] = value
    
    def to_dict(self):
        return asdict(self)