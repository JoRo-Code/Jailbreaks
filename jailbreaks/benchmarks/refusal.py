from typing import List

from jailbreaks.benchmarks.base import Benchmark
from jailbreaks.llm import LLM

from jailbreaks.utils.refusal import is_refusal

import mlflow
import time
from tqdm import tqdm

import logging
logger = logging.getLogger("RefusalBenchmark")

class RefusalBenchmark(Benchmark):    
    def __init__(self, prompts: List[str] = None, max_new_tokens: int = 100):
        super().__init__(f"Refusal")
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens
    
    def evaluate(self, model: LLM):
        logger.info(f"Running {self.name} benchmark with {len(self.prompts)} samples")
        
        results = []
        refusal_count = 0
        total_quality = 0
        
        for i, prompt in enumerate(tqdm(self.prompts, desc="Evaluating prompts")):
            output_text, input_prompt = model.generate(prompt, return_input=True, only_new_tokens=True, decode=True, max_new_tokens=self.max_new_tokens)          
            logger.info(f"output: {output_text}\n\n")
            refused = self._is_refusal(output_text)
            if refused:
                refusal_count += 1
                
            # reponse quality
            quality_score = self._score_quality(prompt, output_text)
            total_quality += quality_score
            
            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else f"run_{int(time.time())}"
            mlflow.log_dict(
                {
                    "input": prompt,
                    "input_prompt": input_prompt,
                    "output": output_text,
                    "refused": refused,
                    "quality_score": quality_score
                },
                f"{self.name}_{run_id}/{i}.json"
            )
            
            results.append({
                "input": prompt,
                "output": output_text,
                "refused": refused,
                "quality_score": quality_score
            })
            
        # aggregate metrics
        refusal_rate = refusal_count / len(self.prompts) if self.prompts else 0
        avg_quality = total_quality / (len(self.prompts) - refusal_count) if (len(self.prompts) - refusal_count) > 0 else 0
        
        mlflow.log_metric(f"{self.name}_refusal_rate", refusal_rate)
        mlflow.log_metric(f"{self.name}_avg_quality", avg_quality)
        
        logger.info(f"{self.name} refusal rate: {refusal_rate:.4f}")
        logger.info(f"{self.name} avg quality score: {avg_quality:.4f}")
        
        return results
    
    def _is_refusal(self, output):
        return is_refusal(output)
    
    def _score_quality(self, prompt, output):
        # TODO: use LLM evaluator
        keywords = prompt.lower().split()
        keywords = [k for k in keywords if len(k) > 3]
        
        if not keywords:
            return 0
            
        output_lower = output.lower()
        matches = sum(1 for k in keywords if k in output_lower)
        return min(1.0, matches / len(keywords))
