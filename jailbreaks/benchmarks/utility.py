import os
import time
import logging
from pydantic import BaseModel

import mlflow
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks.schema import MultipleChoiceSchema
from deepeval.benchmarks import MMLU, HellaSwag
from deepeval.benchmarks.tasks import MMLUTask, HellaSwagTask
from transformers import AutoTokenizer

from jailbreaks.benchmarks.base import Benchmark
from jailbreaks.llm import LLM

logger = logging.getLogger("UtilityBenchmark")

class UtilityBenchmark(Benchmark):
    def __init__(self, n_samples: int = 1):
        super().__init__(f"Utility")

        
        mmlu = MMLU(
            tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE],
            n_shots=0,
            n_problems_per_task=n_samples
        )
        hellaswag = HellaSwag(
            tasks=[HellaSwagTask.DRINKING_BEER],
            n_shots=0,
            n_problems_per_task=n_samples
        )
        
        self.benchmarks = [mmlu, hellaswag]
    
    def log_results(self, benchmark: Benchmark):
        # Get current MLflow run ID
        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else f"run_{int(time.time())}"
        results_dir = os.path.join("results", run_id)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save predictions with unique filename
        predictions_df = benchmark.predictions
        predictions_path = os.path.join(results_dir, f"{benchmark.__class__.__name__}.csv")
        predictions_df.to_csv(predictions_path, index=False)
        
        # Save scores with unique filename
        scores_df = benchmark.task_scores
        scores_path = os.path.join(results_dir, f"{benchmark.__class__.__name__}_scores.csv")
        scores_df.to_csv(scores_path, index=False)
        
        # Log to current MLflow run
        mlflow.log_artifact(predictions_path)
        mlflow.log_artifact(scores_path)
        
        # Log metrics to current MLflow run
        for _, row in scores_df.iterrows():
            task_name = row.get('task', 'unknown_task')
            score = row.get('score', 0)
            mlflow.log_metric(f"{benchmark.__class__.__name__}_{task_name}_score", score)
        
        # Log overall average score
        if not scores_df.empty and 'score' in scores_df.columns:
            avg_score = scores_df['score'].mean()
            mlflow.log_metric(f"{benchmark.__class__.__name__}_average_score", avg_score)
            logger.info(f"{benchmark.__class__.__name__} average score: {avg_score:.4f}")
        
        logger.info(f"Saved results to {results_dir} and logged to MLflow run {run_id}")
        
    def evaluate(self, model: LLM):
        wrapped_model = UtilityBenchmarkModel(model, model.tokenizer, model.name, model.device)
        
        for benchmark in self.benchmarks:
            logger.info(f"Running {benchmark.__class__.__name__} benchmark")
            benchmark.evaluate(model=wrapped_model)  # TODO: batch
            
            self.log_results(benchmark)
            logger.info(f"Completed {benchmark.__class__.__name__} benchmark - {benchmark.task_scores}")
            
            
import json
class UtilityBenchmarkModel(DeepEvalBaseLLM):
    def __init__(self, model:LLM, tokenizer:AutoTokenizer, name:str, device:str="cuda"):
        self.model, self.tokenizer = model, tokenizer
        self.name, self.device = name, device
        
        self.model.to(self.device)
        super().__init__()

    def extract_answer(self, response: str, schema: BaseModel) -> BaseModel|str:
        import re
        import random
        answer = None
        match = re.search(r'ANSWER:\s*([A-D])', response, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
        
        match = re.search(r'^([A-D])\.?\s', response, re.MULTILINE | re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
        
        #single letter answer (A, B, C, or D)
        match = re.search(r'\b([A-D])\b', response, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
        
        # hack when no answer is found
        if answer is None:
            logger.debug("No answer found, returning random choice")
            answer = random.choice(["A", "B", "C", "D"])
        
        logger.debug(f"Extracted answer: {answer}")
        
        return MultipleChoiceSchema(answer=answer.upper())
         
        
    def generate(self, prompt: str, schema: BaseModel|None=None, **kwargs) -> str:
        prompt = prompt.replace("Answer:", "Short answer. Only the first sentence will be checked. E.g. 'ANSWER:A'. ")
        prompt = prompt.replace("Output 'A', 'B', 'C', or 'D'. Full answer not needed.","")
        prompt = [{"role": "user", "content": prompt}]
        prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        response = self.model.generate(prompt, decode=True, max_new_tokens=10)
        
        if schema:
            response = self.extract_answer(response, schema)
        
        logger.debug(f"UtilityBenchmarkModel - final response: {response}")
        
        return response

    # jank required by DeepEval
    async def a_generate(self, prompt: str) -> str: return self.generate(prompt)
    def get_model_name(self): return self.name
    def load_model(self): return self.model
