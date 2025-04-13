import time
from typing import List, Dict, Any, Optional
import wandb
from dataclasses import dataclass, field, asdict
import json
import uuid
from pathlib import Path
from tqdm import tqdm
import torch
import pandas as pd
import os


# Methods
from jailbreaks.methods.base_method import JailBreakMethod

import logging
logger = logging.getLogger(__name__)


@dataclass
class MethodConfig:
    """Base configuration for jailbreak methods"""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    fitted: bool = False
    
    def to_dict(self):
        return asdict(self)


@dataclass
class GeneratedResponse:
    """Structure to hold a generated response"""
    prompt: str
    raw_prompt: str  # The full prompt after jailbreak methods applied
    response: str
    model_id: str
    method_combo: str
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


class JailbreakPipeline:    
    def __init__(
        self, 
        project_name: str,
        model_paths: List[str] = None,
        method_combinations: List[List[JailBreakMethod]] = None,
        benchmarks: List = None,
        evaluators: List = None,
        device: str = None,
        output_dir: str = "results",
        run_id: Optional[str] = None,
        batch_size: int = 8
    ):
        self.model_paths = model_paths or []
        self.method_combinations = method_combinations or [[]]
        self.benchmarks = benchmarks or []
        self.evaluators = evaluators or []
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        
        self.fitted_methods = {}  
        self.generated_responses = {}  
        self.run_id = run_id or str(uuid.uuid4())[:8]  
        self.results_summary = {}
        self.project_name = project_name
        self.output_dir = Path(output_dir + f"/{self.run_id}")
        self.setup_output_dir()
        
        # Initialize dictionaries to track step times
        self.step_times = {}
        self.method_model_times = {}
    
    def setup_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir = self.output_dir / "responses"
        self.responses_dir.mkdir(exist_ok=True)
        self.evaluations_dir = self.output_dir / "evaluations"
        self.evaluations_dir.mkdir(exist_ok=True)
    
    
    def clear_output_dir(self):
        if os.path.exists(self.output_dir):
            import shutil
            shutil.rmtree(self.output_dir)
    
    def reset_output_dir(self):
        self.clear_output_dir()
        self.setup_output_dir()
    
    def _get_method_config(self, method):
        """Extract configuration from a method instance"""
        if hasattr(method, 'config'):
            config = method.config
            return MethodConfig(
                name=method.__str__(), 
                params=asdict(config) if hasattr(config, '__dict__') else config,
                fitted=method.__str__().lower() in self.fitted_methods
            )
        elif hasattr(method, 'params'):
            return MethodConfig(
                name=method.__str__(), 
                params=method.params,
                fitted=method.__str__().lower() in self.fitted_methods
            )
        else:
            return MethodConfig(
                name=method.__str__(),
                fitted=method.__str__().lower() in self.fitted_methods
            )
    
    def _get_evaluator_config(self, evaluator):
        """Extract configuration from an evaluator"""
        if hasattr(evaluator, 'config'):
            return evaluator.config
        elif hasattr(evaluator, 'params'):
            return evaluator.params
        else:
            return {"name": evaluator.__str__()}

    
    def save_evaluation(self, eval_result, benchmark_key, model_name, method_combo, evaluator_name):
        """Save evaluation results to a file"""
        output_path = self.evaluations_dir / f"eval_{benchmark_key}_{model_name}_{method_combo}_{evaluator_name}.json"
        
        with open(output_path, 'w') as f:
            json.dump(eval_result.to_dict(), f, indent=2)
        
        logger.info(f"Saved evaluation results to {output_path}")
    

    def load_responses(self, path):
        """Load pre-generated responses from a path"""
        logger.info(f"Loading responses from {path}")
        path = Path(path)
        
        if path.is_dir():
            # Load all response files from directory
            response_files = list(path.glob("responses_*.json"))
            for file in response_files:
                self._load_response_file(file)
        else:
            # Load single response file
            self._load_response_file(path)
            
        logger.info(f"Loaded {sum(len(responses) for benchmark in self.generated_responses.values() for responses in benchmark.values())} responses")
    
    def _load_response_file(self, file_path):
        """Load a single response file"""
        try:
            with open(file_path, 'r') as f:
                responses_data = json.load(f)
            
            # Parse filename to get benchmark, model, and method
            filename = file_path.stem
            parts = filename.split('_')
            if len(parts) < 4 or parts[0] != "responses":
                logger.warning(f"Unexpected filename format: {filename}, skipping")
                return
            
            benchmark_key = parts[1]
            model_name = parts[2]
            method_combo = '_'.join(parts[3:])
            
            # Convert loaded data to GeneratedResponse objects
            responses = [GeneratedResponse(**data) for data in responses_data]
            
            # Add to generated_responses
            if benchmark_key not in self.generated_responses:
                self.generated_responses[benchmark_key] = {}
            
            key = f"{model_name}_{method_combo}"
            self.generated_responses[benchmark_key][key] = responses
            logger.info(f"Loaded {len(responses)} responses for {benchmark_key}/{model_name}/{method_combo}")
        
        except Exception as e:
            logger.error(f"Error loading response file {file_path}: {str(e)}")
    
    def clear_evaluation_results(self):
        if self.evaluations_dir.exists():
            import shutil
            shutil.rmtree(self.evaluations_dir)
        self.evaluations_dir.mkdir(exist_ok=True)
        logger.info(f"Cleared evaluation results directory: {self.evaluations_dir}")

    def load_evaluation_results(self):
        """Load evaluation results from saved files"""
        logger.info("Loading evaluation results from disk")
        
        self.evaluation_results = {}
        
        eval_files = list(self.evaluations_dir.glob("eval_*.json"))
        if not eval_files:
            logger.warning(f"No evaluation files found in {self.evaluations_dir}")
            return
        
        for file_path in eval_files:
            try:
                # Parse filename to get benchmark, model, method, and evaluator
                filename = file_path.stem
                parts = filename.split('_')
                if len(parts) < 5 or parts[0] != "eval":
                    logger.warning(f"Unexpected filename format: {filename}, skipping")
                    continue
                
                benchmark_key = parts[1]
                model_name = parts[2]
                method_combo = '_'.join(parts[3:-1])  # Join all parts between model and evaluator
                evaluator_name = parts[-1]
                
                # Load evaluation result
                with open(file_path, 'r') as f:
                    eval_data = json.load(f)
                
                # Add to evaluation_results
                if evaluator_name not in self.evaluation_results:
                    self.evaluation_results[evaluator_name] = {}
                
                if benchmark_key not in self.evaluation_results[evaluator_name]:
                    self.evaluation_results[evaluator_name][benchmark_key] = {}
                
                if model_name not in self.evaluation_results[evaluator_name][benchmark_key]:
                    self.evaluation_results[evaluator_name][benchmark_key][model_name] = {}
                
                self.evaluation_results[evaluator_name][benchmark_key][model_name][method_combo] = EvaluationResult(**eval_data)
                logger.info(f"Loaded evaluation results for {benchmark_key}/{model_name}/{method_combo}/{evaluator_name}")
            
            except Exception as e:
                logger.error(f"Error loading evaluation file {file_path}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"Loaded evaluation results for {len(self.evaluation_results)} evaluators")
