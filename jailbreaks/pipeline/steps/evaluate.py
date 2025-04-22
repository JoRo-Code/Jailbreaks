import os
import time
import json
import uuid
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List

import wandb
import pandas as pd

from jailbreaks.pipeline.schemas import (
    EvaluationResult,
    GeneratedResponse
)
from jailbreaks.pipeline.utils import (
    fetch_all_artifacts, 
    load_responses,
    FetchFilter
)
from jailbreaks.evaluators import ResponseEvaluator

logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    project_name: str
    responses_dir: Path
    evaluations_dir: Path
    evaluators: List[ResponseEvaluator]
    eval_run_id: str # optional
    use_local: bool = False
    upload_to_wandb: bool = True
    n_runs: int = None
    n_samples: int = None
    
def evaluate(evaluation_config: EvaluationConfig):
    if not evaluation_config.use_local:
        fetch_all_artifacts(
            project=evaluation_config.project_name,
            output_dir=evaluation_config.responses_dir, 
            fetch_filter=FetchFilter(
                #run_ids=[evaluation_config.eval_run_id],
                art_type="responses"
            )
        )

    generated_responses = load_responses(evaluation_config.responses_dir)
    
    eval_id = evaluation_config.eval_run_id
    if not eval_id:
        eval_id = str(uuid.uuid4())[:8]

    if evaluation_config.upload_to_wandb:
        run_name = f"evaluation_{eval_id}"
        wandb.init(project=evaluation_config.project_name, name=run_name, id=run_name)
        logger.info(f"Initialized: {run_name}")
        
    logger.info("Evaluating responses")

    _evaluate_responses_internal(evaluation_config, generated_responses)

    if evaluation_config.upload_to_wandb:
        wandb.finish()

def _evaluate_responses_internal(evaluation_config: EvaluationConfig, generated_responses: dict[str, dict[str, dict[str, list[GeneratedResponse]]]]):
    
    eval_start_time = time.time()
    
    if not evaluation_config.evaluators:
        logger.warning("No evaluators provided. Skipping evaluation step.")
        return
    
    evaluation_results = {}
    evaluation_times = {}
    
    for evaluator in evaluation_config.evaluators:
        evaluator_name = evaluator.__str__()
        logger.info(f"Evaluating with: {evaluator_name}")
        
        if evaluator_name not in evaluation_results:
            evaluation_results[evaluator_name] = {}
        
        if evaluator_name not in evaluation_times:
            evaluation_times[evaluator_name] = {}
        
        for benchmark_key, model_method_runs in generated_responses.items():
            if benchmark_key not in evaluation_results[evaluator_name]:
                evaluation_results[evaluator_name][benchmark_key] = {}
            
            if benchmark_key not in evaluation_times[evaluator_name]:
                evaluation_times[evaluator_name][benchmark_key] = {}
            
            logger.info(f"  Evaluating {benchmark_key} responses")
            
            for model_name, method_runs in model_method_runs.items():
                for method_combo, runs in method_runs.items():
                                
                    if model_name not in evaluation_results[evaluator_name][benchmark_key]:
                        evaluation_results[evaluator_name][benchmark_key][model_name] = {}
                    
                    evaluation_results[evaluator_name][benchmark_key][model_name][method_combo] = {}
                    
                    logger.info(f"    Evaluating {model_name} with {method_combo}")
                    
                    eval_start_time = time.time()
                    run_count = 0
                    for run_id, responses in runs.items():
                        if evaluation_config.n_samples and len(responses) > evaluation_config.n_samples:
                            responses = responses[:evaluation_config.n_samples]
                        if evaluation_config.n_runs and run_count >= evaluation_config.n_runs:
                            break
                        run_count += 1
                
                        # Evaluate responses
                        try:
                            metrics, sample_results = evaluator.evaluate(responses)
                            
                            # Create evaluation result
                            eval_result = EvaluationResult(
                                model_id=model_name,
                                method_config={"name": method_combo},
                                evaluator_name=evaluator_name,
                                metrics=metrics,
                                runtime_seconds=time.time() - eval_start_time
                            )
                            
                            # Add sample results
                            for result in sample_results:
                                eval_result.sample_results.append(result)
                            
                            evaluation_results[evaluator_name][benchmark_key][model_name][method_combo][run_id] = eval_result
                            
                            # Log metrics
                            log_dict = {
                                "benchmark": benchmark_key,
                                "model": model_name,
                                "method_combo": method_combo,
                                **{f"metrics/{k}": v for k, v in metrics.items()},
                                "evaluation_time": eval_result.runtime_seconds
                            }
                            if evaluation_config.upload_to_wandb:
                                wandb.log(log_dict)
                            
                            # Save results
                            cols = list(sample_results[0].keys())
                            if evaluation_config.upload_to_wandb:
                                results_table = wandb.Table(columns=cols)
                                for result in sample_results:
                                    results_table.add_data(*[result.get(col, "") for col in cols])
                                
                                wandb.log({f"{benchmark_key}_{model_name}_{method_combo}_{evaluator_name}_results_{run_id}": results_table})
                                
                            df = pd.DataFrame(
                                data=[[result.get(col, "") for col in cols] for result in sample_results],
                                columns=cols
                            )
                                
                            artifact_name = f"{benchmark_key}_{model_name}_{method_combo}_{evaluator_name}_{run_id}"
                            
                            # add file for download
                            path = f"{benchmark_key}/{model_name}/{method_combo}/{evaluator_name}/evaluation_{run_id}.csv"

                            csv_path = evaluation_config.evaluations_dir / path
                            os.makedirs(csv_path.parent, exist_ok=True)
                            df.to_csv(csv_path, index=False)
                            if evaluation_config.upload_to_wandb:
                                artifact = wandb.Artifact(name=artifact_name, type="evaluation_results")
                                artifact.add_file(csv_path, name=path)

                                wandb.log_artifact(artifact)
                            
                        except Exception as e:
                            logger.error(f"Error evaluating {model_name} with {method_combo}: {str(e)}")
                            import traceback
                            logger.error(traceback.format_exc())
                    
                # Track evaluation time
                eval_time = time.time() - eval_start_time
                if model_name not in evaluation_times[evaluator_name][benchmark_key]:
                    evaluation_times[evaluator_name][benchmark_key][model_name] = {}
                
                evaluation_times[evaluator_name][benchmark_key][model_name][method_combo] = {
                    'total_time': eval_time,
                    'avg_time_per_sample': eval_time / len(responses) if responses else 0,
                    'num_samples': len(responses)
                }
                
    # Save evaluation times
    os.makedirs(evaluation_config.evaluations_dir, exist_ok=True)
    eval_times_path = evaluation_config.evaluations_dir / "evaluation_times.json"
    with open(eval_times_path, 'w') as f:
        json.dump(evaluation_times, f, indent=2)
    
    logger.info(f"Saved evaluation times to {eval_times_path}")
    
    eval_time = time.time() - eval_start_time
    logger.info(f"Response evaluation completed in {eval_time:.2f}s")


