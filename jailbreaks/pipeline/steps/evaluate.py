import time
import json
import logging

import pandas as pd
import wandb

from jailbreaks.pipeline.pipeline import (
    JailbreakPipeline, 
    EvaluationResult,
    GeneratedResponse
)
from jailbreaks.pipeline.utils import fetch_all_artifacts

logger = logging.getLogger(__name__)

def evaluate(pipeline: JailbreakPipeline):
    fetch_all_artifacts(pipeline.project_name, pipeline.responses_dir, "responses")
    
    run_name = f"evaluation_{pipeline.run_id}"
    wandb.init(project=pipeline.project_name, name=run_name, id=run_name)
    logger.info(f"W&B experiment initialized: {run_name}")
    logger.info("Step 3: Evaluating responses")
    
    evaluate_from_responses_dir(pipeline)
    
    wandb.finish()

def _evaluate_responses_internal(pipeline: JailbreakPipeline):
    
    eval_start_time = time.time()
    
    if not pipeline.evaluators:
        logger.warning("No evaluators provided. Skipping evaluation step.")
        return
    
    pipeline.clear_evaluation_results()
    pipeline.evaluation_results = {}
    pipeline.evaluation_times = {}
    
    for evaluator in pipeline.evaluators:
        evaluator_name = evaluator.__str__()
        logger.info(f"Evaluating with: {evaluator_name}")
        
        if evaluator_name not in pipeline.evaluation_results:
            pipeline.evaluation_results[evaluator_name] = {}
        
        if evaluator_name not in pipeline.evaluation_times:
            pipeline.evaluation_times[evaluator_name] = {}
        
        for benchmark_key, model_method_responses in pipeline.generated_responses.items():
            if benchmark_key not in pipeline.evaluation_results[evaluator_name]:
                pipeline.evaluation_results[evaluator_name][benchmark_key] = {}
            
            if benchmark_key not in pipeline.evaluation_times[evaluator_name]:
                pipeline.evaluation_times[evaluator_name][benchmark_key] = {}
            
            logger.info(f"  Evaluating {benchmark_key} responses")
            
            for model_method_key, responses in model_method_responses.items():
                parts = model_method_key.split('_')
                method_combo = parts[-1]
                model_name = '_'.join(parts[:-1])
                
                logger.info(f"    Evaluating {model_name} with {method_combo}")
                
                eval_start_time = time.time()
                
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
                    
                    # Store for summary tables
                    if model_name not in pipeline.evaluation_results[evaluator_name][benchmark_key]:
                        pipeline.evaluation_results[evaluator_name][benchmark_key][model_name] = {}
                    
                    pipeline.evaluation_results[evaluator_name][benchmark_key][model_name][method_combo] = eval_result
                    
                    # Log metrics
                    log_dict = {
                        "benchmark": benchmark_key,
                        "model": model_name,
                        "method_combo": method_combo,
                        **{f"metrics/{k}": v for k, v in metrics.items()},
                        "evaluation_time": eval_result.runtime_seconds
                    }
                    wandb.log(log_dict)
                    
                    # Create results table
                    cols = list(sample_results[0].keys())
                    results_table = wandb.Table(columns=cols)
                    for result in sample_results:
                        results_table.add_data(*[result.get(col, "") for col in cols])
                    
                    # Log table
                    wandb.log({f"{benchmark_key}_{model_name}_{method_combo}_{evaluator_name}_results_{pipeline.run_id}": results_table})
                    
                    # Save evaluation results
                    pipeline.save_evaluation(eval_result, benchmark_key, model_name, method_combo, evaluator_name)
                    
                    # Track evaluation time
                    eval_time = time.time() - eval_start_time
                    if model_name not in pipeline.evaluation_times[evaluator_name][benchmark_key]:
                        pipeline.evaluation_times[evaluator_name][benchmark_key][model_name] = {}
                    
                    pipeline.evaluation_times[evaluator_name][benchmark_key][model_name][method_combo] = {
                        'total_time': eval_time,
                        'avg_time_per_sample': eval_time / len(responses) if responses else 0,
                        'num_samples': len(responses)
                    }
                    
                except Exception as e:
                    logger.error(f"Error evaluating {model_method_key}: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
    # Save evaluation times
    eval_times_path = pipeline.evaluations_dir / "evaluation_times.json"
    with open(eval_times_path, 'w') as f:
        json.dump(pipeline.evaluation_times, f, indent=2)
    
    logger.info(f"Saved evaluation times to {eval_times_path}")
    
    eval_time = time.time() - eval_start_time
    pipeline.step_times['_evaluate_responses_internal'] = eval_time
    logger.info(f"Response evaluation completed in {eval_time:.2f}s")

def evaluate_from_responses_dir(pipeline: JailbreakPipeline):
    """
    Scans the responses directory structure and evaluates all CSV files found.
    Directory structure expected: benchmark_key/model_name/method_combo/responses_run_id.csv
    """
    
    eval_start_time = time.time()
    
    if not pipeline.evaluators:
        logger.warning("No evaluators provided. Skipping evaluation step.")
        return
    
    pipeline.clear_evaluation_results()
    pipeline.evaluation_results = {}
    pipeline.evaluation_times = {}
    
    # Set up wandb logging
    run_name = f"evaluations_{pipeline.run_id}"
    wandb.init(project=pipeline.project_name, name=run_name, id=run_name)
    logger.info(f"W&B experiment initialized: {run_name}")
    
    responses_dir = pipeline.responses_dir
    
    # Walk through directory structure to find CSV files
    for benchmark_path in responses_dir.glob("*"):
        if not benchmark_path.is_dir():
            continue
        
        benchmark_key = benchmark_path.name
        logger.info(f"Processing benchmark: {benchmark_key}")
        
        if benchmark_key not in pipeline.generated_responses:
            pipeline.generated_responses[benchmark_key] = {}
        
        for model_path in benchmark_path.glob("*"):
            if not model_path.is_dir():
                continue
            
            model_name = model_path.name
            logger.info(f"  Processing model: {model_name}")
            
            for combo_path in model_path.glob("*"):
                if not combo_path.is_dir():
                    continue
                
                method_combo = combo_path.name
                logger.info(f"    Processing method combo: {method_combo}")
                
                # Look for response CSV files
                for csv_file in combo_path.glob("*.csv"):
                    logger.info(f"      Loading responses from: {csv_file}")
                    
                    try:
                        # Load responses from CSV
                        response_df = pd.read_csv(csv_file)
                        
                        # Convert to GeneratedResponse objects
                        responses = []
                        for _, row in response_df.iterrows():
                            gen_response = GeneratedResponse(
                                prompt=row.get('prompt', ''),
                                raw_prompt=row.get('raw_prompt', row.get('prompt', '')),
                                response=row.get('response', ''),
                                model_id=model_name,
                                method_combo=method_combo,
                                metadata={
                                    "benchmark": benchmark_key,
                                    "timestamp": time.time(),
                                }
                            )
                            responses.append(gen_response)
                        
                        # Store responses for evaluation
                        model_method_key = f"{model_name}_{method_combo}"
                        pipeline.generated_responses[benchmark_key][model_method_key] = responses
                        
                    except Exception as e:
                        logger.error(f"Error loading responses from {csv_file}: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
    
    # Now evaluate all loaded responses
    _evaluate_responses_internal(pipeline)
    
    total_eval_time = time.time() - eval_start_time
    logger.info(f"Response evaluation from directory completed in {total_eval_time:.2f}s")
    
    wandb.finish()
