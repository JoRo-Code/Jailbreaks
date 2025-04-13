from jailbreaks.pipeline.pipeline import JailbreakPipeline, EvaluationResult
from logging import getLogger
import time
import json
import wandb

logger = getLogger(__name__)

def evaluate(pipeline: JailbreakPipeline):
    pipeline.load_responses(pipeline.responses_dir)
    run_name = f"evaluation_{pipeline.run_id}"
    wandb.init(project=pipeline.project_name, name=run_name, id=pipeline.run_id)
    logger.info(f"W&B experiment initialized: {run_name}")
    logger.info("Step 3: Evaluating responses")
    _evaluate_responses_internal(pipeline)
    wandb.finish()

def _evaluate_responses_internal(pipeline: JailbreakPipeline):
    
    eval_start_time = time.time()
    
    if not pipeline.evaluators:
        logger.warning("No evaluators provided. Skipping evaluation step.")
        return
    
    if not hasattr(pipeline, 'evaluation_results'):
        pipeline.evaluation_results = {}
    
    # Initialize evaluation timing dictionary if not exists
    if not hasattr(pipeline, 'evaluation_times'):
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
                    results_table = wandb.Table(columns=["prompt", "response", "success", "score"])
                    for result in sample_results:
                        results_table.add_data(
                            result.get("prompt", ""),
                            result.get("response", ""),
                            result.get("success", False),
                            result.get("score", 0.0)
                        )
                    
                    # Log table
                    wandb.log({f"{benchmark_key}_{model_name}_{method_combo}_results": results_table})
                    
                    # Save evaluation results
                    pipeline._save_evaluation(eval_result, benchmark_key, model_name, method_combo, evaluator_name)
                    
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
