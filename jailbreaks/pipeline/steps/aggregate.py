import json
import logging
import wandb

from jailbreaks.pipeline.pipeline import JailbreakPipeline

logger = logging.getLogger(__name__)

def aggregate(pipeline: JailbreakPipeline):
    logger.info("Step 4: Aggregating results")
    pipeline.load_responses(pipeline.responses_dir)
    pipeline.load_evaluation_results()
    if not hasattr(pipeline, 'evaluation_results'):
        logger.warning("No evaluation results found. Skipping comparison tables.")
        return
    
    for evaluator_name, benchmarks in pipeline.evaluation_results.items():
        for benchmark_key, models in benchmarks.items():
            for model_name, method_results in models.items():
                pipeline = create_evaluation_table(pipeline, model_name, benchmark_key, evaluator_name, method_results)
    
    summary_path = pipeline.output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(pipeline.results_summary, f, indent=2)
    
    logger.info(f"Saved summary to {summary_path}")
    
    return pipeline


def create_evaluation_table(pipeline: JailbreakPipeline, model_name, benchmark_key, evaluator_name, method_results):
    run_name = f"{model_name}_{pipeline.run_id}"
    wandb.init(project=pipeline.project_name, name=run_name, id=run_name)
    logger.info(f"W&B experiment initialized: {run_name}")

    # Create method comparison table
    comparison_data = []
    method_names = []
    metric_names = set()
    
    # Collect all metric names
    for method_name, eval_result in method_results.items():
        method_names.append(method_name)
        for metric_name in eval_result.metrics.keys():
            metric_names.add(metric_name)
    

    for benchmark_data in pipeline.generated_responses.values():
        for key, responses in benchmark_data.items():
            parts = key.split('_')
            method_combo = parts[-1]
            model_short_name = '_'.join(parts[:-1])
            
            if model_short_name == model_name and method_combo in method_results:
                # Calculate average generation time
                if responses and "gen_time" in responses[0].metadata:
                    avg_gen_time = sum(r.metadata.get("gen_time", 0) for r in responses) / len(responses)
                    method_results[method_combo].metrics["avg_gen_time"] = avg_gen_time
                    metric_names.add("avg_gen_time")
    
    # Add rows
    for method_name, eval_result in method_results.items():
        # Also store as a dictionary for easier access
        method_data = {"Method": method_name}
        for metric_name in metric_names:
            method_data[metric_name] = eval_result.metrics.get(metric_name, None)
        comparison_data.append(method_data)
    
    # Log comparison table
    table_name = f"{benchmark_key}_{model_name}_{evaluator_name}_comparison"
    
    # Store summary data
    if benchmark_key not in pipeline.results_summary:
        pipeline.results_summary[benchmark_key] = {}
    if model_name not in pipeline.results_summary[benchmark_key]:
        pipeline.results_summary[benchmark_key][model_name] = {}
    
    pipeline.results_summary[benchmark_key][model_name][evaluator_name] = comparison_data
    
    # Also log as a bar chart for key metrics
    for metric_name in metric_names:
        # Create a data table for the bar chart
        bar_data = wandb.Table(columns=["Method", metric_name])
        for method, results in method_results.items():
            bar_data.add_data(method, results.metrics.get(metric_name, 0))
        
        # Log bar chart with correct parameters
        wandb.log({
            f"{benchmark_key}_{evaluator_name}_{metric_name}_chart": 
            wandb.plot.bar(
                bar_data,
                "Method",
                metric_name,
                title=f"{benchmark_key}-{evaluator_name}-{metric_name}"
            )
        })    

    wandb.finish()
    return pipeline