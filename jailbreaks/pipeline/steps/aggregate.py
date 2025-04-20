import json
import logging
import re
import wandb
from collections import defaultdict

from jailbreaks.pipeline.pipeline import JailbreakPipeline

logger = logging.getLogger(__name__)

def aggregate(pipeline: JailbreakPipeline):
    logger.info("Step 4: Aggregating results from wandb")
    api = wandb.Api()
    
    # Initialize results structure
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
    # Fetch all runs from the project
    runs = api.runs(f"{pipeline.project_name}")
    logger.info(f"Found {len(runs)} runs in project {pipeline.project_name}")
    
    # Parse table pattern: "{benchmark}_{model}_{method}_{evaluator}_results"
    table_pattern = re.compile(r"([\w-]+)_([\w\.-]+)_([\w-]+)_([\w]+)_results")
    
    for run in runs:
        logger.info(f"Processing run: {run.name}")
        
        # Get all tables from this run
        for artifact in run.logged_artifacts():
            if artifact.type == "run_table":
                table_name = artifact.name
                match = table_pattern.match(table_name)
                
                if match:
                    benchmark, model, method, evaluator = match.groups()
                    logger.info(f"Found evaluation table: {table_name}")
                    
                    table = artifact.get("table")
                    data = table.data
                    
                    # Extract metrics from table columns
                    metrics = [col for col in data.columns if col != "Method"]
                    
                    # Store results
                    for metric in metrics:
                        for row in data.itertuples():
                            # Assuming first column is Method
                            method_name = getattr(row, "Method", method)
                            metric_value = getattr(row, metric, None)
                            if metric_value is not None:
                                all_results[benchmark][model][evaluator][method_name][metric] = metric_value
    
    # Update pipeline results summary
    pipeline.results_summary = all_results
    
    # Save summary to file
    summary_path = pipeline.output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        # Convert defaultdicts to regular dicts for JSON serialization
        summary_dict = json.loads(json.dumps(pipeline.results_summary))
        json.dump(summary_dict, f, indent=2)
    
    logger.info(f"Saved aggregated summary to {summary_path}")
    
    # Create comparison visualizations
    create_comparison_visualizations(pipeline, all_results)
    
    return pipeline


def create_comparison_visualizations(pipeline, all_results):
    """Create and log comparison visualizations for the aggregated results"""
    run_name = f"aggregated_results_{pipeline.run_id}"
    wandb.init(project=pipeline.project_name, name=run_name, id=run_name)
    
    # For each benchmark, model, evaluator combination
    for benchmark, models in all_results.items():
        for model, evaluators in models.items():
            for evaluator, methods in evaluators.items():
                # Get all metrics used by this evaluator
                all_metrics = set()
                for method_data in methods.values():
                    all_metrics.update(method_data.keys())
                
                # Create a table for each metric
                for metric in all_metrics:
                    table_data = wandb.Table(columns=["Method", metric])
                    
                    for method, metrics in methods.items():
                        if metric in metrics:
                            table_data.add_data(method, metrics[metric])
                    
                    # Log bar chart
                    wandb.log({
                        f"{benchmark}_{model}_{evaluator}_{metric}_chart": 
                        wandb.plot.bar(
                            table_data,
                            "Method",
                            metric,
                            title=f"{benchmark}-{model}-{evaluator}-{metric}"
                        )
                    })
    
    wandb.finish()


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