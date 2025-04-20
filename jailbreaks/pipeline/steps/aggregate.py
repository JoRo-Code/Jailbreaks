import os
import logging
from pathlib import Path
from dataclasses import dataclass

import wandb
import pandas as pd
import numpy as np

from jailbreaks.pipeline.utils import (
    fetch_all_artifacts, 
    load_evaluations
)

logger = logging.getLogger(__name__)

@dataclass
class AggregateConfig:
    project_name: str
    evaluations_dir: Path
    output_dir: Path
    eval_run_id: str # optional
    use_local: bool = False

def aggregate(config: AggregateConfig):
    logger.info("Step 4: Aggregating results from")
    
    config.eval_run_id = f"evaluation_{config.eval_run_id}" # match
    
    if not config.use_local:
        fetch_all_artifacts(
            project=config.project_name, 
            output_dir=config.evaluations_dir,
            art_type="evaluation_results",
            run_ids=[config.eval_run_id]
        )
    
    eval_results = load_evaluations(config.evaluations_dir)
    
    run_name = f"aggregate_{config.eval_run_id}"
    wandb.init(project=config.project_name, name=run_name, id=run_name)
    logger.info(f"Initialized: {run_name}")
    logger.info("Aggregating results")
    
    _aggregate_results(eval_results, config)

def _aggregate_results(eval_results: dict, config: AggregateConfig):

    # result[benchmark_key][model_method_key][evaluator_name][file_path.stem] = parsed_data
    
    all_data = []
    for benchmark_key, model_name_d in eval_results.items():
        for model_name, method_combo_d in model_name_d.items():
            for method_combo, evaluator_name_d in method_combo_d.items():
                
                for evaluator_name, run_id_d in evaluator_name_d.items():
                    
                    # Initialize row data with method name
                    row_data = {'method': method_combo,
                                'model': model_name,
                                'benchmark': benchmark_key,
                                'evaluator': evaluator_name}
                    
                    # Dictionary to collect metric values
                    metrics = {}
                    
                    # Collect all metric values
                    for result_df in run_id_d.values():
                        for col in result_df.columns:
                            try:
                                numeric_values = pd.to_numeric(result_df[col])
                                if col not in metrics:
                                    metrics[col] = []
                                metrics[col].append(numeric_values.mean())
                            except (ValueError, TypeError):
                                continue
                    

                    for metric_name, values in metrics.items():
                        # Store the values as a list
                        values = np.array(values)
                        row_data[f'values_{metric_name}'] = values
                        # Calculate and store the average
                        row_data[f'avg_{metric_name}'] = values.mean()
                        # Calculate and store the standard deviation
                        if len(values) > 1:
                            row_data[f'std_{metric_name}'] = values.std()
                        else:
                            row_data[f'std_{metric_name}'] = np.nan
                            
                    all_data.append(row_data)
                
        # Create DataFrame and save to CSV
        df = pd.DataFrame(all_data)
        # Save with evaluator name and a unique ID
        output_path = config.output_dir / f"{config.eval_run_id}.csv"
        os.makedirs(config.output_dir, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        # Log artifact to wandb
        artifact = wandb.Artifact(f"aggregated_results_{evaluator_name}", type="aggregated_results")
        artifact.add_file(str(output_path))
        wandb.log_artifact(artifact)

# def create_comparison_visualizations(pipeline, all_results):
#     """Create and log comparison visualizations for the aggregated results"""
#     run_name = f"aggregated_results_{pipeline.run_id}"
#     wandb.init(project=pipeline.project_name, name=run_name, id=run_name)
    
#     # For each benchmark, model, evaluator combination
#     for benchmark, models in all_results.items():
#         for model, evaluators in models.items():
#             for evaluator, methods in evaluators.items():
#                 # Get all metrics used by this evaluator
#                 all_metrics = set()
#                 for method_data in methods.values():
#                     all_metrics.update(method_data.keys())
                
#                 # Create a table for each metric
#                 for metric in all_metrics:
#                     table_data = wandb.Table(columns=["Method", metric])
                    
#                     for method, metrics in methods.items():
#                         if metric in metrics:
#                             table_data.add_data(method, metrics[metric])
                    
#                     # Log bar chart
#                     wandb.log({
#                         f"{benchmark}_{model}_{evaluator}_{metric}_chart": 
#                         wandb.plot.bar(
#                             table_data,
#                             "Method",
#                             metric,
#                             title=f"{benchmark}-{model}-{evaluator}-{metric}"
#                         )
#                     })
    
#     wandb.finish()


# def create_evaluation_table(pipeline: JailbreakPipeline, model_name, benchmark_key, evaluator_name, method_results):
#     run_name = f"{model_name}_{pipeline.run_id}"
#     wandb.init(project=pipeline.project_name, name=run_name, id=run_name)
#     logger.info(f"W&B experiment initialized: {run_name}")

#     # Create method comparison table
#     comparison_data = []
#     method_names = []
#     metric_names = set()
    
#     # Collect all metric names
#     for method_name, eval_result in method_results.items():
#         method_names.append(method_name)
#         for metric_name in eval_result.metrics.keys():
#             metric_names.add(metric_name)
    

#     for benchmark_data in pipeline.generated_responses.values():
#         for key, responses in benchmark_data.items():
#             parts = key.split('_')
#             method_combo = parts[-1]
#             model_short_name = '_'.join(parts[:-1])
            
#             if model_short_name == model_name and method_combo in method_results:
#                 # Calculate average generation time
#                 if responses and "gen_time" in responses[0].metadata:
#                     avg_gen_time = sum(r.metadata.get("gen_time", 0) for r in responses) / len(responses)
#                     method_results[method_combo].metrics["avg_gen_time"] = avg_gen_time
#                     metric_names.add("avg_gen_time")
    
#     # Add rows
#     for method_name, eval_result in method_results.items():
#         # Also store as a dictionary for easier access
#         method_data = {"Method": method_name}
#         for metric_name in metric_names:
#             method_data[metric_name] = eval_result.metrics.get(metric_name, None)
#         comparison_data.append(method_data)
    
#     # Log comparison table
#     table_name = f"{benchmark_key}_{model_name}_{evaluator_name}_comparison"
    
#     # Store summary data
#     if benchmark_key not in pipeline.results_summary:
#         pipeline.results_summary[benchmark_key] = {}
#     if model_name not in pipeline.results_summary[benchmark_key]:
#         pipeline.results_summary[benchmark_key][model_name] = {}
    
#     pipeline.results_summary[benchmark_key][model_name][evaluator_name] = comparison_data
    
#     # Also log as a bar chart for key metrics
#     for metric_name in metric_names:
#         # Create a data table for the bar chart
#         bar_data = wandb.Table(columns=["Method", metric_name])
#         for method, results in method_results.items():
#             bar_data.add_data(method, results.metrics.get(metric_name, 0))
        
#         # Log bar chart with correct parameters
#         wandb.log({
#             f"{benchmark_key}_{evaluator_name}_{metric_name}_chart": 
#             wandb.plot.bar(
#                 bar_data,
#                 "Method",
#                 metric_name,
#                 title=f"{benchmark_key}-{evaluator_name}-{metric_name}"
#             )
#         })    

#     wandb.finish()
#     return pipeline