import logging
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np

from jailbreaks.pipeline.utils import (
    fetch_all_artifacts, 
    load_evaluations,
    load_responses,
    FetchFilter
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class AggregateConfig:
    project_name: str
    responses_dir: Path
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
            fetch_filter=FetchFilter(
                run_ids=[config.eval_run_id],
                art_type="evaluation_results",
            ),
        )
        fetch_all_artifacts(
            project=config.project_name, 
            output_dir=config.responses_dir,
            fetch_filter=FetchFilter(
                art_type="responses",
            ),
        )
    
    eval_results = load_evaluations(config.evaluations_dir)
    responses = load_responses(config.responses_dir)
    logger.info("Aggregating results")
    
    return _aggregate_results(responses,eval_results, config)


def _aggregate_results(responses: dict, eval_results: dict, config: AggregateConfig):
    # aggregate all evaluations into a single dataframe
    
    # Create a list to store all individual data points
    all_rows = []
    
    for benchmark_key, model_name_d in eval_results.items():
        for model_name, method_combo_d in model_name_d.items():
            for method_combo, evaluator_name_d in method_combo_d.items():
                for evaluator_name, run_id_d in evaluator_name_d.items():
                    for run_id, result_df in run_id_d.items():
                        # Get response data
                        response_run_id = run_id.split("_")[-1]
                        run_id_responses = responses[benchmark_key][model_name][method_combo][response_run_id]
                        
                        # Add generation times to the result dataframe
                        gen_times = [r.gen_time for r in run_id_responses]
                        
                        # Create a copy of the result dataframe to avoid modifying the original
                        df = result_df.copy()
                        
                        # Add metadata columns
                        df['benchmark'] = benchmark_key
                        df['model'] = model_name
                        df['method'] = method_combo
                        df['evaluator'] = evaluator_name
                        df['run_id'] = run_id
                        
                        # Add generation time if available
                        if len(gen_times) == len(df):
                            df['gen_time'] = gen_times
                        elif len(gen_times) > 0:
                            # If lengths don't match, add the mean generation time to all rows
                            df['gen_time'] = np.mean(gen_times)
                        
                        # Convert numeric columns
                        for col in df.columns:
                            if col not in ['benchmark', 'model', 'method', 'evaluator', 'run_id']:
                                try:
                                    df[col] = pd.to_numeric(df[col])
                                except (ValueError, TypeError):
                                    pass
                        
                        # Append to the list of all rows
                        all_rows.append(df)
    
    # Combine all dataframes into one large dataframe
    if all_rows:
        combined_df = pd.concat(all_rows, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()  # Return empty dataframe if no data
                