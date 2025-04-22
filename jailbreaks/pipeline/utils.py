import time
import logging
import pathlib
import functools
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import wandb

from jailbreaks.pipeline.schemas import (
    EvaluationResult,
    GeneratedResponse
)

logger = logging.getLogger(__name__)

@dataclass
class FetchFilter:
    run_ids: list[str] = None
    n_runs: int = None
    art_type: str = None
    method_names: list[str] = None
    model_names: list[str] = None
    benchmark_names: list[str] = None
    evaluator_names: list[str] = None

def fetch_artifacts_per_run(run, output_dir:pathlib.Path, fetch_filter: FetchFilter):
    matched = False
    for art in run.logged_artifacts():
        should_download = True
        
        if fetch_filter.art_type and art.type != fetch_filter.art_type:
            should_download = False
        
        if fetch_filter.model_names:
            model_match = False
            for model_name in fetch_filter.model_names:
                if model_name in art.name:
                    model_match = True
                    break
            if not model_match:
                should_download = False
        
        if fetch_filter.benchmark_names:
            benchmark_match = False
            for benchmark_name in fetch_filter.benchmark_names:
                if benchmark_name in art.name:
                    benchmark_match = True
                    break
            if not benchmark_match:
                should_download = False
        
        if fetch_filter.method_names:
            method_match = False
            for method_name in fetch_filter.method_names:
                if method_name in art.name:
                    method_match = True
                    break
            if not method_match:
                should_download = False
        
        if should_download:
            run_dir = output_dir
            run_dir.mkdir(parents=True, exist_ok=True)
            art.download(root=str(run_dir))
    return run.name, matched

def fetch_all_artifacts(
    project:str, 
    output_dir:pathlib.Path, 
    fetch_filter: FetchFilter,
    threads:int=8,
    ):
    runs: wandb.Api.runs = wandb.Api().runs(f"{project}")
    fetch_arts = functools.partial(
        fetch_artifacts_per_run, 
        output_dir=output_dir, 
        fetch_filter=fetch_filter
    )
    logger.info("Fetching artifacts from %s", project)
    
    if fetch_filter.run_ids:
        runs = [run for run in runs if run.id in fetch_filter.run_ids]
    
    matched_runs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
        future_to_run = {ex.submit(fetch_arts, run): run for run in runs}
        for future in concurrent.futures.as_completed(future_to_run):
            run_name, matched = future.result()
            logger.info("✓ %s", run_name)
            logger.info("  Matched_runs: %s", len(matched_runs))
            if matched:
                matched_runs.append(run_name)
                # Check if we've reached the desired number of runs
                if fetch_filter.n_runs and len(matched_runs) >= fetch_filter.n_runs:
                    for pending_future in [f for f in future_to_run if not f.done()]:
                        pending_future.cancel()
                    logger.info(f"Reached desired number of runs ({fetch_filter.n_runs}). Stopping.")
                    break
    
    matched_runs_count = len(matched_runs)
    logger.info(f"Total matched runs: {matched_runs_count}")
    return matched_runs_count

def load_structured_data(
    root_dir: Path, 
    file_pattern: str = "*.csv", 
    parser_func = None
) -> dict:
    """
    Generic function to load data from a structured directory hierarchy.
    
    Args:
        root_dir: Root directory to start the search
        file_pattern: Glob pattern to match files (default: "*.csv")
        parser_func: Function to parse each file. Should accept (file_path, context_dict)
                     where context_dict contains metadata about the file location
                     
    Returns:
        Nested dictionary representing the directory structure and parsed data
    """
    result = {}
    
    for benchmark_path in root_dir.glob("*"):
        if not benchmark_path.is_dir():
            continue
        
        benchmark_key = benchmark_path.name
        logger.info(f"Processing benchmark: {benchmark_key}")
        
        if benchmark_key not in result:
            result[benchmark_key] = {}
        
        for model_path in benchmark_path.glob("*"):
            if not model_path.is_dir():
                continue
            
            model_name = model_path.name
            logger.info(f"  Processing model: {model_name}")

            result[benchmark_key][model_name] = {}
            
            for combo_path in model_path.glob("*"):
                if not combo_path.is_dir():
                    continue
                
                method_combo = combo_path.name
                logger.info(f"    Processing method combo: {method_combo}")
                
                model_method_key = f"{model_name}_{method_combo}"
                result[benchmark_key][model_name][method_combo] = {}
                
                # Look for matching files
                for file_path in combo_path.glob(file_pattern):
                    logger.info(f"      Loading file: {file_path}")
                    
                    try:
                        context = {
                            "benchmark": benchmark_key,
                            "model_name": model_name,
                            "method_combo": method_combo,
                            "file_name": file_path.stem
                        }
                        
                        if parser_func:
                            parsed_data = parser_func(file_path, context)
                            result[benchmark_key][model_name][method_combo][file_path.stem] = parsed_data
                        else:
                            # Default behavior for CSV files if no parser provided
                            result[benchmark_key][model_name][method_combo][file_path.stem] = pd.read_csv(file_path)
                            
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
    
    return result

def response_parser(file_path: Path, context: dict) -> list[GeneratedResponse]:
    """Parser function for response CSV files"""
    response_df = pd.read_csv(file_path)
    
    responses = []
    for _, row in response_df.iterrows():
        gen_response = GeneratedResponse(
            prompt=row.get('prompt', ''),
            raw_prompt=row.get('raw_prompt', row.get('prompt', '')),
            response=row.get('response', ''),
            model_id=context["model_name"],
            method_combo=context["method_combo"],
            gen_time=row.get('gen_time', 0),
            metadata={
                "benchmark": context["benchmark"],
                "timestamp": time.time(),
            }
        )
        responses.append(gen_response)
    
    return responses

def evaluation_parser(file_path: Path, context: dict) -> list[EvaluationResult]:
    """Parser function for response CSV files"""
    response_df = pd.read_csv(file_path)
    
    return response_df  

def load_responses(responses_dir: Path) -> dict:
    """
    Scans the responses directory structure and loads all CSV files found.
    Directory structure expected: benchmark_key/model_name/method_combo/responses_run_id.csv
    """
    return load_structured_data(responses_dir, "*.csv", response_parser)

def load_evaluations(evaluations_dir: Path) -> dict:
    """
    Scans the evaluations directory structure and loads all CSV files found.
    Directory structure expected: benchmark_key/model_name/method_combo/evaluator_name/evaluations_run_id.csv
    """
    result = {}
    
    for benchmark_path in evaluations_dir.glob("*"):
        if not benchmark_path.is_dir():
            continue
        
        benchmark_key = benchmark_path.name
        logger.info(f"Processing benchmark: {benchmark_key}")
        
        if benchmark_key not in result:
            result[benchmark_key] = {}
        
        for model_path in benchmark_path.glob("*"):
            if not model_path.is_dir():
                continue
            
            model_name = model_path.name
            logger.info(f"  Processing model: {model_name}")
            if model_name not in result[benchmark_key]:
                result[benchmark_key][model_name] = {}
            
            for combo_path in model_path.glob("*"):
                if not combo_path.is_dir():
                    continue
                
                method_combo = combo_path.name
                logger.info(f"    Processing method combo: {method_combo}")

                if method_combo not in result[benchmark_key][model_name]:
                    result[benchmark_key][model_name][method_combo] = {}
                
                # Add evaluator level
                for evaluator_path in combo_path.glob("*"):
                    if not evaluator_path.is_dir():
                        continue
                    
                    evaluator_name = evaluator_path.name
                    logger.info(f"      Processing evaluator: {evaluator_name}")
                    
                    if evaluator_name not in result[benchmark_key][model_name][method_combo]:
                        result[benchmark_key][model_name][method_combo][evaluator_name] = {}
                    
                    # Look for matching files
                    for file_path in evaluator_path.glob("*.csv"):
                        logger.info(f"        Loading file: {file_path}")
                        
                        try:
                            context = {
                                "benchmark": benchmark_key,
                                "model_name": model_name,
                                "method_combo": method_combo,
                                "evaluator_name": evaluator_name,
                                "file_name": file_path.stem
                            }
                            
                            parsed_data = evaluation_parser(file_path, context)
                            result[benchmark_key][model_name][method_combo][evaluator_name][file_path.stem] = parsed_data
                                
                        except Exception as e:
                            logger.error(f"Error processing {file_path}: {str(e)}")
                            import traceback
                            logger.error(traceback.format_exc())
    
    return result

if __name__ == "__main__":
    fetch_all_artifacts(
        project="test", 
        output_dir=pathlib.Path("download"), 
        art_type="responses", 
        threads=8
    )
