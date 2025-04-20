import time
import logging
import pathlib
import functools
import concurrent.futures
from pathlib import Path

import pandas as pd
import wandb

from jailbreaks.pipeline.pipeline import (
    EvaluationResult,
    GeneratedResponse
)

logger = logging.getLogger(__name__)

def fetch_artifacts(run, output_dir:pathlib.Path, art_type:str):
    for art in run.logged_artifacts():
        if art.type == art_type:
            run_dir = output_dir
            run_dir.mkdir(parents=True, exist_ok=True)
            art.download(root=str(run_dir))
    return run.name

def fetch_all_artifacts(
    project:str, 
    output_dir:pathlib.Path, 
    art_type:str, 
    run_ids:list[str]=None,
    threads:int=8
    ):
    runs: wandb.Api.runs = wandb.Api().runs(f"{project}")
    fetch_arts = functools.partial(fetch_artifacts, output_dir=output_dir, art_type=art_type)
    logger.info("Fetching artifacts from %s", project)
    
    if run_ids:
        runs = [run for run in runs if run.id in run_ids]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
        for finished in ex.map(fetch_arts, runs):
            logger.info("✓ %s", finished)

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
            
            for combo_path in model_path.glob("*"):
                if not combo_path.is_dir():
                    continue
                
                method_combo = combo_path.name
                logger.info(f"    Processing method combo: {method_combo}")
                
                model_method_key = f"{model_name}_{method_combo}"
                result[benchmark_key][model_method_key] = {}
                
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
                            result[benchmark_key][model_method_key][file_path.stem] = parsed_data
                        else:
                            # Default behavior for CSV files if no parser provided
                            result[benchmark_key][model_method_key][file_path.stem] = pd.read_csv(file_path)
                            
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
