import os
import logging
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List
import uuid
from contextlib import suppress

import pandas as pd
import wandb
from tqdm import tqdm

from jailbreaks.llm import LLM
from jailbreaks.pipeline.schemas import (
    GeneratedResponse
)
from jailbreaks.benchmarks import Benchmark
from jailbreaks.methods import JailBreakMethod

logger = logging.getLogger(__name__)

@dataclass
class GenerateConfig:
    project_name: str
    run_id: str
    model_paths: List[str]
    method_combinations: List[List[JailBreakMethod]]
    benchmarks: List[Benchmark]
    batch_size: int
    output_dir: Path = Path("tmp_responses")
    registry_artifact_name: str = "generations_registry"
    

def generate(config: GenerateConfig):
    run_id = str(uuid.uuid4())[:8] if config.run_id is None else config.run_id
    logger.info("Step 2: Generating responses")
    run_name = f"responses_{run_id}"
    
    wandb.init(
        project=config.project_name, 
        name=run_name, 
        id=run_name, 
        config={
            "model_paths": config.model_paths,
            "method_combinations": config.method_combinations,
            "benchmarks": config.benchmarks,
            "run_id": config.run_id
        }
    )
    logger.info(f"W&B experiment initialized: {run_name}")
    generation_start_time = time.time()
    _generate_responses_internal(config)
    total_generation_time = time.time() - generation_start_time
    logger.info(f"Response generation completed in {total_generation_time:.2f}s")
    wandb.finish()

def _generate_responses_internal(config: GenerateConfig):
    
    # Track overall generation metrics
    overall_metrics = {
        "total_models": len(config.model_paths),
        "total_method_combos": len(config.method_combinations),
        "total_benchmarks": len(config.benchmarks),
    }
    wandb.config.update(overall_metrics)

    method_model_times = {}
    generated_responses = {}
    
    # Initialize method_model_times structure
    for benchmark in config.benchmarks:
        benchmark_name = benchmark.__str__()
        benchmark_key = benchmark_name.lower().replace(" ", "_")
        
        if benchmark_key not in method_model_times:
            method_model_times[benchmark_key] = {}
        
        for model_path in config.model_paths:
            model_short_name = model_path.split("/")[-1]
            
            if model_short_name not in method_model_times[benchmark_key]:
                method_model_times[benchmark_key][model_short_name] = {}
            
            for method_combo in config.method_combinations:
                if not method_combo:
                    combo_name = "baseline"
                else:
                    combo_name = "_".join(method.__str__().lower() for method in method_combo)
                
                if combo_name not in method_model_times[benchmark_key][model_short_name]:
                    method_model_times[benchmark_key][model_short_name][combo_name] = {
                        'total_time': 0,
                        'avg_gen_time': 0,
                        'num_samples': 0,
                        'batch_times': []
                    }
    

    def _load_registry() -> pd.DataFrame:
        """
        Download the latest registry artefact if it exists, otherwise
        return an empty DataFrame.
        """
        api = wandb.Api()
        with suppress(Exception):
            art = api.artifact(f"{config.project_name}/{config.registry_artifact_name}:latest")
            csv_path = Path(art.download()).joinpath("registry.csv")
            return pd.read_csv(
                csv_path,
                dtype={
                    "run_name"      : str,
                    "run_id"        : str,
                    "benchmark"     : str,
                    "model"         : str,
                    "method_combo"  : str,
                    "generation_no" : "Int64",
                    "num_responses" : "Int64",
                    "total_gen_time": float,
                }
            )
        
        return pd.DataFrame(columns=[
            "run_name", "run_id",
            "benchmark", "model", "method_combo",
            "generation_no",
            "num_responses", "total_gen_time"
        ])

    registry_df = _load_registry()
    new_rows = [] 

    for benchmark in config.benchmarks:
        benchmark_name = benchmark.__str__()
        benchmark_key = benchmark_name.lower().replace(" ", "_")
        
        logger.info(f"Generating responses for benchmark: {benchmark_name}")
        generated_responses[benchmark_key] = {}
        
        for model_path in config.model_paths:
            model_short_name = model_path.split("/")[-1]
            logger.info(f"  Using model: {model_short_name}")
            
            for method_combo in config.method_combinations:
                if not method_combo:
                    combo_name = "baseline"
                else:
                    combo_name = "_".join(method.__str__().lower() for method in method_combo)
                
                combo_key = combo_name.lower().replace(" ", "_")
                logger.info(f"    Generating with method combo: {combo_name}")
                
                jailbreak_model = LLM(model_path, method_combo)
                
                responses = []
                generation_start = time.time()
                
                # Get prompts from benchmark
                prompts = benchmark.get_prompts()

                batch_size = config.batch_size
                for batch_idx, batch_start in enumerate(tqdm(range(0, len(prompts), batch_size), desc=f"{model_short_name}_{combo_name}")):
                    batch_end = min(batch_start + batch_size, len(prompts))
                    batch_prompts = prompts[batch_start:batch_end]
                
                    try:
                        batch_start_time = time.time()

                        # Generate response
                        raw_prompts = [jailbreak_model.prepare_prompt(prompt) for prompt in batch_prompts]
                        batched_responses = jailbreak_model.generate_batch(batch_prompts, max_new_tokens=benchmark.max_new_tokens)
                        
                        # Calculate batch generation time
                        batch_gen_time = time.time() - batch_start_time
                        per_sample_time = batch_gen_time / len(batch_prompts)
                        
                        # Track timing for this batch
                        method_model_times[benchmark_key][model_short_name][combo_name]['batch_times'].append({
                            'batch_idx': batch_idx,
                            'batch_size': len(batch_prompts),
                            'total_time': batch_gen_time,
                            'per_sample_time': per_sample_time
                        })
                        
                        # Store generated response
                        gen_responses = [GeneratedResponse(
                            prompt=batch_prompts[i],
                            raw_prompt=raw_prompts[i],
                            response=batched_responses[i],
                            model_id=model_path,
                            method_combo=combo_name,
                            gen_time=per_sample_time,
                            metadata={
                                "benchmark": benchmark_name,
                                "prompt_index": i,
                                "timestamp": time.time(),
                                "gen_time": per_sample_time,
                                "batch_idx": batch_idx,
                                "batch_size": len(batch_prompts)
                            }
                        ) for i, _ in enumerate(batched_responses)]
                        
                        responses.extend(gen_responses)
                    
                    except Exception as e:
                        import traceback
                        logger.error(f"Error generating response for batch {batch_idx}: {str(e)}\n{traceback.format_exc()}")
                
                # Calculate and store overall timing metrics for this method/model
                total_generation_time = time.time() - generation_start
                num_samples = len(responses)
                avg_gen_time = total_generation_time / max(num_samples, 1)
                
                method_model_times[benchmark_key][model_short_name][combo_name].update({
                    'total_time': total_generation_time,
                    'avg_gen_time': avg_gen_time,
                    'num_samples': num_samples
                })
                
                jailbreak_model.clear_cache()
                
                # Log generation metrics
                wandb.log({
                    "benchmark": benchmark_name,
                    "model": model_short_name,
                    "method_combo": combo_name,
                    "num_responses": len(responses),
                    "generation_time": total_generation_time,
                    "avg_generation_time": avg_gen_time
                })
                
                # Save responses
                response_df = pd.DataFrame({
                    "prompt": [resp.prompt for resp in responses],
                    "raw_prompt": [resp.raw_prompt for resp in responses],
                    "response": [resp.response for resp in responses],
                    "model": [model_short_name for _ in responses],
                    "method_combo": [combo_name for _ in responses],
                    "gen_time": [resp.metadata["gen_time"] for resp in responses]
                })
                    
                artifact_name = f"{benchmark_key}_{model_short_name}_{combo_key}_responses-{config.run_id}"
                
                # log table for visibility in wandb
                response_table = wandb.Table(dataframe=response_df)
                wandb.log({artifact_name: response_table})
                
                # add file for download
                path = f"{benchmark_key}/{model_short_name}/{combo_key}/responses-{config.run_id}.csv"

                # could also just have them locally, without temp file
                csv_path = config.output_dir / path
                os.makedirs(csv_path.parent, exist_ok=True)
                response_df.to_csv(csv_path, index=False)

                artifact = wandb.Artifact(name=artifact_name, type="responses")
                artifact.add_file(csv_path, name=path)
                
                wandb.log_artifact(artifact)

                prev_runs_mask = (
                    (registry_df["benchmark"]    == benchmark_key) &
                    (registry_df["model"]        == model_short_name) &
                    (registry_df["method_combo"] == combo_name)
                )

                # Look up the previous generation_no (if any) and add 1
                if prev_runs_mask.any():
                    last_gen = registry_df.loc[prev_runs_mask, "generation_no"].max()
                    next_gen_no = (0 if pd.isna(last_gen) else int(last_gen)) + 1
                else:
                    next_gen_no = 1

                new_rows.append({
                    "run_name"      : wandb.run.name,
                    "run_id"        : config.run_id,
                    "benchmark"     : benchmark_key,
                    "model"         : model_short_name,
                    "method_combo"  : combo_name,
                    "generation_no" : next_gen_no,
                    "num_responses" : num_samples,
                    "total_gen_time": total_generation_time
                })
    
    if new_rows:
        # merge the old registry with the freshly-generated rows
        registry_df = pd.concat(
            [registry_df, pd.DataFrame(new_rows)],
            ignore_index=True
        )

        # Keep only the *latest* generation for every combination
        registry_df = (
            registry_df
            .sort_values("generation_no")
            .drop_duplicates(
                subset=["benchmark", "model", "method_combo"],
                keep="last"
            )
            .reset_index(drop=True)
        )


        str_cols  = ["run_name", "run_id",
                     "benchmark", "model", "method_combo"]
        num_cols  = ["generation_no", "num_responses", "total_gen_time"]

        registry_df[str_cols] = registry_df[str_cols].fillna("").astype(str)
        registry_df[num_cols] = registry_df[num_cols].apply(
            pd.to_numeric, errors="coerce"
        )

        # save locally
        reg_path = config.output_dir / "registry.csv"
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        registry_df.to_csv(reg_path, index=False)

        # create new version of the artefact
        reg_art = wandb.Artifact(name=config.registry_artifact_name, type="registry")
        reg_art.add_file(reg_path, name="registry.csv")
        wandb.log_artifact(reg_art)

        # also put it in the current run's logs for quick inspection
        wandb.log({
            "generation_registry": wandb.Table(dataframe=registry_df)
        })

        # Build a compact summary: how many generations per combination?
        summary_df = (
            registry_df
            .groupby(["benchmark", "model", "method_combo"], as_index=False)
            .agg(total_generations=("generation_no", "max"))
            .sort_values(["benchmark", "model", "method_combo"])
        )

        # log the summary table for quick inspection
        wandb.log({
            "generation_registry": wandb.Table(dataframe=registry_df),
            "generation_summary" : wandb.Table(dataframe=summary_df)
        })
    