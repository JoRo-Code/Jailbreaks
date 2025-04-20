import os
import logging
import time
from pathlib import Path

import pandas as pd
import wandb
from tqdm import tqdm

from jailbreaks.llm import LLM
from jailbreaks.pipeline.pipeline import (
    JailbreakPipeline, 
    GeneratedResponse
)

logger = logging.getLogger(__name__)

def generate(pipeline: JailbreakPipeline):
    logger.info("Step 2: Generating responses")
    run_name = f"responses_{pipeline.run_id}"
    
    wandb.init(
        project=pipeline.project_name, 
        name=run_name, 
        id=run_name, 
        config={
            "model_paths": pipeline.model_paths,
            "method_combinations": pipeline.method_combinations,
            "benchmarks": pipeline.benchmarks,
            "run_id": pipeline.run_id
        }
    )
    logger.info(f"W&B experiment initialized: {run_name}")
    generation_start_time = time.time()
    _generate_responses_internal(pipeline)
    total_generation_time = time.time() - generation_start_time
    logger.info(f"Response generation completed in {total_generation_time:.2f}s")
    wandb.finish()

def _generate_responses_internal(pipeline: JailbreakPipeline):
    
    # Track overall generation metrics
    overall_metrics = {
        "total_models": len(pipeline.model_paths),
        "total_method_combos": len(pipeline.method_combinations),
        "total_benchmarks": len(pipeline.benchmarks),
    }
    wandb.config.update(overall_metrics)
            
    
    # Initialize method_model_times structure
    for benchmark in pipeline.benchmarks:
        benchmark_name = benchmark.__str__()
        benchmark_key = benchmark_name.lower().replace(" ", "_")
        
        if benchmark_key not in pipeline.method_model_times:
            pipeline.method_model_times[benchmark_key] = {}
        
        for model_path in pipeline.model_paths:
            model_short_name = model_path.split("/")[-1]
            
            if model_short_name not in pipeline.method_model_times[benchmark_key]:
                pipeline.method_model_times[benchmark_key][model_short_name] = {}
            
            for method_combo in pipeline.method_combinations:
                if not method_combo:
                    combo_name = "baseline"
                else:
                    combo_name = "_".join(method.__str__().lower() for method in method_combo)
                
                if combo_name not in pipeline.method_model_times[benchmark_key][model_short_name]:
                    pipeline.method_model_times[benchmark_key][model_short_name][combo_name] = {
                        'total_time': 0,
                        'avg_gen_time': 0,
                        'num_samples': 0,
                        'batch_times': []
                    }
    
    for benchmark in pipeline.benchmarks:
        benchmark_name = benchmark.__str__()
        benchmark_key = benchmark_name.lower().replace(" ", "_")
        
        logger.info(f"Generating responses for benchmark: {benchmark_name}")
        pipeline.generated_responses[benchmark_key] = {}
        
        for model_path in pipeline.model_paths:
            model_short_name = model_path.split("/")[-1]
            logger.info(f"  Using model: {model_short_name}")
            
            for method_combo in pipeline.method_combinations:
                if not method_combo:
                    combo_name = "baseline"
                    method_configs = [{"name": "baseline"}]
                else:
                    combo_name = "_".join(method.__str__().lower() for method in method_combo)
                    method_configs = [pipeline._get_method_config(method) for method in method_combo]
                
                combo_key = combo_name.lower().replace(" ", "_")
                logger.info(f"    Generating with method combo: {combo_name}")
                
                jailbreak_model = LLM(model_path, method_combo)
                
                responses = []
                generation_start = time.time()
                
                # Get prompts from benchmark
                prompts = benchmark.get_prompts()

                batch_size = pipeline.batch_size
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
                        pipeline.method_model_times[benchmark_key][model_short_name][combo_name]['batch_times'].append({
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
                
                pipeline.method_model_times[benchmark_key][model_short_name][combo_name].update({
                    'total_time': total_generation_time,
                    'avg_gen_time': avg_gen_time,
                    'num_samples': num_samples
                })
                
                jailbreak_model.clear_cache()
                
                # Store responses
                key = f"{model_short_name}_{combo_key}"
                pipeline.generated_responses[benchmark_key][key] = responses
                
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
                    
                artifact_name = f"{benchmark_key}_{model_short_name}_{combo_key}_responses-{pipeline.run_id}"
                
                # log table for visibility in wandb
                response_table = wandb.Table(dataframe=response_df)
                wandb.log({artifact_name: response_table})
                
                # add file for download
                path = f"{benchmark_key}/{model_short_name}/{combo_key}/responses-{pipeline.run_id}.csv"

                csv_path = Path("tmp_responses") / path
                os.makedirs(csv_path.parent, exist_ok=True)
                response_df.to_csv(csv_path, index=False)

                artifact = wandb.Artifact(name=artifact_name, type="responses")
                artifact.add_file(csv_path, name=path)
                
                os.remove(csv_path)

                wandb.log_artifact(artifact) # TODO: check if need to use run - run.log_artifact(artifact)

                
    