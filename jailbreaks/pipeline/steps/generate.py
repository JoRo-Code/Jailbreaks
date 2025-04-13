
from logging import getLogger
from jailbreaks.pipeline.pipeline import JailbreakPipeline, GeneratedResponse
from tqdm import tqdm
import time
import json
import pandas as pd
import wandb

from jailbreaks.llm import LLM

logger = getLogger(__name__)

def generate(pipeline: JailbreakPipeline):
    logger.info("Step 2: Generating responses")
    run_name = f"responses_{pipeline.run_id}"
    
    wandb.init(project=pipeline.project_name, name=run_name, id=pipeline.run_id)
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
                
                output_path = pipeline.responses_dir / f"responses_{benchmark_key}_{model_short_name}_{combo_key}.json"
    
                with open(output_path, 'w') as f:
                    json.dump([r.to_dict() for r in responses], f, indent=2)
                csv_file = pipeline.responses_dir / f"responses_{benchmark_key}_{model_short_name}_{combo_key}.csv"
                response_df.to_csv(csv_file, index=False)
                logger.info(f"Saved responses to {output_path}")

                # TODO: Log artifact
                
                # response_table = wandb.Table(dataframe=response_df)
                
                # response_table_artifact = wandb.Artifact(f"{benchmark_key}_{model_short_name}_{combo_key}_responses", type="dataset")
                # response_table_artifact.add(response_table, "response_table")

                # response_table_artifact.add_file(csv_file)
                # response_table_artifact.save()

                # wandb.log_artifact(response_table_artifact)

                
    