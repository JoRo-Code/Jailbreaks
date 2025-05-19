import os
from datetime import datetime
import wandb
import logging
from typing import List
from dataclasses import dataclass

from jailbreaks.methods import JailBreakMethod

logger = logging.getLogger(__name__)

@dataclass
class FitConfig:
    method_combinations: List[List[JailBreakMethod]]
    model_paths: List[str]
    project_name: str
    log_dir: str
    refit: bool = True

def fit(config: FitConfig):
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"fit_{date}"
    wandb.init(project=config.project_name, name=run_name, id=run_name)
    logger.info("Step 1: Fitting methods")
    
    fit_times = {}
    for method_combo in config.method_combinations:
        if not method_combo:
            continue  # Skip baseline (no methods)
            
        combo_name = "_".join(method.__str__().lower() for method in method_combo)
        logger.info(f"Fitting method combination: {combo_name}")
        
        for method in method_combo:
            method_name = method.__str__().lower()
            if hasattr(method, 'fit') and callable(method.fit):
                logger.info(f"  Fitting method: {method_name}")
                try:
                    for model_path in config.model_paths:
                        method.set_fit_dir(config.log_dir)
                        import time
                        start_time = time.time()
                        method.fit(model_path, refit=config.refit)
                        method.save()
                        end_time = time.time()
                        fit_times[(model_path, method_name)] = end_time - start_time
                except Exception as e:
                    logger.error(f"  Error fitting method {method_name}: {str(e)}")
            else:
                logger.info(f"  Method {method_name} doesn't require fitting")
    
    try:
        fit_times_for_wandb = {}
        for (model_path, method_name), t in fit_times.items():
            fit_times_for_wandb.setdefault(model_path, {})[method_name] = t
        wandb.log({f"fit_times_{date}": fit_times_for_wandb})
    except Exception as e:
        logger.error(f"Error logging fit times: {str(e)}")
    logger.info(f"Step 1: Fitting methods complete. Total time taken: {sum(fit_times.values())} seconds")
    csv_path = os.path.join(config.log_dir, f"fit_times_{date}.csv")
    os.makedirs(config.log_dir, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("model_path,method_name,time\n")
        for (model_path, method_name), time in fit_times.items():
            f.write(f"{model_path},{method_name},{time}\n")


    
