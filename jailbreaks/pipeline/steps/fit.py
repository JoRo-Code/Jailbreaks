import logging
from typing import List
from dataclasses import dataclass

from jailbreaks.methods import JailBreakMethod

logger = logging.getLogger(__name__)

@dataclass
class FitConfig:
    method_combinations: List[List[JailBreakMethod]]
    model_paths: List[str]
    refit: bool = True

def fit(config: FitConfig):
    logger.info("Step 1: Fitting methods")
    
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
                        method.fit(model_path, refit=config.refit)
                        method.save()
                except Exception as e:
                    logger.error(f"  Error fitting method {method_name}: {str(e)}")
            else:
                logger.info(f"  Method {method_name} doesn't require fitting")