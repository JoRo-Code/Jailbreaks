
from logging import getLogger
from jailbreaks.pipeline.pipeline import JailbreakPipeline

logger = getLogger(__name__)

def fit(pipeline: JailbreakPipeline, refit: bool = True):
    logger.info("Step 1: Fitting methods")
    
    for method_combo in pipeline.method_combinations:
        if not method_combo:
            continue  # Skip baseline (no methods)
            
        combo_name = "_".join(method.__str__().lower() for method in method_combo)
        logger.info(f"Fitting method combination: {combo_name}")
        
        for method in method_combo:
            method_name = method.__str__().lower()
            if hasattr(method, 'fit') and callable(method.fit):
                logger.info(f"  Fitting method: {method_name}")
                try:
                    # If method needs to be fitted on a model
                    if not method_name in pipeline.fitted_methods:
                        for model_path in pipeline.model_paths:
                            method.fit(model_path, refit=refit)
                            method.save()
                        pipeline.fitted_methods[method_name] = method
                        logger.info(f"  Method {method_name} fitted and saved")
                except Exception as e:
                    logger.error(f"  Error fitting method {method_name}: {str(e)}")
            else:
                logger.info(f"  Method {method_name} doesn't require fitting")