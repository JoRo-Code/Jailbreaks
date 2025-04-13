from jailbreaks.pipeline.pipeline import JailbreakPipeline
from jailbreaks.utils.model_loading import load_model, load_tokenizer

def download(pipeline: JailbreakPipeline):
    from jailbreaks.utils.model_loading import load_model, load_tokenizer
    for model_path in pipeline.model_paths:
        load_model(model_path, device="cpu")
        load_tokenizer(model_path, device="cpu")