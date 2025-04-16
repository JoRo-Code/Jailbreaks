import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from jailbreaks.methods.model import DiffInMeans
from jailbreaks.utils.model_loading import load_model, load_tokenizer
from jailbreaks.data import get_advbench_instructions, get_harmless_instructions

model_path = "Qwen/Qwen2-0.5B-Instruct"

# ensure the models are downloaded, no need to move to GPU
load_model(model_path, device="cpu")
load_tokenizer(model_path, device="cpu")

harmful_prompts = get_advbench_instructions()
harmless_prompts = get_harmless_instructions()
N = 32

# initialize method
d = DiffInMeans(use_cache=True, harmful_prompts=harmful_prompts[:N], harmless_prompts=harmless_prompts[:N])

# fit the model
d.fit(model_path, refit=True)
d.save()
