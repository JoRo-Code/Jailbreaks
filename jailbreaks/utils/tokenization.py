from typing import List
import logging
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

MISTRAL_CHAT_TEMPLATE = """[INST]{instruction}[/INST]"""

def format_prompts(prompts: List[str], tokenizer: AutoTokenizer) -> List[str]:
    model_path = tokenizer.name_or_path.lower()
    
    if "qwen" in model_path:
        logger.debug(f"Using hardcoded Qwen chat template for {tokenizer.name_or_path}")
        return [QWEN_CHAT_TEMPLATE.format(instruction=prompt) for prompt in prompts]
    elif "llama" in model_path:
        logger.debug(f"Using hardcoded Llama chat template for {tokenizer.name_or_path}")
        return [LLAMA3_CHAT_TEMPLATE.format(instruction=prompt) for prompt in prompts]
    elif "mistral" in model_path:
        logger.debug(f"Using hardcoded Mistral chat template for {tokenizer.name_or_path}")
        return [MISTRAL_CHAT_TEMPLATE.format(instruction=prompt) for prompt in prompts]
    else:
        # If no method and not a recognized model, raise an error
        raise ValueError(
            f"Tokenizer {tokenizer.name_or_path} does not have a chat_template configured "
            f"and is not a recognized model type (Qwen/Llama). Cannot format prompts."
        )

def tokenize_prompts(prompts: List[str], tokenizer: AutoTokenizer) -> List[str]:
    if tokenizer.pad_token is None:
        logger.debug("Tokenizer does not have a pad token. Setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token # TODO: add custom?

    if tokenizer.padding_side != 'left':
        logger.debug(f"Tokenizer padding side is '{tokenizer.padding_side}'. Forcing to 'left'.")
        tokenizer.padding_side = 'left'
    return tokenizer(prompts, padding=True,truncation=False, return_tensors="pt")