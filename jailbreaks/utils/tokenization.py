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

PHI_4_CHAT_TEMPLATE = """
<|im_start|>user<|im_sep|>
{instruction}<|im_end|>
<|im_start|>assistant<|im_sep|>
"""

GEMMA_2_CHAT_TEMPLATE = """
<bos><start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

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
    elif "phi" in model_path:
        logger.debug(f"Using hardcoded Phi chat template for {tokenizer.name_or_path}")
        return [PHI_4_CHAT_TEMPLATE.format(instruction=prompt) for prompt in prompts]
    elif "gemma" in model_path:
        logger.debug(f"Using hardcoded Gemma chat template for {tokenizer.name_or_path}")
        return [GEMMA_2_CHAT_TEMPLATE.format(instruction=prompt) for prompt in prompts]
    else:
        # If no method and not a recognized model, raise an error
        raise ValueError(
            f"Tokenizer {tokenizer.name_or_path} does not have a chat_template configured "
            f"and is not a recognized model type. Cannot format prompts. Configure a chat template for this model in the tokenization.py file."
        )

def tokenize_prompts(prompts: List[str], tokenizer: AutoTokenizer) -> List[str]:
    return tokenizer(prompts, padding=True,truncation=False, return_tensors="pt")