import contextlib
import gc
from typing import Dict, List, Union

import ray
import torch
from transformers import PreTrainedTokenizer
from vllm import LLM, TokensPrompt
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)


def delete_llm(llm: Union[LLM, None]):
    """
    Deletes the llm pipeline and frees the GPU memory.

    Args:
        llm: VLM LLM object (if None, the function will do nothing)
    """

    if llm is None:
        return

    print("Deleting the llm pipeline and freeing the GPU memory.")
    try:
        vram_usage_before = torch.cuda.memory_allocated() / 1024**2
    except AttributeError:
        print(
            "You requested to clean the CUDA memory, but it seems that cuda is not initialized yet..."
        )
        return

    destroy_model_parallel()
    destroy_distributed_environment()
    try:
        llm.llm_engine.engine_core.shutdown()
    except AttributeError:
        try:
            llm.llm_engine.shutdown()
        except AttributeError:
            pass

    try:
        del llm.llm_engine.model_executor
    except AttributeError:
        try:
            del llm.llm_engine.engine_core.model_executor
        except AttributeError:
            pass

    del llm
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    try:
        vram_usage_after = torch.cuda.memory_allocated() / 1024**2
    except AttributeError:
        vram_usage_after = -1.00
        print(
            "Something went wrong while getting the VRAM usage after deleting the llm pipeline and freeing the GPU memory."
        )
    print(
        f"VRAM usage before: {vram_usage_before:.2f} MB, after: {vram_usage_after:.2f} MB"
    )
    if vram_usage_after < vram_usage_before or vram_usage_after < 128.00:
        print("Successfully deleted the llm pipeline and freed the GPU memory.")
    else:
        print(
            "Something went wrong while deleting the llm pipeline and freeing the GPU memory."
        )


def encode_conversation(
    conversations: List[List[Dict[str, str]]],
    tokenizer: PreTrainedTokenizer,
    max_model_len: int,
    max_new_tokens: int,
    strip_prompt: bool = False,
    add_generation_prompt: bool = True,
) -> List[TokensPrompt]:
    batch_messages = []
    for example in conversations:
        conv = tokenizer.apply_chat_template(
            example, tokenize=False, add_generation_prompt=add_generation_prompt
        )

        if strip_prompt:
            conv = conv.strip()
        conv = tokenizer.encode(conv, return_tensors=None)
        if len(conv) > (max_model_len - max_new_tokens):
            print(
                f"Prompt is too long for the model. Left truncation from {len(conv)} to {max_model_len - max_new_tokens} tokens."
            )
            conv = conv[len(conv) - (max_model_len - max_new_tokens) :]
        batch_messages.append(TokensPrompt(prompt_token_ids=conv))

    return batch_messages
