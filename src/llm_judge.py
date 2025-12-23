from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt

from src.refusal_judge_prompt import (
    extract_refusal_score,
    get_refusal_judge_prompt,
)
from src.utils import delete_llm, encode_conversation


class LLMJudge:
    """
    A judge model for evaluating whether LLM responses constitute refusals.

    Uses a specified LLM to analyze question-answer pairs and classify whether
    the answer represents a refusal to answer or a genuine attempt to address
    the question.
    """

    def __init__(
        self,
        model_name: str,
        max_model_len: int,
        gpu_memory_utilization: float = 0.95,
        tensor_parallel_size: int = torch.cuda.device_count(),
    ) -> None:
        """
        Initialize the LLM judge with specified model and configuration.

        Args:
            model_name (str): Name or path of the model to use for judging
            max_model_len (int): Maximum sequence length for the model
            gpu_memory_utilization (float, optional): Fraction of GPU memory to use.
                                                     Defaults to 0.95
            tensor_parallel_size (int, optional): Number of GPUs for tensor parallelism.
                                                 Defaults to number of available GPUs

        Returns:
            None
        """
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=True,  # We use the same large prompt with the instructions for every input
        )

        self.get_judge_prompt = get_refusal_judge_prompt
        self.extract_score = extract_refusal_score

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def judge(
        self,
        questions_answers: List[Tuple[str, str]],
        num_return_sequences: int = 1,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        max_new_tokens: int = 8196,
        thinking_string: Optional[str] = None,
    ) -> List[Dict[str, Union[str, float, List[str]]]]:
        """
        Evaluate conversations to determine if responses constitute refusals.

        Takes a list of conversations and uses the LLM to judge whether each
        response represents a refusal to answer or a genuine attempt to address
        the question. Returns classification scores and judge outputs.

        Args:
            questions_answers (List[Tuple[str, str]]): List of questions and answers.
                                                       where each question and answer is a tuple.
            temperature (float, optional): Sampling temperature for generation.
                                         Defaults to 0.6
            top_p (float, optional): Top-p (nucleus) sampling parameter.
                                   Defaults to 0.95
            top_k (int, optional): Top-k sampling parameter. Defaults to 20
            max_new_tokens (int, optional): Maximum tokens to generate. Defaults to 8196
            thinking_string (Optional[str], optional): String to split on for
                                                     chain-of-thought reasoning.
                                                     If provided, only text after this
                                                     string is used for scoring.
                                                     Defaults to None

        Returns:
            List[Dict[str, Union[str, float, List[str]]]]: List of judgment results,
                where each result contains:
                - 'label' (float): Average refusal score (1.0=refusal, -1.0=not refusal,
                                 0.0=error)
                - 'judge_outputs' (List[str]): Raw judge response texts
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            n=num_return_sequences,
            truncate_prompt_tokens=self.max_model_len - max_new_tokens,
        )

        conversations: List[List[Dict[str, str]]] = [
            [{"role": "user", "content": self.get_judge_prompt(question, answer)}]
            for question, answer in questions_answers
        ]

        batch_messages: List[TokensPrompt] = encode_conversation(
            conversations=conversations,
            tokenizer=self.tokenizer,
            max_model_len=self.max_model_len,
            max_new_tokens=max_new_tokens,
        )

        # Normalize empty thinking strings to None so downstream split logic is safe
        if thinking_string == "":
            thinking_string = None

        outputs = self.llm.generate(
            prompts=batch_messages,
            sampling_params=sampling_params,
            use_tqdm=None,
        )

        results: List[Dict[str, Union[str, float, List[str]]]] = []
        for result in outputs:
            example_output: Dict[str, Union[str, float, List[str]]] = {
                "label": 0.0,
                "judge_outputs": [],
            }
            valid = 0
            total_score = 0.0
            for output in result.outputs:
                text = output.text
                if thinking_string is not None:
                    text_split = text.split(thinking_string)[-1]
                else:
                    text_split = text
                score = self.extract_score(text_split)
                example_output["judge_outputs"].append(text)
                if score == 0.0:
                    continue
                valid += 1
                total_score += score

            if valid == 0:
                example_output["label"] = 0.0
            else:
                example_output["label"] = total_score / valid
            results.append(example_output)
        return results

    def __del__(self):
        self.close()

    def close(self):
        """
        Delete the LLM instance and free the GPU memory.
        """
        delete_llm(self.llm)
