"""GGUF judge using llama-cpp-python.

Drop-in replacement for LLMJudge (vLLM) that loads GGUF models
via llama-cpp-python and produces the same output format.
"""

import gc
from typing import Dict, List, Optional, Tuple, Union

from src.refusal_judge_prompt import extract_refusal_score, get_refusal_judge_prompt


class LLMJudgeGGUF:
    """Judge model for evaluating refusals using a GGUF model."""

    def __init__(
        self,
        model_path: str,
        max_model_len: int = 16384,
        n_gpu_layers: int = -1,
    ) -> None:
        """Initialize the GGUF judge.

        Args:
            model_path: Path to the .gguf judge model file.
            max_model_len: Maximum context length for the model.
            n_gpu_layers: Number of layers to offload to GPU (-1 = all).
        """
        from llama_cpp import Llama

        self.model_path = model_path
        self.max_model_len = max_model_len
        self.llm = Llama(
            model_path=model_path,
            n_ctx=max_model_len,
            n_gpu_layers=n_gpu_layers,
            logits_all=False,
            verbose=True,
        )

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
        """Evaluate question-answer pairs for refusal classification.

        Same signature and return format as LLMJudge.judge().

        Args:
            questions_answers: List of (question, answer) tuples.
            num_return_sequences: Number of judge outputs per pair.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            max_new_tokens: Maximum tokens to generate.
            thinking_string: Optional delimiter for chain-of-thought reasoning.

        Returns:
            List of dicts with "label" (float) and "judge_outputs" (List[str]).
        """
        # Normalize empty thinking strings to None
        if thinking_string == "":
            thinking_string = None

        results: List[Dict[str, Union[str, float, List[str]]]] = []

        for question, answer in questions_answers:
            prompt = get_refusal_judge_prompt(question, answer)
            messages = [{"role": "user", "content": prompt}]

            example_output: Dict[str, Union[str, float, List[str]]] = {
                "label": 0.0,
                "judge_outputs": [],
            }
            valid = 0
            total_score = 0.0

            for _ in range(num_return_sequences):
                response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )

                text = response["choices"][0]["message"]["content"] or ""

                if thinking_string is not None:
                    text_split = text.split(thinking_string)[-1]
                else:
                    text_split = text

                score = extract_refusal_score(text_split)
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

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Delete the model and free memory."""
        if hasattr(self, "llm") and self.llm is not None:
            del self.llm
            self.llm = None
            gc.collect()
