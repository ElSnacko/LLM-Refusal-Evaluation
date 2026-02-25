"""GGUF answer generator using llama-cpp-python.

Drop-in replacement for GenerateAnswers (vLLM) that loads GGUF models
via llama-cpp-python and produces the same output format.

The logprob parsing logic (geom_mean_prob / parse_log_progs) is inlined
here so that this module has no dependency on vLLM.
"""

import gc
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


def _geom_mean_prob(logs: List[float]) -> float:
    """Geometric mean probability from log-probabilities (mirrors answer_generator.geom_mean_prob)."""
    if logs is None or len(logs) == 0:
        return -1.0
    avg_log = sum(logs) / len(logs)
    return math.exp(avg_log)


class _GGUFTokenCandidate:
    """Adapter satisfying the TokenCandidate protocol for llama-cpp-python logprobs."""

    __slots__ = ("rank", "logprob", "decoded_token")

    def __init__(self, token: str, logprob: float, rank: int = 1) -> None:
        self.rank = rank
        self.logprob = logprob
        self.decoded_token = token


def _adapt_chat_logprobs(
    logprobs_content: List[Dict[str, Any]],
) -> Sequence[Mapping[str, _GGUFTokenCandidate]]:
    """Convert llama-cpp-python OpenAI-format logprobs to parse_log_progs format.

    llama-cpp-python returns per-position dicts with keys "token" and "logprob".
    parse_log_progs expects Sequence[Mapping[str, TokenCandidate]] where each
    mapping has {decoded_token_str: candidate_with_rank_logprob_decoded_token}.
    """
    adapted: List[Mapping[str, _GGUFTokenCandidate]] = []
    for pos in logprobs_content:
        token = pos["token"]
        logprob = pos["logprob"]
        candidate = _GGUFTokenCandidate(token=token, logprob=logprob, rank=1)
        adapted.append({token: candidate})
    return adapted


def _parse_log_probs(
    logprobs: Sequence[Mapping[str, _GGUFTokenCandidate]],
    thinking_string: Optional[str] = None,
) -> Tuple[float, float, float]:
    """Parse per-token logprobs into thinking/answer segments.

    Mirrors answer_generator.parse_log_progs but uses only stdlib math
    so there is no torch/vLLM dependency.
    """
    logprobs_answer: List[float] = []
    logprobs_think: List[float] = []
    current: List[float] = logprobs_think if thinking_string else logprobs_answer

    for output in logprobs:
        values = output.values()
        chosen = max(values, key=lambda x: x.rank)
        current.append(chosen.logprob)
        if thinking_string is not None and chosen.decoded_token == thinking_string:
            current = logprobs_answer

    if thinking_string is None:
        answer_prob = _geom_mean_prob(logprobs_answer)
        thinking_prob = 0.0
        cum = answer_prob
    else:
        thinking_prob = _geom_mean_prob(logprobs_think)
        answer_prob = _geom_mean_prob(logprobs_answer)
        both = (logprobs_think or []) + (logprobs_answer or [])
        cum = _geom_mean_prob(both)

    return thinking_prob, answer_prob, cum


class GenerateAnswersGGUF:
    """Wrapper for generating answers from a GGUF model with log-probability metrics."""

    def __init__(
        self,
        model_path: str,
        max_model_len: int = 8192,
        n_gpu_layers: int = -1,
    ) -> None:
        """Initialize a GGUF model runner.

        Args:
            model_path: Path to the .gguf model file.
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
            logits_all=True,
            verbose=True,
        )

    def generate_answers(
        self,
        questions: List[str],
        max_new_tokens: int,
        num_return_sequences: int,
        thinking_string: Optional[str] = None,
        strip_prompt: bool = False,
        repetition_penalty: float = 1.0,
    ) -> List[List[Dict[str, Union[str, float]]]]:
        """Generate model answers and compute segment-wise probabilities.

        Same signature and return format as GenerateAnswers.generate_answers().

        Args:
            questions: Batch of questions.
            max_new_tokens: Maximum number of new tokens to generate.
            num_return_sequences: Number of sampled outputs per input.
            thinking_string: Optional delimiter between thinking/answer segments.
            strip_prompt: Unused (kept for interface compatibility).
            repetition_penalty: Repetition penalty (mapped to repeat_penalty).

        Returns:
            Nested list: for each question, a list of dicts with keys
            "text", "thinking_prob", "answer_prob", "cum".
        """
        outputs: List[List[Dict[str, Union[str, float]]]] = []

        for question in questions:
            messages = [{"role": "user", "content": question.strip()}]
            example_output: List[Dict[str, Union[str, float]]] = []

            for _ in range(num_return_sequences):
                response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=-1,
                    repeat_penalty=repetition_penalty,
                    logprobs=True,
                    top_logprobs=1,
                )

                choice = response["choices"][0]
                text = choice["message"]["content"] or ""

                # Extract logprobs if available
                logprobs_data = choice.get("logprobs")
                if logprobs_data and logprobs_data.get("content"):
                    adapted = _adapt_chat_logprobs(logprobs_data["content"])
                    thinking_prob, answer_prob, cum = _parse_log_probs(
                        adapted, thinking_string
                    )
                else:
                    thinking_prob = 0.0
                    answer_prob = -1.0
                    cum = -1.0

                example_output.append(
                    {
                        "text": text,
                        "thinking_prob": thinking_prob,
                        "answer_prob": answer_prob,
                        "cum": cum,
                    }
                )

            outputs.append(example_output)

        return outputs

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Delete the model and free memory."""
        if hasattr(self, "llm") and self.llm is not None:
            del self.llm
            self.llm = None
            gc.collect()
