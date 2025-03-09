import math
import textwrap
from typing import List

import numpy as np
import torch
import vllm
from transformers import LogitsProcessor
from vllm.lora.request import LoRARequest

from .config import difficulty_lora_path, llm, tokenizer

# 3クラス分類
DIFFICULTY2TOKEN = {
    "<15 min fix": "easy",
    "15 min - 1 hour": "medium",
    "1-4 hours": "difficult",
    ">4 hours": "difficult",
}
TOKENS = ["easy", "medium", "difficult"]
TOKEN2LABEL = {token: i for i, token in enumerate(TOKENS)}


PROMPT_TEMPLATE: str = textwrap.dedent("""
    Below is a GitHub-related issue.

    {problem_statement}

    Please review the problem statement and estimate how long it will take to resolve the issue. Choose the appropriate word from the options below:
        easy: <15 min fix
        medium: 15 min - 1 hour
        difficult: >1 hour

    Answer only with one word (easy, medium, or difficult).
    """)


def make_inference_prompt(problem_statement, tokenizer):
    """
    正解を付加しない「Answer:」までのプロンプトを作成
    """
    messages = [
        {"role": "user", "content": PROMPT_TEMPLATE.format(problem_statement=problem_statement)},
    ]
    # 事前に定義したテンプレートをもとに、モデルの入力形式に合わせた文字列を作成
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not prompt.endswith("Answer:"):
        prompt += "Answer:"
    return prompt


def get_easy_probs(problem_statements: list[str]) -> np.ndarray:
    prompt_text_list = [make_inference_prompt(problem_statement, tokenizer) for problem_statement in problem_statements]

    keep_ids = []
    for x in TOKENS:
        c = tokenizer.encode(x, add_special_tokens=False)[0]
        keep_ids.append(c)

    class DigitLogitsProcessor(LogitsProcessor):
        def __init__(self, tokenizer):
            self.allowed_ids = keep_ids

        def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
            scores[self.allowed_ids] += 100
            return scores

    logits_processors = [DigitLogitsProcessor(tokenizer)]
    sampling_params = vllm.SamplingParams(
        n=1,  # Number of output sequences to return for each prompt.
        top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
        temperature=0,  # randomness of the sampling
        seed=777,  # Seed for reprodicibility
        skip_special_tokens=True,  # Whether to skip special tokens in the output.
        max_tokens=1,  # Maximum number of tokens to generate per output sequence.
        logits_processors=logits_processors,
        logprobs=5,
    )
    responses = llm.generate(
        prompt_text_list,
        sampling_params=sampling_params,
        use_tqdm=True,
        lora_request=LoRARequest("sql_adapter", 1, lora_path=difficulty_lora_path),
    )

    results = []
    errors = 0

    for i, response in enumerate(responses):
        try:
            x = response.outputs[0].logprobs[0]
            logprobs = []
            for k in keep_ids:
                if k in x:
                    logprobs.append(math.exp(x[k].logprob))
                else:
                    logprobs.append(0)
                    print(f"bad logits {i}")
            logprobs = np.array(logprobs)
            logprobs /= logprobs.sum()
            results.append(logprobs)
        except:
            # print(f"error {i}")
            results.append(np.array([0.25] * len(keep_ids)))
            errors += 1
    pred_array = np.array(results).reshape((-1, 3))

    easy_probs = pred_array[:, 0]

    return easy_probs
