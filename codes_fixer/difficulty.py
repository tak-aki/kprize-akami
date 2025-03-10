import math
import os
import textwrap
from typing import List

import numpy as np
import torch
from vllm import SamplingParams, LLM
from transformers import LogitsProcessor
from vllm.lora.request import LoRARequest

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
logger.propagate = False

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
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": PROMPT_TEMPLATE.format(problem_statement=problem_statement)},
    ]
    # 事前に定義したテンプレートをもとに、モデルの入力形式に合わせた文字列を作成
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not prompt.endswith("Answer:"):
        prompt += "Answer:"
    return prompt


def get_easy_probs(problem_statements: list[str], return_model: bool=False) -> np.ndarray:

    ## Initialize LLM
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if os.getenv("KAGGLE_KERNEL_RUN_TYPE") or os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        llm_model_pth: str = "/kaggle/input/m/mtfall/deepseek-r1/transformers/deepseek-r1-distill-llama-70b-awq/1"
        difficulty_lora_path: str = (
            "/kaggle/input/kprize-akami-difficulty-model/output_train-exp003-70b_003-fold0-checkpoint-100"
        )
        num_gpus: int = 4
    else:
        llm_model_pth: str = "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ"
        difficulty_lora_path: str = "/home/takuya.akiyama/work/kprize-akami/output_train/output_train-exp004-70b_003-fold0-checkpoint-100"
        num_gpus: int = torch.cuda.device_count()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    MAX_NUM_SEQS: int = 6
    MAX_MODEL_LEN: int = 32_768

    llm: LLM = LLM(
        model=llm_model_pth,
        max_num_seqs=MAX_NUM_SEQS,  # Maximum number of sequences per iteration. Default is 256
        max_model_len=MAX_MODEL_LEN,  # Model context length
        trust_remote_code=True,  # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
        tensor_parallel_size=num_gpus,  # The number of GPUs to use for distributed execution with tensor parallelism
        gpu_memory_utilization=0.95,  # The ratio (between 0 and 1) of GPU memory to reserve for the model
        enable_prefix_caching=True, 
        enable_lora=True,
        max_lora_rank=32,
        seed=2024,
    )

    tokenizer = llm.get_tokenizer()

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
    sampling_params = SamplingParams(
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
    logger.info(f"easy_probs={easy_probs}")

    if return_model:
        return [
            easy_probs, 
            {
                "llm": llm,
                "tokenizer": tokenizer,
                "sampling_params": None,
                "MAX_TOKENS": None,
                "MAX_MODEL_LEN": MAX_MODEL_LEN,
            }
        ]
    else:
        return easy_probs
