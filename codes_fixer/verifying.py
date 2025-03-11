import os
import re
import xml.etree.ElementTree as ET
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import unidiff
import torch
from vllm import RequestOutput, SamplingParams, LLM

from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import gc
import ray
import contextlib

from .config import BATCH_SIZE, MAX_NUM_SEQS, VALIDATION_COPY_COUNT
from .utils import count_tokens

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

verifying_prompt: str = """
This is the problem statement.

{problem_statement}

This is the proposed patch to fix the problem.

{patch_string}

Evaluate each item below with yes or no.
1. Completeness of modification
This patch completely fixes the issue mentioned in the problem statement.
2. Side effects
The patch does not cause any side effects or cause other tests to fail.
3. Non-interference with testing
The patch does not modify any test files.

Return the strings of evaluation result in this format

<root>
    <entry>
        <items>Completeness of modification</items>
        <label>Yes</label>
    </entry>
    <entry>
        <items>Side effects</items>
        <label>No</label>
    </entry>
    <entry>
        <items>Non-interference with testing</items>
        <label>Yes</label>
    </entry>
</root>

Reminder
- Only evaluate, do not provide suggestion on how to fix.
- Enter the exact title of the evaluation item in each item element. No other characters are allowed.
- Enter strictly yes or no for each label element. No other characters are allowed.
""".strip()


def is_valid_patch_format(patch_string: str) -> bool:
    """
    A quick check to confirm if a patch could be valid.
    """
    if not (isinstance(patch_string, str)):
        return False
    try:
        patch_set = unidiff.PatchSet(patch_string)
        if len(patch_set) == 0:
            return False
    except Exception:
        return False
    return True


def patch_dry_run_succeeds(patch_string: str, repo_path: str, timeout: int = 60) -> bool:
    """
    A robust check if the patch will proceed without any errors.
    Should be run after `is_valid_patch_format()`: the patch
    command can hang if the inputs are sufficiently invalid.

    Args:
        patch_path: Path to a file containing the patch.
        repo_path: Path to the directory to be patched.
        timeout: Number of seconds before the dry run will be cancelled.
    """
    patch_path = Path("patch.txt").resolve()
    with patch_path.open("w") as f:
        f.write(patch_string)

    cmd = f"patch --quiet --dry-run -p1 -i {str(patch_path)} -d {repo_path}"
    try:
        subprocess.run(cmd, shell=True, check=True, timeout=timeout)
        return True
    except Exception:
        return False

def extract_evaluation_results(text):
    """
    テキストから評価結果を抽出する。
    Output:
    evaluation_results: dict
        評価結果の辞書。キーはitemsの内容、値はlabelの内容。
    num_yes: int
        labelが"Yes"の数。
    """
    # テキストから<root>...</root>の部分を抽出
    match = re.search(r'<root>.*</root>', text, re.DOTALL)
    if not match:
        # raise ValueError("XML部分が見つかりません。")
        return {}, 0

    xml_content = match.group(0)

    try:
        # XMLを解析
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        # raise ValueError(f"XMLの解析中にエラーが発生しました: {e}"
        return {}, 0

    evaluation_results = {}
    num_yes = 0
    for entry in root.findall('entry'):
        item = entry.find('items')
        label = entry.find('label')
        if item is not None and label is not None:
            evaluation_results[item.text.strip()] = label.text.strip()
            if label.text.strip() == "Yes":
                num_yes += 1
        # else:
        #     raise ValueError("entry内にitemsまたはlabelが見つかりません。")

    return evaluation_results, num_yes

def choose_patch_string(
    patch_strings: list[Optional[str]], 
    judgments_aggregated: List[List[int]], 
    repo_path: str, 
) -> tuple[list[int], Optional[str]]:
    best_score = -4
    best_patch_string = None

    scores = []
    for judgments, patch_string in zip(judgments_aggregated, patch_strings):
        if patch_string is None:
            score = -3
            scores.append(score)
            continue

        if not is_valid_patch_format(patch_string):
            score = -2
            scores.append(score)
            continue

        if not patch_dry_run_succeeds(patch_string, repo_path):
            score = -1
            scores.append(score)
            continue

        score = sum(judgments)
        scores.append(score)

        if score > best_score:
            best_score = score
            best_patch_string = patch_string

    return scores, best_patch_string


def get_verification(
    problem_statement: str,
    candidate_file: List[dict],
    patch_strings: List[Optional[str]],
    repo_path: str,
    model: Optional[dict] = None,
    return_model: bool = False,
) -> Tuple[List[List[str]], List[List[bool]]]:
    if model:
        llm = model["llm"]
        tokenizer = model["tokenizer"]
        sampling_params = model["sampling_params"]
        MAX_TOKENS = model["MAX_TOKENS"]
        MAX_MODEL_LEN = model["MAX_MODEL_LEN"]
    else:
        ## Initialize LLM
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        if os.getenv("KAGGLE_KERNEL_RUN_TYPE") or os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            llm_model_pth: str = "/kaggle/input/m/mtfall/deepseek-r1/transformers/deepseek-r1-distill-llama-70b-awq/1"
            num_gpus: int = 4
        else:
            llm_model_pth: str = "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ"
            num_gpus: int = torch.cuda.device_count()

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))
        MAX_TOKENS: int = 4096
        MAX_MODEL_LEN: int = 32_768

        llm: LLM = LLM(
            model=llm_model_pth,
            max_num_seqs=MAX_NUM_SEQS,  # Maximum number of sequences per iteration. Default is 256
            max_model_len=MAX_MODEL_LEN,  # Model context length
            trust_remote_code=True,  # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
            tensor_parallel_size=num_gpus,  # The number of GPUs to use for distributed execution with tensor parallelism
            gpu_memory_utilization=0.95,  # The ratio (between 0 and 1) of GPU memory to reserve for the model
            enable_prefix_caching=True, 
            seed=2024,
        )
        tokenizer = llm.get_tokenizer()
        sampling_params: SamplingParams = SamplingParams(
            temperature=0.6,  # randomness of the sampling
            min_p=0.01,
            skip_special_tokens=True,  # Whether to skip special tokens in the output
            max_tokens=MAX_TOKENS,
        )

    inference_idx_to_input_idx: list[int] = [
        input_idx
        for _ in range(VALIDATION_COPY_COUNT)
        for input_idx, patch_string in enumerate(patch_strings)
        if patch_string is not None
        and is_valid_patch_format(patch_string) and patch_dry_run_succeeds(patch_string, repo_path)
    ]
    logger.info(f"inference_idx_to_input_idx: {inference_idx_to_input_idx}")

    list_of_messages: List[List[Dict[str, str]]] = [
        [
            {
                "role": "user",
                "content": verifying_prompt.format(
                    problem_statement=problem_statement[:20_000],
                    patch_string=patch_strings[input_idx],
                ),
            },
        ]
        for input_idx in inference_idx_to_input_idx
    ]

    prompt_texts: List[str] = [
        (
            tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)  # type: ignore
        )
        + "<think>\n"
        for messages in list_of_messages
    ]
    # logger.info(prompt_texts)

    logger.info(f"prompt_texts token length: {[count_tokens(text, tokenizer) for text in prompt_texts]}")
    request_outputs: list[RequestOutput] = llm.generate(prompt_texts, sampling_params=sampling_params)
    response_texts: List[str] = [request_output.outputs[0].text for request_output in request_outputs]
    logger.info(f"response_texts_from_inference token length : {[count_tokens(text, tokenizer) for text in response_texts]}")

    completion_texts = [prompt_text + response_text for prompt_text, response_text in zip(prompt_texts, response_texts)]
    judgments_flattened: List[dict] = [extract_evaluation_results(response_text) for response_text in response_texts]
    logger.info(f"judgments_flattened: {judgments_flattened}")

    judgments_aggregated: List[List[int]] = [[] for _ in range(BATCH_SIZE)]
    completion_text_aggregated: List[List[str]] = [[] for _ in patch_strings]
    for inference_idx, (completion_text, judgement) in enumerate(zip(completion_texts, judgments_flattened)):
        input_idx = inference_idx_to_input_idx[inference_idx]
        completion_text_aggregated[input_idx].append(completion_text)
        judgments_aggregated[input_idx].append(judgement[1])
    logger.info(f"num evaluation yes count: {judgments_aggregated}")

    # cleanup
    del llm.llm_engine.model_executor
    del llm
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()

    return completion_text_aggregated, judgments_aggregated
