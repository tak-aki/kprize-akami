# patching.py
import os
from typing import Dict, List, Optional, Tuple
import torch

from vllm import RequestOutput, SamplingParams, LLM

from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import gc
import ray
import contextlib

from .config import BATCH_SIZE, MAX_NUM_SEQS
from .utils import count_tokens, extract_patch_string, load_file_content

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

patching_prompt: str = """
In this task, you will be provided with a software development issue from a real-world GitHub repository, along with the content of retrieved code files for modification. 
Your objective is to carefully analyze and understand the issue in the context of the provided files, and create a patch that completely resolves the problem if given files needs modification. 

<problem_statement>
{problem_statement}
</problem_statement>

This is the file that is thought to be relevant

<file_path>
{file_path}
</file_path>

<file_content>
{file_content}
</file_content>

You will break down the task into two steps. 

First, analyze the problem_statement and retrieved file to identify which part of the codes is the cause of the issue. 
When doing so, focus only on the functional parts and do not consider minor details such as the accuracy of comments in the code. 
If, after this analysis, you determine that the retrieved file is irrelevant to resolving the issue or is insufficient, stop the task.

Second, based on the cause of the issue, output a diff format patch for modification. 
The diff patch should be output based on the example below. 
The diff patch must be necessary and sufficient to resolve the problem, and does not affect any functionality other than the problem.
When creating the diff, if you determine that the retrieved file is irrelevant to resolving the issue or is insufficient, stop the task without creating a diff.

Example:

```diff
--- a/first.txt
+++ b/first.txt
@@ -1,3 +1,3 @@
 start
-first change
+new first change
 middle
@@ -7,4 +7,4 @@
 some content
-second change
+new second change
 more content
--- a/second.txt
+++ b/second.txt
@@ -1,3 +1,3 @@
 beginning
-old line
+new line
 end
```

Reminder
- Put your diff within ```diff and ``` and make sure the diff is valid.
- A diff must be created at most once.
- If the given file is irrelevant or insufficient to resolve the issue, do not create patch-format output at all and say it is not possible.
""".strip()


def get_patch_string(
        problem_statement: str, 
        codebase_path: str, 
        candidate_file_list: List[List[str]], 
        directory_string: str, 
        model: Optional[dict] = None,
        return_model: bool = False, 
        ) -> Tuple[List[str], List[Optional[str]]]:
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

    list_of_messages = [
        [
            {
                "role": "user",
                "content": patching_prompt.format(
                    problem_statement=problem_statement[:20_000],
                    # directory_string=directory_string[:30_000],
                    file_path=candidate_files[0],
                    file_content=load_file_content(codebase_path, candidate_files[0], with_line_numbers=True),
                ),
            },
        ]
        for candidate_files in candidate_file_list
    ]

    prompt_texts: List[str] = [
        (
            tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)  # type: ignore
        )
        for messages in list_of_messages
    ]
    # logger.info(prompt_texts)
    logger.info(f"prompt_texts token length: {[count_tokens(text, tokenizer) for text in prompt_texts]}")

    inference_idx_to_input_idx: list[int] = [
        input_idx
        for input_idx, text in enumerate(prompt_texts)
        if count_tokens(text, tokenizer) < (MAX_MODEL_LEN - 100) # 100 is a buffer
    ]
    prompt_texts = [prompt_texts[input_idx] for input_idx in inference_idx_to_input_idx]
    logger.info(f"inference_idx_to_input_idx: {inference_idx_to_input_idx}]")

    request_outputs: list[RequestOutput] = llm.generate(prompt_texts, sampling_params=sampling_params)
    response_texts_from_inference: List[str] = [request_output.outputs[0].text for request_output in request_outputs]
    logger.info(f"response_texts_from_inference token length : {[count_tokens(text, tokenizer) for text in response_texts_from_inference]}")
    completion_texts_from_inference = [
        prompt_text + response_text for prompt_text, response_text in zip(prompt_texts, response_texts_from_inference)
    ]
    patch_strings_from_inference: List[Optional[str]] = [
        extract_patch_string(response_text) for response_text in response_texts_from_inference
    ]
    logger.info(f"is diff format output : {[s is not None for s in patch_strings_from_inference]}")

    completion_texts: list[str] = ["" for _ in range(BATCH_SIZE)]
    patch_strings: List[Optional[str]] = [None for _ in range(BATCH_SIZE)]
    for inference_idx, (completion_text, patch_string) in enumerate(
        zip(completion_texts_from_inference, patch_strings_from_inference)
    ):
        input_idx = inference_idx_to_input_idx[inference_idx]
        completion_texts[input_idx] = completion_text
        patch_strings[input_idx] = patch_string

    if return_model:
        return [
            completion_texts, 
            patch_strings, 
            {
                "llm": llm,
                "tokenizer": tokenizer,
                "sampling_params": sampling_params,
                "MAX_TOKENS": MAX_TOKENS,
                "MAX_MODEL_LEN": MAX_MODEL_LEN,
            }
        ]
    else:
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
        return completion_texts, patch_strings
