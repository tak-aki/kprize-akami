# patching.py
import sys
import json
from typing import Dict, List, Optional, Tuple
import gc
import contextlib

from config import llm_model_path, BATCH_SIZE, MAX_NUM_SEQS, num_gpus
from utils import count_tokens, extract_and_make_patch_string, load_file_content

patching_prompt: str = """
In this task, you will be provided with a software development issue from a real-world GitHub repository, along with the full content of retrieved code files for modification. Your objective is to carefully analyze and understand the issue in the context of the provided files, explain your reasoning process for addressing it, and identify the exact file paths and original code snippets that require modification. Based on this analysis, you will propose new code snippets to replace the identified ones to effectively resolve the issue.

This is the problem statement.

{problem_statement}

This is the file that is thought to be relevant

<file_path>
{file_path}
</file_path>

<file_content>
{file_content}
</file_content>

Write a modification proposal output as a conclusion of your analysis.
The modification should not cause other tests to fail.
The output must strictly follow the format below:

<modification>
{{
    "type": "object",
    "properties": {{
        "edited code": {{
            "type": "array",
            "items": {{
                "type": "object",
                "properties": {{
                    "file": {{
                        "type": "string",
                    }},
                    "code snippet to be modified": {{
                        "type": "string",
                    }},
                    "edited code snippet": {{
                        "type": "string",
                    }},
                }},
            }},
        }},
    }},
}}
</modification>

Points of format:
- The "edited code" field should contain an array of objects.
- The "code snippet to be modified" field must be the source code before modification including the line number.
- The "code snippet to be modified" field must include not only the lines to be modified but also a buffer of several surrounding lines, even if they do not need to be fixed.
- The "edited code snippet" field must be the modified code and must not contain line numbers.
- The code part listed in The "edited code snippet" field must match the code including the buffer  listed in The "code snippet to be modified" field

Example:

<modification>
{example_json_str}
</modification>

Reminder
- Put your modification within <modification> and </modification> strictly and also follow the format and the points of format.
""".strip()

example_dict = {
    "edited code": [
        {
            "file": "src/flask/blueprints.py",
            "code snippet to be modified": """188     template_folder=template_folder,
189     root_path=root_path,
190     )
191     self.name = name
192     self.url_prefix = url_prefix
193     self.subdomain = subdomain""",
            "edited code snippet": """    template_folder=template_folder,
    root_path=root_path,
    )

    if "." in name:
        raise ValueError("'name' may not contain a dot '.' character.")

    self.name = name
    self.url_prefix = url_prefix
    self.subdomain = subdomain"""
        },
    ]
}
example_json_str = json.dumps(example_dict, indent=4)

def get_patch_string(
        problem_statement: str, 
        codebase_path: str, 
        candidate_file_list: List[List[str]], 
        model: Optional[dict] = None,
        return_model: bool = False, 
        ) -> Tuple[List[str], List[Optional[str]]]:
    from vllm import RequestOutput, SamplingParams, LLM
    import torch
    import ray
    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        destroy_distributed_environment,
    )

    if model:
        llm = model["llm"]
        tokenizer = model["tokenizer"]
        MAX_MODEL_LEN = model["MAX_MODEL_LEN"]
    else:
        ## Initialize LLM
        MAX_MODEL_LEN: int = 32_768
        llm: LLM = LLM(
            model=llm_model_path,
            max_num_seqs=MAX_NUM_SEQS,  # Maximum number of sequences per iteration. Default is 256
            max_model_len=MAX_MODEL_LEN,  # Model context length
            trust_remote_code=True,  # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
            tensor_parallel_size=num_gpus,  # The number of GPUs to use for distributed execution with tensor parallelism
            gpu_memory_utilization=0.95,  # The ratio (between 0 and 1) of GPU memory to reserve for the model
            enable_prefix_caching=True, 
            seed=2024,
        )
        tokenizer = llm.get_tokenizer()

    MAX_TOKENS: int = 4096
    sampling_params: SamplingParams = SamplingParams(
        temperature=0.6,  # randomness of the sampling
        min_p=0.01,
        skip_special_tokens=True,  # Whether to skip special tokens in the output
        max_tokens=MAX_TOKENS,
    )

    list_of_messages: List[List[Dict[str, str]]] = [
        [
            {
                "role": "user",
                "content": patching_prompt.format(
                    problem_statement=problem_statement[:20_000],
                    file_path=candidate_files[0],
                    file_content=load_file_content(codebase_path, candidate_files[0], with_line_numbers=True),
                    example_json_str=example_json_str,
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
    print(f"prompt_texts token length: {[count_tokens(text, tokenizer) for text in prompt_texts]}")

    inference_idx_to_input_idx: list[int] = [
        input_idx
        for input_idx, text in enumerate(prompt_texts)
        if count_tokens(text, tokenizer) < (MAX_MODEL_LEN - 100) # 100 is a buffer
    ]
    prompt_texts = [prompt_texts[input_idx] for input_idx in inference_idx_to_input_idx]
    print(f"inference_idx_to_input_idx: {inference_idx_to_input_idx}]")

    print("get_patch_string", [count_tokens(text, tokenizer) for text in prompt_texts])
    request_outputs: list[RequestOutput] = llm.generate(prompt_texts, sampling_params=sampling_params)
    response_texts_from_inference: List[str] = [request_output.outputs[0].text for request_output in request_outputs]
    print(
        "get_patch_string",
        [count_tokens(text, tokenizer) for text in response_texts_from_inference],
    )
    completion_texts_from_inference = [
        prompt_text + response_text for prompt_text, response_text in zip(prompt_texts, response_texts_from_inference)
    ]
    patch_strings_from_inference: List[Optional[str]] = [
        extract_and_make_patch_string(response_text) for response_text in response_texts_from_inference
    ]
    print(f"output diff format: {[s is not None for s in patch_strings_from_inference]}")

    completion_texts: list[str] = ["" for _ in range(BATCH_SIZE)]
    patch_strings: List[Optional[str]] = [None for _ in range(BATCH_SIZE)]
    for inference_idx, (completion_text, patch_string) in enumerate(
        zip(completion_texts_from_inference, patch_strings_from_inference)
    ):
        input_idx = inference_idx_to_input_idx[inference_idx]
        completion_texts[input_idx] = completion_text
        patch_strings[input_idx] = patch_string

    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm
    # with contextlib.suppress(AssertionError):
    #     torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()

    return completion_texts, patch_strings
