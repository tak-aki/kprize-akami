# patching.py
import sys
import json
from typing import Dict, List, Optional, Tuple

from vllm import RequestOutput, SamplingParams

from .config import MAX_TOKENS, llm, tokenizer
from .utils import count_tokens, extract_and_make_patch_string

patching_prompt: str = """
In this task, you will be provided with a software development issue from a real-world GitHub repository, along with narrowed code snippets that are likely candidates for modification. Your objective is to carefully analyze and understand the issue in the context of the provided files, explain your reasoning process for addressing it, and identify the exact file paths and original code snippets that require modification. Based on this analysis, you will propose new code snippets to replace the identified ones to effectively resolve the issue.

This is the problem statement.

{problem_statement}

These are the files that is thought to be relevant

{file_content_string}

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

def get_patch_string(problem_statement: str, file_content_strings: List[str]) -> Tuple[List[str], List[Optional[str]]]:
    sampling_params: SamplingParams = SamplingParams(
        temperature=0.6,  # randomness of the sampling
        min_p=0.01,
        skip_special_tokens=True,  # Whether to skip special tokens in the output
        max_tokens=MAX_TOKENS,
    )

    inference_idx_to_input_idx: list[int] = [
        input_idx for input_idx, file_content_string in enumerate(file_content_strings) if file_content_string != ""
    ]

    list_of_messages: List[List[Dict[str, str]]] = [
        [
            {
                "role": "user",
                "content": patching_prompt.format(
                    problem_statement=problem_statement[:20_000],
                    file_content_string=file_content_strings[input_idx][:30_000],
                    example_json_str=example_json_str,
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
    # print(prompt_texts)

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

    completion_texts: list[str] = ["" for _ in file_content_strings]
    patch_strings: List[Optional[str]] = [None for _ in file_content_strings]
    for inference_idx, (completion_text, patch_string) in enumerate(
        zip(completion_texts_from_inference, patch_strings_from_inference)
    ):
        input_idx = inference_idx_to_input_idx[inference_idx]
        completion_texts[input_idx] = completion_text
        patch_strings[input_idx] = patch_string

    return completion_texts, patch_strings
