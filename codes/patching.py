# patching.py
from typing import Dict, List, Optional, Tuple

from vllm import RequestOutput, SamplingParams

from .config import MAX_TOKENS, llm, tokenizer
from .utils import count_tokens, extract_patch_string

patching_prompt: str = """
You will be implementing a git diff patch to solve an issue with the code repository.
This is the problem statement.

{problem_statement}

These are the files that is thought to be relevant

{file_content_string}

Write a git diff within ```diff and ``` that fully fixes the problem.
The git diff should not cause other tests to fail.

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
- Only the last diff printed will be considered.
""".strip()


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
        extract_patch_string(response_text) for response_text in response_texts_from_inference
    ]

    completion_texts: list[str] = ["" for _ in file_content_strings]
    patch_strings: List[Optional[str]] = [None for _ in file_content_strings]
    for inference_idx, (completion_text, patch_string) in enumerate(
        zip(completion_texts_from_inference, patch_strings_from_inference)
    ):
        input_idx = inference_idx_to_input_idx[inference_idx]
        completion_texts[input_idx] = completion_text
        patch_strings[input_idx] = patch_string

    return completion_texts, patch_strings
