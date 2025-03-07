# patching.py
from typing import Dict, List, Optional, Tuple

from vllm import RequestOutput, SamplingParams

from .config import BATCH_SIZE, MAX_TOKENS, llm, tokenizer
from .utils import count_tokens, extract_patch_string

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


def get_patch_string(problem_statement: str, candidate_file: List[dict], directory_string: str) -> Tuple[List[str], List[Optional[str]]]:
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
                    file_path=candidate_file[0]["file_path"][:1000],
                    file_content=candidate_file[0]["file_content"][:100_000],
                ),
            },
        ]
        for _ in range(BATCH_SIZE)
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

    completion_texts: list[str] = ["" for _ in range(BATCH_SIZE)]
    patch_strings: List[Optional[str]] = [None for _ in range(BATCH_SIZE)]
    for input_idx, (completion_text, patch_string) in enumerate(
        zip(completion_texts_from_inference, patch_strings_from_inference)
    ):
        completion_texts[input_idx] = completion_text
        patch_strings[input_idx] = patch_string

    return completion_texts, patch_strings
