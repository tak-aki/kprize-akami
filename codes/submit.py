import io
import os
import re
import subprocess
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import unidiff

# https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/560682#3113134
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"

start_time = time.time()


from vllm import LLM, RequestOutput, SamplingParams

warnings.simplefilter("ignore")


## Initialize LLM
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if os.getenv("KAGGLE_KERNEL_RUN_TYPE") or os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    llm_model_pth: str = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-32b-awq/1"
    num_gpus: int = 4
else:
    llm_model_pth: str = "inarikami/DeepSeek-R1-Distill-Qwen-32B-AWQ"
    num_gpus: int = torch.cuda.device_count()

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))

BATCH_SIZE: int = 6
VALIDATION_COPY_COUNT: int = 1
MAX_TOKENS: int = 4096

MAX_NUM_SEQS: int = 6
MAX_MODEL_LEN: int = 32_768

llm: LLM = LLM(
    model=llm_model_pth,
    max_num_seqs=MAX_NUM_SEQS,  # Maximum number of sequences per iteration. Default is 256
    max_model_len=MAX_MODEL_LEN,  # Model context length
    trust_remote_code=True,  # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
    tensor_parallel_size=num_gpus,  # The number of GPUs to use for distributed execution with tensor parallelism
    gpu_memory_utilization=0.95,  # The ratio (between 0 and 1) of GPU memory to reserve for the model
    seed=2024,
)

tokenizer = llm.get_tokenizer()


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


## Helper functions
def stringify_directory(directory: str) -> str:
    full_paths: List[str] = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path: str = os.path.join(root, file)
            full_paths.append(full_path)
    return "\n".join(full_paths)


def extract_file_query(xml_content: str) -> Dict[str, List[str]]:
    import xml.etree.ElementTree as ET

    # Prepare a data structure to collect results
    parsed_data: Dict[str, List[str]] = {}
    pattern: str = r"<root>(.*?)</root>"
    matches: List[str] = re.findall(pattern, xml_content, re.DOTALL)

    for match in matches:
        try:
            # Parse the XML
            root = ET.fromstring("<root>" + match + "</root>")

            # Find all <entry> elements
            for entry in root.findall("entry"):
                # Extract the <filepath> text
                filepath = entry.find("filepath")
                filepath_text: Optional[str] = (
                    filepath.text.strip() if filepath is not None and filepath.text is not None else None
                )

                # Locate <strings_to_search> container
                strings_container = entry.find("strings_to_search")

                # Gather each <string_to_search> text
                search_strings: List[str] = []
                if strings_container is not None:
                    for s in strings_container.findall("string_to_search"):
                        if s.text is not None:
                            search_strings.append(s.text.strip())

                # Store in a dictionary: { filepath: [search_strings...] }
                parsed_data[filepath_text] = search_strings  # type: ignore
        except:
            print("Error parsing output")
            print(xml_content)
            return {}

    return parsed_data


reading_prompt: str = """
You will be implementing a git diff patch to solve an issue with the code repository.
You will first need to select files in the file directory.

This is the problem statement.

{problem_statement}

This is the file directory

<directory>
{directory_string}
</directory>

Which files should be inspected so that we can solve the problem?
When we inspect each file, what strings should be searched?

Return the strings to search in this format

(explanation)

<root>
    <entry>
        <filepath>filepath</filepath>  
        <strings_to_search>
            <string_to_search>string_to_search</string_to_search>
            ...
            <string_to_search>string_to_search</string_to_search>
        </strings_to_search>
    </entry>
    <entry>
        <filepath>filepath</filepath>
        <strings_to_search>
            <string_to_search>string_to_search</string_to_search>
            ...
            <string_to_search>string_to_search</string_to_search>
        </strings_to_search>
    </entry>
    ...
</root>
...

Notes:
- Make sure to encode each entry between <root> and </root>
- Return the FULL filepath - exactly as specified in <directory> and </directory>
    - Example: <filepath>repo/path/to/directory/file.py</filepath>
- If you are searching for a word instead of a substring, maybe add spaces or brackets before and after the string
    - For example, if you are searching for uses of the function `calculate`, use ` calculate(` as the search string instead of `calculate`
- Prefer searching longer strings
    - Avoid searching for strings that might appear in many parts of the codebase
- Search the test files as well to understand the feature behavior
    - Also search for the relevant function calls in the test files
""".strip()


def get_selection_query(directory_string: str, problem_statement: str) -> Tuple[List[str], List[Dict[str, List[str]]]]:
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
                "content": reading_prompt.format(
                    problem_statement=problem_statement[:20_000],
                    directory_string=directory_string[:30_000],
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

    print("get_selection_query", [count_tokens(text) for text in prompt_texts])
    request_outputs: list[RequestOutput] = llm.generate(prompt_texts, sampling_params=sampling_params)
    if not request_outputs:
        return [], []
    response_texts: List[str] = [request_output.outputs[0].text for request_output in request_outputs]
    print("get_selection_query", [count_tokens(text) for text in response_texts])

    completion_texts = [prompt_text + response_text for prompt_text, response_text in zip(prompt_texts, response_texts)]
    file_queries: List[Dict[str, List[str]]] = [extract_file_query(response_text) for response_text in response_texts]
    return completion_texts, file_queries


REPO_PATH: str = "repo"


def fetch_file_contents(files_to_search: Dict[str, List[str]], context_lines: int = 12, max_gap: int = 0) -> str:
    from io import StringIO
    from typing import Tuple

    def find_lines_in_files_with_context(
        search_map: Dict[str, List[str]], context_lines: int = context_lines
    ) -> List[List[List[Tuple[int, str]]]]:
        """
        Given a dictionary mapping file paths to a list of search terms,
        open each file and gather *snippets* of lines that contain any
        of those search terms, including 'context_lines' before and after.

        Returns a list of lists:
        [
            [  # For file1
                [ (line_number, text), (line_number, text), ... ],
                [ ... ],
            ],
            [  # For file2
                ...
            ],
            ...
        ]
        """
        all_matches_per_file: List[List[List[Tuple[int, str]]]] = []

        for path, terms in search_map.items():
            if not os.path.isfile(path):
                # If the file is not found, record an empty list
                all_matches_per_file.append([])
                continue

            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            file_snippets: List[List[Tuple[int, str]]] = []
            num_lines: int = len(lines)

            for i, line in enumerate(lines, start=1):
                if any(t in line for t in terms):
                    start_idx: int = max(1, i - context_lines)
                    end_idx: int = min(num_lines, i + context_lines)
                    snippet: List[Tuple[int, str]] = []
                    for snippet_no in range(start_idx, end_idx + 1):
                        text_content: str = lines[snippet_no - 1].rstrip("\n")
                        snippet.append((snippet_no, text_content))
                    file_snippets.append(snippet)

            all_matches_per_file.append(file_snippets)

        return all_matches_per_file

    # ---------------------------------------------------------
    # 3. MERGE OVERLAPPING/ADJACENT SNIPPETS
    # ---------------------------------------------------------

    def merge_file_snippets(file_snippets: List[List[Tuple[int, str]]], gap: int = 0) -> List[List[Tuple[int, str]]]:
        """
        Merge overlapping or nearly adjacent snippets in a single file’s snippet list.
        """
        intervals: List[Tuple[int, int, List[Tuple[int, str]]]] = []
        for snippet in file_snippets:
            if snippet:
                start_line: int = snippet[0][0]
                end_line: int = snippet[-1][0]
                intervals.append((start_line, end_line, snippet))

        intervals.sort(key=lambda x: x[0])  # sort by start line

        merged: List[Tuple[int, int, List[Tuple[int, str]]]] = []
        for start, end, snippet in intervals:
            if not merged:
                merged.append((start, end, snippet))
                continue

            prev_start, prev_end, prev_snippet = merged[-1]
            if start <= prev_end + gap:
                new_end: int = max(end, prev_end)
                combined_dict: Dict[int, str] = {}
                for ln, txt in prev_snippet:
                    combined_dict[ln] = txt
                for ln, txt in snippet:
                    combined_dict[ln] = txt
                merged_snippet: List[Tuple[int, str]] = [(ln, combined_dict[ln]) for ln in sorted(combined_dict)]
                merged[-1] = (prev_start, new_end, merged_snippet)
            else:
                merged.append((start, end, snippet))

        # Extract just the merged snippet portion
        return [x[2] for x in merged]

    def merge_all_snippets(
        all_files_snips: List[List[List[Tuple[int, str]]]], gap: int = 0
    ) -> List[List[List[Tuple[int, str]]]]:
        """
        Merge snippet blocks within each file.
        all_files_snips is a list-of-lists:
          [
            [ snippetA, snippetB, ... ],  # file 1
            [ snippetC, snippetD, ... ],  # file 2
          ]
        """
        merged: List[List[List[Tuple[int, str]]]] = []
        for snips in all_files_snips:
            merged.append(merge_file_snippets(snips, gap=gap))
        return merged

    # ---------------------------------------------------------
    # 4. RUN LOGIC: generate files, search, merge, and BUILD A STRING
    # ---------------------------------------------------------

    has_any_matches: bool = False

    # 1) Gather snippets around each match
    context_snippets: List[List[List[Tuple[int, str]]]] = find_lines_in_files_with_context(
        files_to_search, context_lines=context_lines
    )

    # 2) Merge overlapping snippets
    merged_snips: List[List[List[Tuple[int, str]]]] = merge_all_snippets(context_snippets, gap=max_gap)

    # 3) Build a string (instead of printing)
    output = StringIO()

    # Header
    output.write("Sample files created successfully.\n\n")
    output.write("Search Results (by file, merging any overlapping context):\n\n")

    # For each file
    for (filepath, terms), snippet_list in zip(files_to_search.items(), merged_snips):
        output.write(f"[file name]: {filepath[len(REPO_PATH) + 1 :]}\n")
        terms_searched_as_str = "\n".join(terms)
        output.write(f"[terms searched]:\n{terms_searched_as_str}\n")
        output.write("[file content begin]\n")
        if not snippet_list:
            output.write("  No matches found.\n")
        else:
            has_any_matches = True
            for snippet_idx, snippet in enumerate(snippet_list, start=1):
                snippet_start: int = snippet[0][0]
                snippet_end: int = snippet[-1][0]
                output.write(f"\nMatch #{snippet_idx}, lines {snippet_start} to {snippet_end}:\n")
                for line_no, text in snippet:
                    output.write(f"  {line_no:3d} | {text}\n")
                output.write("\n")
        output.write("[file content end]\n\n")

    file_content_string: str = output.getvalue()

    if has_any_matches:
        return file_content_string
    return ""


def extract_patch_string(text: str) -> Optional[str]:
    pattern: str = r"\n```diff\n(.*?)\n```"
    matches: List[str] = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    return matches[-1] + "\n"


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

    print("get_patch_string", [count_tokens(text) for text in prompt_texts])
    request_outputs: list[RequestOutput] = llm.generate(prompt_texts, sampling_params=sampling_params)
    response_texts_from_inference: List[str] = [request_output.outputs[0].text for request_output in request_outputs]
    print(
        "get_patch_string",
        [count_tokens(text) for text in response_texts_from_inference],
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


verifying_prompt: str = """
This is the problem statement.

{problem_statement}

These are the files that is thought to be relevant, which may not be complete.

{file_content_string}

This is the proposed patch to fix the problem.

{patch_string}

Evaluate whether the patch works
- The patch fully fixes the problem described in the problem statement.
- The patch does not cause side effects and make any other tests fail.

End your response with exactly either of
- <label>Yes</label>, this fixes the problem.
- <label>No</label>, this does not fix the problem.

Reminder
- Only evaluate, do not provide suggestion on how to fix.
- Remember to write exactly either of <label>Yes</label> or <label>No</label> in the last line
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


def patch_dry_run_succeeds(patch_string: str, repo_path: str = REPO_PATH, timeout: int = 60) -> bool:
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


def get_verification(
    problem_statement: str,
    file_content_strings: List[str],
    patch_strings: List[Optional[str]],
    repo_path: str,
) -> Tuple[List[List[str]], List[List[bool]]]:
    assert len(file_content_strings) == len(patch_strings)
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
        and is_valid_patch_format(patch_string)  # and patch_dry_run_succeeds(patch_string, repo_path)
    ]
    print(inference_idx_to_input_idx)

    list_of_messages: List[List[Dict[str, str]]] = [
        [
            {
                "role": "user",
                "content": verifying_prompt.format(
                    problem_statement=problem_statement[:20_000],
                    file_content_string=file_content_strings[input_idx][:30_000],
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
    # print(prompt_texts)

    print("get_verification", [count_tokens(text) for text in prompt_texts])
    request_outputs: list[RequestOutput] = llm.generate(prompt_texts, sampling_params=sampling_params)
    response_texts: List[str] = [request_output.outputs[0].text for request_output in request_outputs]
    print("get_verification", [count_tokens(text) for text in response_texts])

    completion_texts = [prompt_text + response_text for prompt_text, response_text in zip(prompt_texts, response_texts)]
    judgments_flattened: List[bool] = ["<label>Yes</label>" in response_text for response_text in response_texts]
    print(judgments_flattened)

    judgments_aggregated: List[List[bool]] = [[] for _ in file_content_strings]
    completion_text_aggregated: List[List[str]] = [[] for _ in patch_strings]
    for inference_idx, (completion_text, judgement) in enumerate(zip(completion_texts, judgments_flattened)):
        input_idx = inference_idx_to_input_idx[inference_idx]
        completion_text_aggregated[input_idx].append(completion_text)
        judgments_aggregated[input_idx].append(judgement)
    print(judgments_aggregated)

    return completion_text_aggregated, judgments_aggregated


def choose_patch_string(
    patch_strings: list[Optional[str]], judgments_aggregated: List[List[bool]], repo_path: str
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

        score = judgments.count(True)
        scores.append(score)

        if score > best_score:
            best_score = score
            best_patch_string = patch_string

    return scores, best_patch_string


def predict_inner(
    problem_statement: str,
    repo_archive: io.BytesIO,
    pip_packages_archive: io.BytesIO,
    env_setup_cmds_templates: list[str],
    skip_prediction: bool = False,
    save_result: bool = True,
) -> str:
    """
    Args:
        problem_statement: The text of the git issue.
        repo_path: A BytesIO buffer path with a .tar containing the codebase that must be patched. The gateway will make this directory available immediately before this function runs.
        pip_packages_archive: A BytesIO buffer path with a .tar containing the wheel files necessary for running unit tests.
        env_setup_cmds_templates: Commands necessary for installing the pip_packages_archive.
    """
    if skip_prediction:
        return None

    directory: str = REPO_PATH

    directory_string = stringify_directory(directory)

    selection_completion_texts, file_queries = get_selection_query(directory_string, problem_statement)

    file_content_strings: List[str] = [fetch_file_contents(file_query) for file_query in file_queries]

    patch_completion_texts, patch_strings = get_patch_string(problem_statement, file_content_strings)

    verification_completion_texts_aggregated, judgments_aggregated = get_verification(
        problem_statement, file_content_strings, patch_strings, directory
    )

    scores, patch_string = choose_patch_string(patch_strings, judgments_aggregated, directory)

    if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        data = {
            "problem_statement": [problem_statement] * len(file_queries),
            "selection_completion_text": selection_completion_texts,
            "selection_completion_length": [
                count_tokens(completion_text) for completion_text in selection_completion_texts
            ],
            "file_query": file_queries,
            "file_content_string": file_content_strings,
            "patch_completion_text": patch_completion_texts,
            "patch_completion_length": [count_tokens(completion_text) for completion_text in patch_completion_texts],
            "patch_string": patch_strings,
        }

        for copy_idx in range(VALIDATION_COPY_COUNT):
            data[f"verification_completion_text_{copy_idx}"] = [
                completion_texts[copy_idx] if completion_texts else None
                for completion_texts in verification_completion_texts_aggregated
            ]
            data[f"verification_completion_length_{copy_idx}"] = [
                count_tokens(completion_texts[copy_idx]) if completion_texts else None
                for completion_texts in verification_completion_texts_aggregated
            ]
            data[f"judgment_{copy_idx}"] = [
                judgments[copy_idx] if judgments else None for judgments in judgments_aggregated
            ]

        data["judgment_count_true"] = [judgments.count(True) for judgments in judgments_aggregated]
        data["score"] = scores

        if save_result:
            pd.DataFrame(data).to_csv(f"{str(int(time.time() - start_time)).zfill(5)}.csv", index=False)

    print("submitted patch_string")
    print(patch_string)

    if patch_string is None:
        return None

    return patch_string
