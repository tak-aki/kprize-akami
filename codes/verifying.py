import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import py_compile

import unidiff
from vllm import RequestOutput, SamplingParams

from .config import MAX_TOKENS, REPO_PATH, VALIDATION_COPY_COUNT, llm, tokenizer
from .utils import count_tokens

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

def find_patched_files(patch:str) -> List[str]:
    patched_files = {
        "added": [],  # 追加されたファイル
        "removed": [],  # 削除されたファイル
        "modified": [],  # 編集されたファイル
    }
    try: 
        patch_set = unidiff.PatchSet(patch)

        # ファイルごとに変更内容をチェック(addedは不要)
        for patched_file in patch_set:
            if patched_file.is_modified_file:
                patched_files["modified"].append(patched_file.path)
            if patched_file.is_removed_file:
                patched_files["removed"].append(patched_file.path)
            if patched_file.is_added_file:
                patched_files["added"].append(patched_file.path)

        return patched_files
    except Exception as e:
        print(f"Failed to find gold files: {e}")
        return patched_files

def check_syntax(file_path: str):
    try:
        py_compile.compile(file_path, doraise=True)
        print(f"Syntax OK: {file_path}")
        return True
    except py_compile.PyCompileError as e:
        print(f"Syntax Error in {file_path}: {e}")
        return False

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

    dry_cmd = f"patch --quiet --dry-run -p1 -i {str(patch_path)} -d {repo_path}"
    try:
        subprocess.run(dry_cmd, shell=True, check=True, timeout=timeout)
        print("Dry run succeeded")

        patched_files = find_patched_files(patch_string)
        apply_cmd = f"patch --quiet -p1 -i {str(patch_path)} -d {repo_path}"
        subprocess.run(apply_cmd, shell=True, check=True, timeout=timeout)
        print("Patch Applied")

        syntax_results = []
        for patched_file in patched_files["modified"] + patched_files["added"]:
            syntax_results.append(check_syntax(f"{repo_path}/{patched_file}"))

        reverse_cmd = f"patch --quiet -R -p1 -i {str(patch_path)} -d {repo_path}"
        subprocess.run(reverse_cmd, shell=True, check=True, timeout=timeout)
        print("Patch Reversed")
        
        if all(syntax_results):
            print("All Patched File Syntax OK")
            return True
        else:
            print("Some Patched File Syntax Error")
            return False

    except Exception as e:
        print(f"Error has occurred in checking the patch: {e}")
        return False


def choose_patch_string(
    patch_strings: list[Optional[str]], judgments_aggregated: List[List[bool]], repo_path: str
) -> tuple[list[int], Optional[str]]:
    best_score = 0
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
    print("choose_patch_string score:", scores)

    return scores, best_patch_string


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
        and is_valid_patch_format(patch_string) and patch_dry_run_succeeds(patch_string, repo_path)
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

    print("get_verification", [count_tokens(text, tokenizer) for text in prompt_texts])
    request_outputs: list[RequestOutput] = llm.generate(prompt_texts, sampling_params=sampling_params)
    response_texts: List[str] = [request_output.outputs[0].text for request_output in request_outputs]
    print("get_verification", [count_tokens(text, tokenizer) for text in response_texts])

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
