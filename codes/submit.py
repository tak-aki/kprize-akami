import io
import os
import time
from typing import List

import pandas as pd

start_time = time.time()

from .config import REPO_PATH, VALIDATION_COPY_COUNT, tokenizer
from .fetch_file import fetch_file_contents
from .patching import get_patch_string
from .selection_query import get_selection_query
from .utils import count_tokens, stringify_directory
from .verifying import choose_patch_string, get_verification


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
                count_tokens(completion_text, tokenizer) for completion_text in selection_completion_texts
            ],
            "file_query": file_queries,
            "file_content_string": file_content_strings,
            "patch_completion_text": patch_completion_texts,
            "patch_completion_length": [
                count_tokens(completion_text, tokenizer) for completion_text in patch_completion_texts
            ],
            "patch_string": patch_strings,
        }

        for copy_idx in range(VALIDATION_COPY_COUNT):
            data[f"verification_completion_text_{copy_idx}"] = [
                completion_texts[copy_idx] if completion_texts else None
                for completion_texts in verification_completion_texts_aggregated
            ]
            data[f"verification_completion_length_{copy_idx}"] = [
                count_tokens(completion_texts[copy_idx], tokenizer) if completion_texts else None
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
