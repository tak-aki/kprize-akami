import io
import os
import time
from pathlib import Path
from typing import List

import pandas as pd

start_time = time.time()

from .config import BATCH_SIZE, REPO_PATH, VALIDATION_COPY_COUNT, tokenizer
from .bm25 import get_bm25_top_files
from .patching import get_patch_string
from .utils import count_tokens, stringify_directory
from .verifying import choose_patch_string, get_verification


def predict_inner(
    problem_statement: str,
    repo_archive: io.BytesIO,
    pip_packages_archive: io.BytesIO,
    env_setup_cmds_templates: list[str],
    skip_prediction: bool = False,
    save_result: bool = True,
    max_file_lines: int = 10,
    output_dir: str | None = None,
) -> str:
    """
    Args:
        problem_statement: The text of the git issue.
        repo_path: A BytesIO buffer path with a .tar containing the codebase that must be patched.
            The gateway will make this directory available immediately before this function runs.
        pip_packages_archive: A BytesIO buffer path with a .tar containing the wheel files necessary for running unit tests.
        env_setup_cmds_templates: Commands necessary for installing the pip_packages_archive.
    """
    if skip_prediction:
        return None

    directory: str = REPO_PATH

    bm25_top_files = get_bm25_top_files(problem_statement, directory, top_k=1)

    directory_string = stringify_directory(directory)
    patch_completion_texts, patch_strings = get_patch_string(problem_statement, bm25_top_files, directory_string)
    verification_completion_texts_aggregated, judgments_aggregated = get_verification(
        problem_statement, bm25_top_files, patch_strings, directory
    )

    scores, patch_string = choose_patch_string(patch_strings, judgments_aggregated, directory)

    if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        data = {
            "problem_statement": [problem_statement] * BATCH_SIZE,
            "bm25_top_files": bm25_top_files * BATCH_SIZE,
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
        data["elapsed_time"] = time.time() - start_time

        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(data).to_csv(Path(output_dir) / "predictions.csv", index=False)

    print("submitted patch_string")
    print(patch_string)

    if patch_string is None:
        return None

    return patch_string
