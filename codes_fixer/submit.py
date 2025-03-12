import io
import json
import os
import argparse
import time
import gc
from pathlib import Path
from typing import List
import torch

import pandas as pd

start_time = time.time()

from config import BATCH_SIZE, VALIDATION_COPY_COUNT
from difficulty import get_easy_probs
from llm_selection import get_llm_selection
from bm25 import get_bm25_top_files
from llm_retrieve import get_llm_retrieval
from patching import get_patch_string
from utils import count_tokens, stringify_directory
from verifying import choose_patch_string, get_verification

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

def predict_inner(
    problem_statement: str,
    patch_filepath: str,
    # repo_archive: io.BytesIO,
    # pip_packages_archive: io.BytesIO,
    # env_setup_cmds_templates: list[str],
    skip_prediction: bool = False,
    # save_result: bool = True,
    difficulty_threshold: float = 0.5,
    # max_file_lines: int = 10,
    output_dir: str | None = None,
    directory: str = "repo",
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
        print("Skipping prediction")
        return None
    print("Start running the inner predictor.")
    
    easy_probs = get_easy_probs([problem_statement])
    easy_prob = easy_probs[0]
    if easy_prob < difficulty_threshold:
        print(f"Skipping prediction because the problem is too difficult (easy_prob={easy_prob:.2f})")
        return None

    directory_string = stringify_directory(directory)

    selection_completion_texts, llm_selected_files = get_llm_selection(directory_string, problem_statement)

    bm25_top_files = get_bm25_top_files(problem_statement, directory, top_k=30)

    concat_files = [sf + [bf for bf in bm25_top_files if bf not in sf] for sf in llm_selected_files] # llm selectionにあるファイルはbm25から除去しつつ結合

    llm_retrieval_completion_texts, llm_retrieved_files = get_llm_retrieval(problem_statement, directory, concat_files)

    patch_completion_texts, patch_strings = get_patch_string(problem_statement, directory, llm_retrieved_files)
    verification_completion_texts_aggregated, judgments_aggregated = get_verification(
        problem_statement, bm25_top_files, patch_strings, directory
    )

    scores, patch_string = choose_patch_string(patch_strings, judgments_aggregated, directory)

    if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        data = {
            "problem_statement": [problem_statement] * BATCH_SIZE,
            "easy_prob": [easy_prob] * BATCH_SIZE,
            "llm_selection_completion_texts": selection_completion_texts,
            "llm_selected_files": llm_selected_files,
            "bm25_top_files": [bm25_top_files] * BATCH_SIZE,
            "llm_retrieval_completion_texts": llm_retrieval_completion_texts,
            "llm_retrieved_files": llm_retrieved_files,
            "patch_completion_text": patch_completion_texts,
            "patch_string": patch_strings,
        }

        for copy_idx in range(VALIDATION_COPY_COUNT):
            data[f"verification_completion_text_{copy_idx}"] = [
                completion_texts[copy_idx] if completion_texts else None
                for completion_texts in verification_completion_texts_aggregated
            ]
            data[f"judgment_{copy_idx}"] = [
                judgments[copy_idx] if judgments else None for judgments in judgments_aggregated
            ]

        data["judgment_count_true"] = [sum(judgments) for judgments in judgments_aggregated]
        data["score"] = scores
        data["elapsed_time"] = [time.time() - start_time] * BATCH_SIZE

        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            json.dump(data, open(Path(output_dir) / "predictions.json", "w"))
            pd.DataFrame(data).to_csv(Path(output_dir) / "predictions.csv", index=False)

    print("submitted patch_string")
    print(patch_string)

    # if patch_string is None:
    #     return None

    # return patch_string

    if patch_string is None:
        patch_string = ""

    with open(patch_filepath, "w") as f:
        f.write(patch_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    print("Start running the predictor. ")

    parser.add_argument('problem_filepath')
    parser.add_argument('patch_filepath')
    parser.add_argument('--skip_prediction')
    parser.add_argument('--output_dir')
    parser.add_argument('--directory', default='repo')

    args = parser.parse_args()
    problem_filepath = args.problem_filepath
    patch_filepath = args.patch_filepath
    skip_prediction = args.skip_prediction
    if skip_prediction == "True":
        skip_prediction = True
    else:
        skip_prediction = False
    output_dir = args.output_dir
    directory = args.directory
    
    with open(problem_filepath, "r") as f:
        problem_statement = f.read()

    predict_inner(
        problem_statement=problem_statement,
        patch_filepath=patch_filepath,
        skip_prediction=skip_prediction,
        output_dir=output_dir,
        directory=directory
    )