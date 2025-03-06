import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import cast, List

import numpy as np
import pandas as pd

from codes_fixer.retrieval_wrapper import REPO_PATH, retrieve
from retrieval_checker.src.setup import clone_and_checkout, setup_data, find_gold_files
from retrieval_checker.src.utils import save_json, set_seed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
logger.propagate = False

def eval_retrieval(gold_files: List[str], retrieved_files: List[str]) -> dict:
    # precision, recall, MAP@30を計算
    retrieved_set = set(retrieved_files)
    gold_set = set(gold_files)

    # Precision
    true_positives = len(retrieved_set & gold_set)
    precision = true_positives / len(retrieved_set) if retrieved_set else 0

    # Recall
    recall = true_positives / len(gold_set) if gold_set else 0

    return {
        "num_gold_files": len(gold_files),
        "num_retrieved_files": len(retrieved_files),
        "precision": precision,
        "recall": recall,}

def main():
    split = "test"  # train, test, dev
    dataset_name = "princeton-nlp/SWE-bench"
    num_instances = 100
    seed = 1029

    start_time = time.time()
    set_seed(seed)
    rng = np.random.default_rng(seed)

    data_dir = Path("input/") / dataset_name.split("/")[-1].lower()
    cache_dir = data_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir = data_dir / "processed"
    data_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Setting up the dataset.")
    swe_bench_data = setup_data(cache_dir, data_output_dir, dataset_name=dataset_name, split=split)
    swe_bench_data = rng.choice(swe_bench_data, num_instances, replace=False)

    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = Path(f"output/{run_id}")
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Start running.")
    result_list = []
    for data in swe_bench_data:
        instance_id = data["instance_id"]
        patch = data["patch"]
        gold_files = find_gold_files(patch)

        shutil.rmtree(REPO_PATH, ignore_errors=True)

        logger.info(f"Processing {instance_id}.")

        result_dir = output_dir / instance_id
        result_dir.mkdir(exist_ok=True)

        # Setup
        logger.info(f"Setting up the environment for {instance_id}.")
        clone_and_checkout(data["repo"], data["base_commit"], REPO_PATH)

        # Predict the patch
        logger.info(f"Retrieve files for {instance_id}.")
        retrieved_files = retrieve(
            data["problem_statement"],
            repo_archive=None,
            pip_packages_archive=None,
            env_setup_cmds_templates=None,
            skip_prediction=False,
            output_dir=result_dir,
        )

        # Evaluate
        logger.info(f"Evaluating the retrieval for {instance_id}.")
        eval_result = eval_retrieval(gold_files, retrieved_files)        

        result = {
            "instance_id": instance_id,
            "gold_files": gold_files,
            "retrieved_files": retrieved_files
        } | eval_result
        result_path = result_dir / "result.json"

        logger.info(f"Evaluation result: {result}")
        save_json(result, result_path)
        result_list.append(result)
    
    # Save the results
    result_df = pd.DataFrame(result_list)
    result_df.to_csv(output_dir / "result.csv", index=False)

if __name__ == "__main__":
    main()
