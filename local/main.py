import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import docker
import numpy as np

from codes_fixer.submit import predict_inner
from local.src.setup import clone_and_checkout, setup_data
from local.src.utils import save_json, set_seed
from swebench.harness.constants import SWEbenchInstance
from swebench.harness.docker_utils import clean_images, list_images
from swebench.harness.run_evaluation import run_instance
from swebench.harness.test_spec.test_spec import make_test_spec

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
logger.propagate = False


def main():
    split = "test"  # train, test, dev
    dataset_name = "princeton-nlp/SWE-bench"
    num_instances = 100
    seed = 1029
    REPO_PATH = "repo"

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

    logger.info("Start running the predictor.")

    docker_client = docker.from_env()

    n_corrent = 0
    n_wrong = 0
    n_skipped = 0

    wrong_instances = {}

    for data in swe_bench_data:
        instance = cast(SWEbenchInstance, data)
        instance_id = instance["instance_id"]
        fail_to_pass = json.loads(instance["FAIL_TO_PASS"])
        pass_to_pass = json.loads(instance["PASS_TO_PASS"])

        fail_to_pass_test_count = len(fail_to_pass)
        pass_to_pass_test_count = len(pass_to_pass)
        num_tests = fail_to_pass_test_count + pass_to_pass_test_count

        shutil.rmtree(REPO_PATH, ignore_errors=True)

        logger.info("-----------------------------------------------------------------------------")
        logger.info(f"*** Current result: {n_corrent} correct, {n_wrong} wrong, {n_skipped} skipped. ***")
        logger.info("-----------------------------------------------------------------------------")
        logger.info(f"Processing {instance_id}.")

        result_dir = output_dir / instance_id
        result_dir.mkdir(exist_ok=True)

        # Setup
        logger.info(f"Setting up the environment for {instance_id}.")
        clone_and_checkout(instance["repo"], instance["base_commit"], REPO_PATH)

        # Predict the patch
        logger.info(f"Predicting the patch for {instance_id}.")
        patch = predict_inner(
            instance["problem_statement"],
            repo_archive=None,
            pip_packages_archive=None,
            env_setup_cmds_templates=None,
            skip_prediction=False,
            output_dir=result_dir,
        )

        result = {
            "instance_id": instance_id,
            "status": "skipped",
            "success_count": 0,
            "n_tests": len(fail_to_pass + pass_to_pass),
            "patch": patch,
            "test_results": [],
        }
        result_path = result_dir / "result.json"

        if patch is None:
            n_skipped += 1
            save_json(result, result_path)
            continue

        # Test
        logger.info(f"Testing the patch for {instance_id}.")
        test_spec = make_test_spec(instance, namespace="swebench", instance_image_tag="latest")
        prediction = {
            "instance_id": instance_id,
            "model_name_or_path": None,
            "model_patch": patch,
        }

        test_result = run_instance(
            test_spec,
            prediction,
            rm_image=True,
            force_rebuild=False,
            client=docker_client,
            run_id=run_id,
            timeout=1800,
            rewrite_reports=False,
        )

        if test_result is None:
            logger.info(f"Failed to run the test for {instance_id}.")
            wrong_instances[instance_id] = "Error in test execution"
            n_wrong += 1
            result["status"] = "wrong"
            save_json(result, result_path)
            continue

        report = test_result[1][instance_id]

        if not report["patch_successfully_applied"]:
            logger.info(f"Failed to apply the patch for {instance_id}.")

            n_wrong += 1
            result["status"] = "wrong"
            save_json(result, result_path)
            continue

        test_results = report["tests_status"]
        fail_to_pass_success_count = len(test_results["FAIL_TO_PASS"]["success"])
        pass_to_pass_success_count = len(test_results["PASS_TO_PASS"]["success"])
        success_count = fail_to_pass_success_count + pass_to_pass_success_count
        status = "correct" if report["resolved"] else "wrong"

        if status == "correct":
            logger.info(f"Patch for {instance_id} is correct. {success_count}/{num_tests} tests passed.")
            n_corrent += 1
        else:
            logger.info(f"Patch for {instance_id} is wrong. {success_count}/{num_tests} tests passed.")
            logger.info(f"  FAIL_TO_PASS: {fail_to_pass_success_count}/{fail_to_pass_test_count}")
            logger.info(f"  PASS_TO_PASS: {pass_to_pass_success_count}/{pass_to_pass_test_count}")
            wrong_instances[instance_id] = f"Test failed. ({success_count}/{num_tests})"
            n_wrong += 1

        result["status"] = status
        result["success_count"] = success_count
        result["test_results"] = test_results
        save_json(result, result_path)

    clean_images(docker_client, list_images(docker_client), cache_level="env", clean=False)
    shutil.rmtree(REPO_PATH, ignore_errors=True)

    logger.info("-----------------------------------------------------------------------------")
    logger.info("Wrong instances:")
    for instance_id, reason in wrong_instances.items():
        logger.info(f"  {instance_id}: {reason}")
    logger.info("-----------------------------------------------------------------------------")

    elapsed_time = time.time() - start_time
    logger.info(f"Elapsed time: {elapsed_time // 3600} hours {elapsed_time % 3600 // 60} minutes.")
    logger.info(f"Result: {n_corrent} correct, {n_wrong} wrong, {n_skipped} skipped.")


if __name__ == "__main__":
    main()
