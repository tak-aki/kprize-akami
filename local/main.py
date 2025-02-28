import json
import logging
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from codes.submit import REPO_PATH, predict_inner
from local.src.setup import setup_data, setup_demo_environment
from local.src.utils import SWEBenchInstance

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
    start_time = time.time()
    split = "test"  # train, test, dev
    num_instances = 100

    data_dir = Path("input/swe-bench/")
    cache_dir = data_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Setting up the dataset.")
    swe_bench_data = setup_data(cache_dir, output_dir, dataset_name="princeton-nlp/SWE-bench", split=split)
    swe_bench_data = swe_bench_data[:num_instances]

    output_dir = Path("output/")
    output_dir.mkdir(exist_ok=True)

    logger.info("Start running the predictor.")

    n_corrent = 0
    n_wrong = 0
    n_skipped = 0

    for data in swe_bench_data:
        instance = SWEBenchInstance.from_df_row(data)
        shutil.rmtree(REPO_PATH, ignore_errors=True)

        # Setup
        logger.info(f"Setting up the environment for {instance.instance_id}.")
        env = setup_demo_environment(instance, REPO_PATH)

        # Predict the patch
        logger.info(f"Predicting the patch for {instance.instance_id}.")
        patch = predict_inner(
            instance.problem_statement,
            repo_archive=None,
            pip_packages_archive=None,
            env_setup_cmds_templates=None,
            skip_prediction=False,
            save_result=False,
        )

        if patch is None:
            n_skipped += 1
            continue

        result_dir = output_dir / instance.instance_id
        result_dir.mkdir(exist_ok=True)

        with open(result_dir / "patch.txt", "w") as f:
            f.write(patch)

        # Test
        logger.info(f"Testing the patch for {instance.instance_id}.")

        try:
            env.apply_patch(patch)
        except Exception as e:
            logger.error(f"Failed to apply the patch for {instance.instance_id}.")
            logger.error(e)
            n_wrong += 1
            continue

        env.apply_patch(instance.test_patch)
        test_results = [env.run_pytest(test) for test in instance.fail_to_pass + instance.pass_to_pass]

        if all([result.success for result in test_results]):
            logger.info(f"Patch for {instance.instance_id} is correct.")
            n_corrent += 1
        else:
            logger.info(f"Patch for {instance.instance_id} is wrong.")
            n_wrong += 1

        with open(result_dir / "result.json", "w") as f:
            json.dump([asdict(result) for result in test_results], f, indent=4)

    shutil.rmtree(REPO_PATH, ignore_errors=True)

    elapsed_time = time.time() - start_time
    logger.info(f"Elapsed time: {elapsed_time // 3600} hours {elapsed_time % 3600 // 60} minutes.")
    logger.info(f"n_correct: {n_corrent}, n_wrong: {n_wrong}, n_skipped: {n_skipped}")


if __name__ == "__main__":
    main()
