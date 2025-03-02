import json
import logging
import os
import subprocess
from pathlib import Path

from datasets import load_dataset, load_from_disk

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
logger.propagate = False


# リポジトリをローカルで再現
def clone_and_checkout(repo: str, commit_hash: str, repo_dir: str):
    repo_url = f"https://github.com/{repo}.git"

    if not os.path.exists(repo_dir):
        subprocess.run(
            ["git", "clone", repo_url, repo_dir], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    try:
        subprocess.run(
            ["git", "checkout", commit_hash],
            cwd=repo_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        pass

    try:
        subprocess.run(["git", "stash"], cwd=repo_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(
            ["git", "checkout", commit_hash],
            cwd=repo_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    # ここでもエラーが出たら一度ディレクトリを削除し、cloneからやり直す
    except subprocess.CalledProcessError:
        pass

    try:
        subprocess.run(["rm", "-rf", repo_dir], check=True)
        subprocess.run(
            ["git", "clone", repo_url, repo_dir],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "checkout", commit_hash],
            cwd=repo_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    # それでもエラーが出たらエラーを投げる
    except subprocess.CalledProcessError:
        raise


def setup_data(
    data_dir: str, output_dir: str, dataset_name: str = "princeton-nlp/SWE-bench", split: str = "train"
) -> list[dict]:
    output_path = Path(output_dir) / f"dataset_{split}.json"

    if output_path.exists():
        logger.info("Loading cached data.")
        with output_path.open("r") as f:
            processing_dataset = json.load(f)
        return processing_dataset

    # --- SWE-bench データセットのダウンロード、配置 ---
    dataset_path = Path(data_dir) / split

    if not dataset_path.exists():
        dataset_path.mkdir(parents=True, exist_ok=True)
        swebench = load_dataset(dataset_name, split=split)
        swebench.save_to_disk(data_dir)
    else:
        swebench = load_from_disk(data_dir)

    swebench_df = swebench.to_pandas()
    processing_dataset_str = swebench_df.to_json(orient="records")
    processing_dataset = json.loads(processing_dataset_str)

    with open(output_path, "w") as f:
        f.write(processing_dataset_str)

    logger.info("Data setup completed.")
    logger.info(f"Pick {len(processing_dataset)} sample from {len(swebench_df)} samples.")
    return processing_dataset
