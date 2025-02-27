import json
import logging
import os
import subprocess

from datasets import load_dataset, load_from_disk
from tqdm import tqdm

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
def clone_and_checkout(owner: str, repo_name: str, commit_hash: str, repo_dir: str):
    repo_url = f"https://github.com/{owner}/{repo_name}.git"

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
    data_dir: str, output_path: str, dataset_name: str = "princeton-nlp/SWE-bench", debug: bool = False
) -> list[dict]:
    logger.info("Setting up data...")

    # --- SWE-bench データセットのダウンロード、配置 ---
    if not os.path.exists(data_dir):
        swebench = load_dataset(dataset_name, split="train")
        swebench.save_to_disk(data_dir)
    else:
        swebench = load_from_disk(data_dir)

    swebench_df = swebench.to_pandas()

    if debug:
        swebench_df = swebench_df.iloc[:100]

    # 処理のベースとなるデータセットのセットアップ
    processing_dataset = []
    for _, row in tqdm(swebench_df.iterrows(), total=len(swebench_df), desc="Process data"):
        instance_id = row["instance_id"]
        owner, repo_name = row["repo"].split("/")
        commit_hash = row["base_commit"]
        problem_statement = row["problem_statement"]
        patch = row["patch"]
        processing_data = {
            "instance_id": instance_id,
            "owner": owner,
            "repo_name": repo_name,
            "commit_hash": commit_hash,
            "problem_statement": problem_statement,
            "patch": patch,
        }
        processing_dataset.append(processing_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processing_dataset, f, indent=4)

    logger.info("Data setup completed.")
    logger.info(f"Pick {len(processing_dataset)} sample from {len(swebench_df)} samples.")
    return processing_dataset
