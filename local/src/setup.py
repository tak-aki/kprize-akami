import json
import logging
from pathlib import Path

from datasets import load_dataset, load_from_disk

from .repository import GitHubRepo, RepoUVManager
from .utils import SWEBenchInstance

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
logger.propagate = False


def setup_environment(
    instance: SWEBenchInstance, root_dir: str | Path = "/kaggle/tmp", fallback_python_version: str = "3.10"
) -> RepoUVManager:
    """
    Sets up an environment for a given SWE-Bench instance following the steps:
        (1) Generate a name for the environment (instance ID + commit hash).
        (2) Initialize the repository handler and clone the repo at environment_setup_commit.
        (3) Initialize the RepoUVManager with fallback_python_version (or a more advanced detection).
        (4) Create the virtual environment.
        (5) Install dependencies (requirements.txt, pyproject.toml, or setup.py).
        (6) Checkout the base_commit after installation.

    Args:
        instance (SWEBenchInstance):
            Contains information about the environment (repo, commits, etc.).
        root_dir (str | Path):
            The root directory where the repository should be cloned.
        fallback_python_version (str):
            The Python version to use if no advanced detection is done.

    Returns:
        RepoUVManager:
            - The specialized UV environment manager.
    """

    # ----------------------------------------------------------
    # (1) Create the GitHub repo object
    # ----------------------------------------------------------
    github_repo = GitHubRepo.from_swebench_instance(instance, root_dir=root_dir)

    # ----------------------------------------------------------
    # (2) Initialize a RepoUVManager with the new environment name
    #     and link to the cloned GitHubRepo
    # ----------------------------------------------------------
    # We store the environment in root_dir/env_name (or any path you like)
    repo_uv = RepoUVManager(
        venv_dir_path=root_dir, github_repo=github_repo, fallback_python_version=fallback_python_version
    )

    # ----------------------------------------------------------
    # (3) Clone the repo and checkout the correct commit for
    #     env setup. Done internally now within __init__
    # ----------------------------------------------------------
    # repo_uv.clone_and_checkout_repo()

    # ----------------------------------------------------------
    # (4) Remove github actions
    # ----------------------------------------------------------
    repo_uv.remove_github_actions()

    # ----------------------------------------------------------
    # (5) Create the UV virtual environment
    # ----------------------------------------------------------
    repo_uv.initialize()

    # ----------------------------------------------------------
    # (6) Install dependencies, if any
    #     - This might look in requirements.txt or do 'uv pip install .'
    # ----------------------------------------------------------
    repo_uv.install_repo_dependencies()

    # ----------------------------------------------------------
    # (7) Checkout the base_commit if different from environment_setup_commit
    # ----------------------------------------------------------
    if instance.base_commit != instance.environment_setup_commit:
        repo_uv.github_repo.checkout_commit(instance.base_commit)

    # Return both objects so user can interact further
    return repo_uv


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
