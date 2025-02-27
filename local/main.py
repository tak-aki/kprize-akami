import json
import shutil
from pathlib import Path

from codes.submit import REPO_PATH, predict_inner
from local.src.setup import clone_and_checkout, setup_data


def main():
    data_dir = Path("input/swe-bench/")
    output_path = data_dir / "processed/dataset_setuped.json"

    if output_path.exists():
        swe_bench_data = json.load(output_path.open())
    else:
        data_dir.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(exist_ok=True)

        swe_bench_data = setup_data(
            data_dir / "cache/", output_path, dataset_name="princeton-nlp/SWE-bench", debug=False
        )

    result_dir = data_dir / "result/"
    result_dir.mkdir(exist_ok=True)
    shutil.rmtree(REPO_PATH, ignore_errors=True)

    for data in swe_bench_data:
        clone_and_checkout(data["owner"], data["repo_name"], data["commit_hash"], REPO_PATH)
        patch = predict_inner(data["problem_statement"], None, None, None, False)

        instance_id = data["instance_id"]
        with open(result_dir / f"{instance_id}.patch", "w") as f:
            f.write(patch)

        shutil.rmtree(REPO_PATH, ignore_errors=True)


if __name__ == "__main__":
    main()
