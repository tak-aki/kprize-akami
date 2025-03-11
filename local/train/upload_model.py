import json
import shutil
from pathlib import Path
from typing import Any, List

import click
from kaggle.api.kaggle_api_extended import KaggleApi


@click.command()
@click.option("--title", "-t", default="kprize-akami-difficulty-model")
@click.option(
    "--dirs",
    "-d",
    type=list[str],
    default=[
        "output_train/exp004/003/fold0/checkpoint-100",  #  0.6000
        "output_train/exp004/70b_003/fold0/checkpoint-100",  #  0.6300
    ],
)
@click.option("--user_name", "-u", default="kami634")
@click.option("--new", "-n", is_flag=True)
def main(
    title: str,
    dirs: List[Path],
    user_name: str = "kami634",
    new: bool = False,
):
    """extentionを指定して、dir以下のファイルをzipに圧縮し、kaggleにアップロードする。

    Args:
        title (str): kaggleにアップロードするときのタイトル
        dir (Path): アップロードするファイルがあるディレクトリ
        user_name (str, optional): kaggleのユーザー名.
        new (bool, optional): 新規データセットとしてアップロードするかどうか.
    """
    tmp_dir = Path("./tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ディレクトリ全体をtmp_dirにコピー
    for dir in dirs:
        name = "-".join(dir.split("/"))
        dir = Path(dir)

        shutil.copytree(dir, tmp_dir / name)

    # dataset-metadata.jsonを作成
    dataset_metadata: dict[str, Any] = {}
    dataset_metadata["id"] = f"{user_name}/{title}"
    dataset_metadata["licenses"] = [{"name": "CC0-1.0"}]
    dataset_metadata["title"] = title
    with open(tmp_dir / "dataset-metadata.json", "w") as f:
        json.dump(dataset_metadata, f, indent=4)

    # api認証
    api = KaggleApi()
    api.authenticate()

    if new:
        api.dataset_create_new(
            folder=tmp_dir,
            dir_mode="tar",
            convert_to_csv=False,
            public=False,
        )
    else:
        api.dataset_create_version(
            folder=tmp_dir,
            version_notes="",
            dir_mode="tar",
            convert_to_csv=False,
        )

    # delete tmp dir
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
