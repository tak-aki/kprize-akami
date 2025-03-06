import io
import os
import re
import time
from pathlib import Path
from typing import List

import pandas as pd

start_time = time.time()

from .config import REPO_PATH, VALIDATION_COPY_COUNT
from .bm25 import get_bm25_top_files

def retrieve(
    problem_statement: str,
    repo_archive: io.BytesIO,
    pip_packages_archive: io.BytesIO,
    env_setup_cmds_templates: list[str],
    skip_prediction: bool = False,
    save_result: bool = True,
    max_file_lines: int = 10,
    output_dir: str | None = None,
) -> List[List[str]]:
    """
    retrieval性能をlocalで評価するためのwrapper関数
    Batch内で一番ヒット数が多かったものだけを返す
    """
    if skip_prediction:
        return None

    directory: str = REPO_PATH

    bm25_top_files = get_bm25_top_files(problem_statement, directory, top_k=5)

    return [f["file_path"] for f in bm25_top_files]
