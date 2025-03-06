import io
import os
import re
import time
from pathlib import Path
from typing import List

import pandas as pd

start_time = time.time()

from .config import REPO_PATH, VALIDATION_COPY_COUNT, tokenizer
from .fetch_file import fetch_file_contents
from .patching import get_patch_string
from .selection_query import get_selection_query
from .utils import count_tokens, stringify_directory
from .verifying import choose_patch_string, get_verification


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

    directory_string = stringify_directory(directory)

    selection_completion_texts, file_queries = get_selection_query(directory_string, problem_statement)

    file_content_strings: List[str] = [fetch_file_contents(file_query) for file_query in file_queries]

    file_names_list = [re.findall(r"\[file name\]: (.+)", file_content_string) for file_content_string in file_content_strings]

    #一番ヒット数が多かったものだけを返す
    file_names = max(file_names_list, key=len)

    return file_names
