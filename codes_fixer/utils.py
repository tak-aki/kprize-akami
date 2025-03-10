# utils.py
import os
import re
import time
from typing import Dict, List, Optional

import pandas as pd
import unidiff

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
logger.propagate = False


def stringify_directory(directory: str) -> str:
    rel_paths: List[str] = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, start=directory)
            rel_paths.append(rel_path)
    return "\n".join(rel_paths)


def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text))


def walk_directory(directory: str, depth: int = 2) -> List[str]:
    relative_paths: List[str] = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            # directoryを削除して相対パスを取得
            rel_path = os.path.relpath(full_path, directory)
            if len(rel_path.split(os.sep)) >= depth:
                relative_paths.append(rel_path)
    return relative_paths


def extract_patch_string(text: str) -> Optional[str]:
    pattern = r"\n```diff\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    return matches[-1] + "\n"


def is_valid_patch_format(patch_string: str) -> bool:
    if not isinstance(patch_string, str):
        return False
    try:
        patch_set = unidiff.PatchSet(patch_string)
        return len(patch_set) > 0
    except Exception:
        return False


def save_results(data: dict, start_time: float):
    filename = f"{str(int(time.time() - start_time)).zfill(5)}.csv"
    pd.DataFrame(data).to_csv(filename, index=False)


def load_file_content(codebase_path: str, file_path_rel: str, with_line_numbers: bool=False) -> str:
    """
    Loads the content of a file.
    """
    file_path = os.path.join(codebase_path, file_path_rel)
    if not os.path.exists(file_path):
        logger.info(f"File not found: {file_path}")
        return ""
    if not os.path.isfile(file_path):
        logger.info(f"Not a file: {file_path}")
        return ""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content =  f.read()
    if with_line_numbers:
        content = "\n".join([f"{i + 1}: {line}" for i, line in enumerate(content.split("\n"))])
    return content