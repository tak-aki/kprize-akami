# utils.py
import os
import re
import time
from typing import Dict, List, Optional

import pandas as pd
import unidiff


def stringify_directory(directory: str) -> str:
    full_paths: List[str] = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_paths.append(os.path.join(root, file))
    return "\n".join(full_paths)


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
