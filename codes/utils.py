# utils.py
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import unidiff


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


def extract_file_and_error_lines(file_paths: List[str], problem_statement: str) -> List[Tuple[str, Optional[int]]]:
    """
    指定されたファイルパスのリストに対して、problem_statement内から
    以下の各形式でファイルパスとエラー行番号を抽出します。

    1. プレーンな表記: "ファイルパス:数字" または "ファイルパス, line 数字"
    2. GitHub のリンク形式:
       "https://github.com/owner/repo/blob/branch/ファイルパス#L数字"
    3. "at line" 形式:
       "at line 数字 of `ファイルパス`"

    戻り値は、(ファイルパス, エラー行番号) のタプルのリストです。
    エラー行番号が見つからない場合は None として返します。
    """
    results = []

    for fp in file_paths:
        # ファイルパス中の特殊文字をエスケープ
        escaped_fp = re.escape(fp)

        # ① プレーンな表記のパターン
        #    例: src/app.py:123 または src/app.py, line 123
        pattern_plain = rf"({escaped_fp})(?:(?::\s*(\d+))|(?:,\s*line\s+(\d+)))"
        for match in re.finditer(pattern_plain, problem_statement, flags=re.IGNORECASE):
            file_found = match.group(1)
            line_str = match.group(2) or match.group(3)
            line_number = int(line_str) if line_str is not None else None
            results.append((file_found, line_number))

        # ② GitHubリンク形式のパターン
        #    例: https://github.com/owner/repo/blob/branch/src/app.py#L123
        pattern_github = rf"https?://github\.com/[^/]+/[^/]+/blob/[^/]+/{escaped_fp}(?:#L(\d+))?"
        for match in re.finditer(pattern_github, problem_statement, flags=re.IGNORECASE):
            line_str = match.group(1)
            line_number = int(line_str) if line_str is not None else None
            results.append((fp, line_number))

        # ③ "at line" 形式のパターン
        #    例: at line 326 of `astropy/wcs/wcsapi/fitswcs.py`
        pattern_at_line = rf"at\s+line\s+(\d+)\s+of\s+`{escaped_fp}`"
        for match in re.finditer(pattern_at_line, problem_statement, flags=re.IGNORECASE):
            line_str = match.group(1)
            line_number = int(line_str) if line_str is not None else None
            results.append((fp, line_number))

    return list(set(results))


def stringify_directory(directory: str) -> str:
    full_paths: List[str] = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_paths.append(os.path.join(root, file))
    return "\n".join(full_paths)


def extract_file_query(xml_content: str) -> Dict[str, List[str]]:
    import xml.etree.ElementTree as ET

    parsed_data: Dict[str, List[str]] = {}
    pattern: str = r"<root>(.*?)</root>"
    matches: List[str] = re.findall(pattern, xml_content, re.DOTALL)
    for match in matches:
        try:
            root = ET.fromstring("<root>" + match + "</root>")
            for entry in root.findall("entry"):
                filepath = entry.find("filepath")
                filepath_text: Optional[str] = (
                    filepath.text.strip() if filepath is not None and filepath.text is not None else None
                )
                strings_container = entry.find("strings_to_search")
                search_strings: List[str] = []
                if strings_container is not None:
                    for s in strings_container.findall("string_to_search"):
                        if s.text is not None:
                            search_strings.append(s.text.strip())
                parsed_data[filepath_text] = search_strings  # type: ignore
        except Exception as e:
            print("Error parsing output", e)
            print(xml_content)
            return {}
    return parsed_data


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


def patch_dry_run_succeeds(patch_string: str, repo_path: str, timeout: int = 60) -> bool:
    patch_path = Path("patch.txt").resolve()
    with patch_path.open("w") as f:
        f.write(patch_string)
    cmd = f"patch --quiet --dry-run -p1 -i {str(patch_path)} -d {repo_path}"
    try:
        subprocess.run(cmd, shell=True, check=True, timeout=timeout)
        return True
    except subprocess.CalledProcessError:
        return False


def save_results(data: dict, start_time: float):
    filename = f"{str(int(time.time() - start_time)).zfill(5)}.csv"
    pd.DataFrame(data).to_csv(filename, index=False)
