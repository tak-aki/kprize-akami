# utils.py
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import unidiff


def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text))


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
    except Exception:
        return False


def save_results(data: dict, start_time: float):
    filename = f"{str(int(time.time() - start_time)).zfill(5)}.csv"
    pd.DataFrame(data).to_csv(filename, index=False)
