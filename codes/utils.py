# utils.py
import json
import os
import difflib
import re
import time
from typing import Dict, List, Optional

import pandas as pd
import unidiff


def stringify_directory(directory: str) -> str:
    full_paths: List[str] = []
    banned_strings = [".venv", ".pyc", ".txt", ".pytest_cache", ".github", "/doc/", "/tests/"]

    for root, dirs, files in os.walk(directory):
        for file in files:
            for banned_string in banned_strings:
                if banned_string in root or banned_string in file:
                    break
            else:
                full_path: str = os.path.join(root, file)
                full_paths.append(full_path)
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

def remove_line_numbers(content):
    # Remove line numbers from the file content
    return re.sub(r"^\d+\s", "", content, flags=re.MULTILINE)

def generate_git_diff(json_str):

    try:
        json_data = json.loads(json_str)
        edited_code = json_data["edited code"]
    except Exception as e:
        print(f"Error in parsing json output for task code editing: {e}")
        return ""
    
    patch = ""
    for edit in edited_code:
        try:
            file_path = edit["file"]
            old_snippet = remove_line_numbers(edit["code snippet to be modified"].rstrip()).split("\n")
            new_snippet = edit["edited code snippet"].rstrip().split("\n")
            
            diff = difflib.unified_diff(
                old_snippet, new_snippet,
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                lineterm=""
            )
        
            patch += "\n".join(diff) + "\n"
        except Exception as e:
            print(f"Error in generating git diff for task code editing: {e}")
    
    return patch

def extract_and_make_patch_string(text: str) -> Optional[str]:
    import xml.etree.ElementTree as ET

    pattern: str = r"<modification>(.*?)</modification>"
    matches: List[str] = re.findall(pattern, text, re.DOTALL)

    if len(matches) == 0:
        print("No <modification> field found")
        return ""
    
    patch = generate_git_diff(matches[0])
    
    if patch == "":
        return None
    else:
        return patch


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
