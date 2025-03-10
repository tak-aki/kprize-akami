# llm_retrieve.py
from typing import Dict, List, Optional, Tuple
import ast
import re
import os
import json

from vllm import RequestOutput, SamplingParams, LLM
import torch
from .utils import count_tokens
from .config import BATCH_SIZE, MAX_NUM_SEQS

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

# def count_tokens(tokenizer, text): # Qwen2ベースのものにはこの書き方
#     return len(tokenizer(text)["input_ids"])

def find_readme(repo_path):
    # READMEファイル名の候補リスト（大文字小文字を区別しない）
    readme_variations = ["README.md", "README.rst", "README.txt", "README"]

    # リポジトリトップ階層の全ファイルをリストアップ
    for root, _, files in os.walk(repo_path):
        if root == repo_path:  # トップ階層だけを探索
            # ファイル名を小文字に変換してチェック
            lower_case_files = {file.lower(): file for file in files}
            for readme in readme_variations:
                if readme.lower() in lower_case_files:
                    # 実際のファイル名を取得
                    readme_path = os.path.join(root, lower_case_files[readme.lower()])
                    with open(readme_path, "r", encoding="utf-8") as f:
                        return f.read()
    return ""  # READMEが見つからなかった場合

def extract_code_skeleton(file_content: str) -> str:
    """
    Extracts a code skeleton from a file content.
    Includes module docstrings, class headers, method signatures, and the first/last 5 lines of functions.
    """
    skeleton_parts = []

    # Extract module docstring if available
    module_docstring_match = re.search(r'^"""(.*?)"""', file_content, re.DOTALL)
    if module_docstring_match:
        skeleton_parts.append(f'"""{module_docstring_match.group(1).strip()}"""')

    # Extract class definitions
    class_matches = re.finditer(r'(class\s+[\w_]+[\(].*?[\)]*\s*[:].*)', file_content, re.DOTALL)
    for match in class_matches:
        skeleton_parts.append(match.group(1))

        # Extract method signatures
        method_matches = re.finditer(r'(def\s+[\w_]+\s*[\(].*?[\)]\s*[:].*)', file_content, re.DOTALL)
        for method_match in method_matches:
            skeleton_parts.append(f"    {method_match.group(1)}")

    # Extract function definitions
    function_matches = re.finditer(r'(def\s+[\w_]+\s*[\(].*?[\)]\s*[:].*)', file_content, re.DOTALL)
    for match in function_matches:
        func_def = match.group(1).strip()
        skeleton_parts.append(func_def)

        func_content_match = re.search(
            re.escape(func_def) + r'\n(.*?)\n(?=def|\Z)', file_content, re.DOTALL
        )
        if func_content_match:
            func_content = func_content_match.group(1).strip().splitlines()
            first_5_lines = "\n    ".join(func_content[:5]).strip() if len(func_content) >=5 else  "\n    ".join(func_content).strip()
            last_5_lines =  "\n    ".join(func_content[-5:]).strip() if len(func_content) >=5 else ""


            skeleton_parts.append(f"    [First five lines]:\n    {first_5_lines}")
            if last_5_lines:
                skeleton_parts.append(f"    [Last five lines]:\n    {last_5_lines}")

    return "\n".join(skeleton_parts)

def extract_source_segment(source_lines, node):
    """
    node の lineno, end_lineno を利用してソースコードの該当部分を抽出。
    """
    # ast の行番号は 1-based, Python のリストは 0-based のため調整する
    start = node.lineno - 1
    end = node.end_lineno
    return source_lines[start:end]

def get_class_bases(class_node):
    """
    クラス定義の継承元を文字列として取得 (例: "MyClass(BaseClass)").
    Bases(継承元)が複数ある場合はカンマ区切りでまとめる。
    """
    base_names = []
    for base in class_node.bases:
        # 例えば ast.Name(id='BaseClass') や ast.Attribute(value=..., attr=...) 等があり得る
        base_str = ast.unparse(base)
        base_names.append(base_str)
    if base_names:
        return f"({', '.join(base_names)})"
    return ""

def get_function_signature(func_node):
    """
    関数/メソッドのシグニチャ (例: "foo(bar, baz=1)") を文字列として返す。
    デコレータ行は無視して、"def ..." となっている行を抽出する。
    返り値から "def " を取り除き、末尾の ":" も削除した文字列を返す。
    """
    unparsed = ast.unparse(func_node)
    lines = unparsed.split('\n')
    for line in lines:
        line_stripped = line.strip()
        # デコレータは '@' で始まる行が多いのでスキップ
        # "def " で始まる行を見つけたらそこからシグニチャを取り出す
        if line_stripped.startswith('def '):
            # "def " 以降を取り、末尾の ':' を削る
            sig = line_stripped[4:].rstrip(':')
            return sig
    # 念のため何も見つからなかった場合
    return func_node.name

def get_docstring(node):
    """
    ast.get_docstring で取得した docstring (もしくは None) を返す。
    ここでは無ければ空文字列にする。
    """
    doc = ast.get_docstring(node)
    if doc is None:
        return ""
    return doc

def get_partial_content(lines, max_head=5, max_tail=5):
    """
    関数の全文を行リストで受け取り、先頭 max_head 行と末尾 max_tail 行を抜き出して
    '...\n' を挟んだ文字列を返す。
    """
    if len(lines) <= max_head + max_tail:
        # 行数が少ない場合はそのまま結合
        return "\n".join(lines)
    head_part = lines[:max_head]
    tail_part = lines[-max_tail:]
    return "\n".join(head_part) + "\n...\n" + "\n".join(tail_part)

def parse_python_file(source, file_path):
    """
    Python のソースコードを解析して、モジュール docstring, クラス定義, 関数定義を抽出する。
    """

    # 行リストも保持 (後で部分的なソース抽出に使う)
    source_lines = source.splitlines()

    # AST 解析
    try:
        tree = ast.parse(source)

        # モジュール docstring
        module_docstring = get_docstring(tree)

        classes_info = []
        functions_info = []

        # トップレベルで定義されている node を走査
        for node in tree.body:
            # クラス定義かどうか
            if isinstance(node, ast.ClassDef):
                class_name = node.name + get_class_bases(node)
                class_docstring = get_docstring(node)

                # メソッドを取得
                methods = []
                for class_body_item in node.body:
                    if isinstance(class_body_item, ast.FunctionDef):
                        if class_body_item.name.startswith('_'):
                            # 内部用メソッドは抜き出さない
                            continue
                        methods.append(get_function_signature(class_body_item))

                classes_info.append({
                    "name": class_name,
                    "docstring": class_docstring,
                    "methods": methods
                })

            # 関数定義かどうか (トップレベル)
            elif isinstance(node, ast.FunctionDef):
                func_name = get_function_signature(node)

                # 関数部分のソースコードを取得して先頭・末尾 5 行だけ抜き出す
                lines = extract_source_segment(source_lines, node)
                partial_content = get_partial_content(lines, 5, 5)

                functions_info.append({
                    "name": func_name,
                    "content": partial_content
                })

        # 結果をまとめて JSON 出力用の dict を作る
        result = {
            "file_path": file_path,
            "module_docstring": module_docstring,
            "classes": classes_info,
            "functions": functions_info
        }
        return result
    except Exception as e:
        logger.info(f"Error parsing file {file_path}: {e}")
        return None

def extract_retrieved_files(response_text: str) -> List[str]:
    """
    Extracts the retrieved files from the response text.
    """
    try:
        files = json.loads(response_text)
    except json.JSONDecodeError:
        logger.info("Error parsing JSON response.")
        files = {
            "files for editing": []
        }
    return files["files for editing"]

def load_file_content(codebase_path: str, file_path_rel: str) -> str:
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
        return f.read()

def get_llm_retrieval(problem_statement: str, codebase_path: str, candidate_file_batch: List[dict]):

    ## Initialize LLM
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if os.getenv("KAGGLE_KERNEL_RUN_TYPE") or os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        llm_model_pth: str = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-32b-awq/1" #TODO: Change this to the correct model
        num_gpus: int = 4
    else:
        llm_model_pth: str = "internlm/SWE-Fixer-Retriever-7B"
        num_gpus: int = torch.cuda.device_count()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))

    MAX_TOKENS: int = 2048

    MAX_MODEL_LEN: int = 65_536


    llm: LLM = LLM(
        model=llm_model_pth,
        max_num_seqs=MAX_NUM_SEQS,  # Maximum number of sequences per iteration. Default is 256
        max_model_len=MAX_MODEL_LEN,  # Model context length
        trust_remote_code=True,  # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
        tensor_parallel_size=num_gpus,  # The number of GPUs to use for distributed execution with tensor parallelism
        gpu_memory_utilization=0.95,  # The ratio (between 0 and 1) of GPU memory to reserve for the model
        enable_prefix_caching=True, 
        seed=2024,
    )

    tokenizer = llm.get_tokenizer()

    sampling_params: SamplingParams = SamplingParams(
        temperature=0.6,  # randomness of the sampling
        min_p=0.01,
        skip_special_tokens=True,  # Whether to skip special tokens in the output
        max_tokens=MAX_TOKENS,
    )

    # プロンプトの作成
    readme = find_readme(codebase_path)
    list_of_retrieve_prompt = []
    for candidate_file in candidate_file_batch:
        file_documentations = [parse_python_file(load_file_content(codebase_path, file_path), file_path) for file_path in candidate_file]
        #file_documentationsからNoneを除外
        file_documentations = [f for f in file_documentations if f is not None]
        logger.info(f"num valid files: {len(file_documentations)}")

        if len(file_documentations) == 0:
            logger.info("All files are invalid")
            list_of_retrieve_prompt.append("")
            continue
        prompt_json = {
            "input": {
                "issue": problem_statement,
                "readme file": readme,
                "retrieved file documentations": file_documentations,
                "task": "In this task, you will be provided with a software development issue from a real-world GitHub repository, along with the repository's README file and a preliminarily retrieved file (documentation). Your objective is to carefully analyze the issue in the context of the provided file and Determine whether an issue and a file are related."
            }, 
            "output control": {
                "files for editing": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            }
        }
        while count_tokens(json.dumps(prompt_json), tokenizer) > (MAX_MODEL_LEN - 100): # 100 is a buffer
            logger.info(
                f"Exceeding token limit ({count_tokens(json.dumps(prompt_json), tokenizer)} > {MAX_MODEL_LEN}), remove last file and remaining files: {len(prompt_json['input']['retrieved file documentations'])-1}"
            )
            del prompt_json["input"]["retrieved file documentations"][-1]
        retrieve_prompt = json.dumps(prompt_json)
        list_of_retrieve_prompt.append(retrieve_prompt)

    list_of_messages = [
        [
            # {"role": "system", "content": None},
            {"role": "user", "content": retrieve_prompt},
        ]
        for retrieve_prompt in list_of_retrieve_prompt
    ]

    prompt_texts: List[str] = [
        tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)  # type: ignore
        for messages in list_of_messages
    ]

    logger.info(f"prompt_texts token length {[count_tokens(text, tokenizer) for text in prompt_texts]}")
    request_outputs: list[RequestOutput] = llm.generate(prompt_texts, sampling_params=sampling_params)
    response_texts_from_inference: List[str] = [request_output.outputs[0].text for request_output in request_outputs]
    logger.info(f"response_texts_from_inference token length : {[count_tokens(text, tokenizer) for text in response_texts_from_inference]}")
    completion_texts_from_inference = [
        prompt_text + response_text for prompt_text, response_text in zip(prompt_texts, response_texts_from_inference)
    ]
    retrieved_files_from_inference: List[Optional[str]] = [
        extract_retrieved_files(response_text) for response_text in response_texts_from_inference
    ]

    completion_texts: list[str] = ["" for _ in candidate_file_batch]
    retrieved_files: List[Optional[str]] = [None for _ in candidate_file_batch]
    for input_idx, (completion_text, retrieved_file) in enumerate(
        zip(completion_texts_from_inference, retrieved_files_from_inference)
    ):
        completion_texts[input_idx] = completion_text
        retrieved_files[input_idx] = retrieved_file

    logger.info(f"num retrieved files: {[len(f) for f in retrieved_files]}")

    return completion_texts, retrieved_files
