import io
import os
import re
import time
from pathlib import Path
from typing import List

import pandas as pd

start_time = time.time()

from .config import BATCH_SIZE 
from .llm_selection import get_llm_selection
from .bm25 import get_bm25_top_files
from .llm_retrieve import get_llm_retrieval
from .utils import count_tokens, stringify_directory

def retrieve(
    problem_statement: str,
    skip_prediction: bool = False,
    output_dir: str | None = None,
    directory: str = "repo",
) -> List[List[str]]:
    """
    retrieval性能をlocalで評価するためのwrapper関数
    Batch内で一番ヒット数が多かったものだけを返す
    """
    if skip_prediction:
        return None

    directory_string = stringify_directory(directory)

    selection_completion_texts, llm_selected_files = get_llm_selection(directory_string, problem_statement)
    bm25_top_files = get_bm25_top_files(problem_statement, directory, top_k=30)

    concat_files = [sf + [bf for bf in bm25_top_files if bf not in sf] for sf in llm_selected_files] # llm selectionにあるファイルはbm25から除去しつつ結合

    llm_retrieval_completion_texts, llm_retrieved_files = get_llm_retrieval(problem_statement, directory, concat_files)

    data = {
            "problem_statement": [problem_statement for _ in range(BATCH_SIZE)],
            "llm_selection_completion_texts": selection_completion_texts,
            "llm_selected_files": llm_selected_files,
            "bm25_top_files": [bm25_top_files for _ in range(BATCH_SIZE)],
            "llm_retrieval_completion_texts": llm_retrieval_completion_texts,
            "llm_retrieved_files": llm_retrieved_files,
        }
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(data).to_csv(Path(output_dir) / "predictions.csv", index=False)

    return llm_selected_files, bm25_top_files, llm_retrieved_files
