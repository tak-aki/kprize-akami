import os
from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain_community.docstore.document import Document

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

class BM25:
    def __init__(self, top_k=30):
        self.top_k = top_k
        self.retriever = None

    def fit(self, file_docs):
        self.retriever = BM25Retriever.from_documents(file_docs, k=self.top_k)

    def get_rel_files(self, query: str) -> List[dict]:
        """
        Calculates BM25 scores for a given query against a list of documents.
        """
        retriever_output = self.retriever.invoke(query)
        bm25_rel_files = []
        for i, doc in enumerate(retriever_output):
            file_path = doc.metadata.get("file_path", "unknown")
            file_content = doc.page_content
            # contentに行番号をつける
            # file_content = "".join(f"{i:4}|{line}" for i, line in enumerate(file_content.splitlines(keepends=True), start=1))

            bm25_rel_files.append(file_path)
        
        return bm25_rel_files
    
def load_repository_docs(repo_path: str):
    """
    指定したディレクトリ内のファイルを全て読み込み、
    langchain.docstore.document.Document のリストを返す関数。

    :param repo_path: リポジトリのルートディレクトリのパス
    :return: Documentオブジェクトのリスト
    """
    docs = []

    for root, dirs, files in os.walk(repo_path):
        # .git ディレクトリはスキップ（必要に応じて他のディレクトリもスキップ可能）
        if ".git" in dirs:
            dirs.remove(".git")
        if ".github" in dirs:
            dirs.remove(".github")
        if "tests" in dirs:
            dirs.remove("tests")

        for filename in files:
            # Pythonファイル以外はスキップ
            if not filename.endswith(".py"):
                continue
            file_path = os.path.join(root, filename)

            # バイナリファイル等でエラーにならないよう、try-exceptを用いて読み込み
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # 相対パスを取得
                rel_path = os.path.relpath(file_path, start=repo_path)

                # contentの冒頭にpath情報を追加
                content = f"{rel_path}\n{content}"

                # Documentのpage_content にテキスト、metadata にファイルのパス等を保持
                doc = Document(
                    page_content=content,
                    metadata={"file_path": rel_path}
                )
                docs.append(doc)
            except Exception as e:
                # バイナリや読み込み不可のファイルをスキップ
                print(f"Skipping file {file_path} due to error: {e}")

    return docs

def get_bm25_top_files(issue: str, codebase_path: str, top_k: int = 30) -> List[str]:
    """
    Retrieves top-k relevant files using BM25 and a fine-tuned retriever model.
    """
    logger.info(f"Retrieving top {top_k} files using BM25 for the given issue.")
    file_docs = load_repository_docs(codebase_path)

    bm25 = BM25(top_k)
    bm25.fit(file_docs)
    top_files = bm25.get_rel_files(issue)
    logger.info(f"Retrieved {len(top_files)} files.")

    return top_files
