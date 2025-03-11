import os
from io import StringIO
from typing import Dict, List, Tuple

from .config import REPO_PATH


def fetch_file_contents(files_to_search: Dict[str, List[str]], context_lines: int = 12, max_gap: int = 0) -> str:
    def find_lines_in_files_with_context(
        search_map: Dict[str, List[str]], context_lines: int = context_lines
    ) -> List[List[List[Tuple[int, str]]]]:
        """
        Given a dictionary mapping file paths to a list of search terms,
        open each file and gather *snippets* of lines that contain any
        of those search terms, including 'context_lines' before and after.

        Returns a list of lists:
        [
            [  # For file1
                [ (line_number, text), (line_number, text), ... ],
                [ ... ],
            ],
            [  # For file2
                ...
            ],
            ...
        ]
        """
        all_matches_per_file: List[List[List[Tuple[int, str]]]] = []

        for path, terms in search_map.items():
            if not os.path.isfile(path):
                # If the file is not found, record an empty list
                all_matches_per_file.append([])
                continue

            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            file_snippets: List[List[Tuple[int, str]]] = []
            num_lines: int = len(lines)

            for i, line in enumerate(lines, start=1):
                if any(t in line for t in terms):
                    start_idx: int = max(1, i - context_lines)
                    end_idx: int = min(num_lines, i + context_lines)
                    snippet: List[Tuple[int, str]] = []
                    for snippet_no in range(start_idx, end_idx + 1):
                        text_content: str = lines[snippet_no - 1].rstrip("\n")
                        snippet.append((snippet_no, text_content))
                    file_snippets.append(snippet)

            all_matches_per_file.append(file_snippets)

        return all_matches_per_file

    def merge_file_snippets(file_snippets: List[List[Tuple[int, str]]], gap: int = 0) -> List[List[Tuple[int, str]]]:
        """
        Merge overlapping or nearly adjacent snippets in a single file’s snippet list.
        """
        intervals: List[Tuple[int, int, List[Tuple[int, str]]]] = []
        for snippet in file_snippets:
            if snippet:
                start_line: int = snippet[0][0]
                end_line: int = snippet[-1][0]
                intervals.append((start_line, end_line, snippet))

        intervals.sort(key=lambda x: x[0])  # sort by start line

        merged: List[Tuple[int, int, List[Tuple[int, str]]]] = []
        for start, end, snippet in intervals:
            if not merged:
                merged.append((start, end, snippet))
                continue

            prev_start, prev_end, prev_snippet = merged[-1]
            if start <= prev_end + gap:
                new_end: int = max(end, prev_end)
                combined_dict: Dict[int, str] = {}
                for ln, txt in prev_snippet:
                    combined_dict[ln] = txt
                for ln, txt in snippet:
                    combined_dict[ln] = txt
                merged_snippet: List[Tuple[int, str]] = [(ln, combined_dict[ln]) for ln in sorted(combined_dict)]
                merged[-1] = (prev_start, new_end, merged_snippet)
            else:
                merged.append((start, end, snippet))

        # Extract just the merged snippet portion
        return [x[2] for x in merged]

    def merge_all_snippets(
        all_files_snips: List[List[List[Tuple[int, str]]]], gap: int = 0
    ) -> List[List[List[Tuple[int, str]]]]:
        """
        Merge snippet blocks within each file.
        all_files_snips is a list-of-lists:
            [
            [ snippetA, snippetB, ... ],  # file 1
            [ snippetC, snippetD, ... ],  # file 2
            ]
        """
        merged: List[List[List[Tuple[int, str]]]] = []
        for snips in all_files_snips:
            merged.append(merge_file_snippets(snips, gap=gap))
        return merged

    has_any_matches: bool = False

    # 1) Gather snippets around each match
    context_snippets: List[List[List[Tuple[int, str]]]] = find_lines_in_files_with_context(
        files_to_search, context_lines=context_lines
    )

    # 2) Merge overlapping snippets
    merged_snips: List[List[List[Tuple[int, str]]]] = merge_all_snippets(context_snippets, gap=max_gap)

    # 3) Build a string (instead of printing)
    output = StringIO()

    # Header
    output.write("Sample files created successfully.\n\n")
    output.write("Search Results (by file, merging any overlapping context):\n\n")

    # For each file
    for (filepath, terms), snippet_list in zip(files_to_search.items(), merged_snips):
        output.write(f"[file name]: {filepath[len(REPO_PATH) + 1 :]}\n")
        terms_searched_as_str = "\n".join(terms)
        output.write(f"[terms searched]:\n{terms_searched_as_str}\n")
        output.write("[file content begin]\n")
        if not snippet_list:
            output.write("  No matches found.\n")
        else:
            has_any_matches = True
            for snippet_idx, snippet in enumerate(snippet_list, start=1):
                snippet_start: int = snippet[0][0]
                snippet_end: int = snippet[-1][0]
                output.write(f"\nMatch #{snippet_idx}, lines {snippet_start} to {snippet_end}:\n")
                for line_no, text in snippet:
                    output.write(f"{line_no:3d} {text}\n")
                output.write("\n")
        output.write("[file content end]\n\n")

    file_content_string: str = output.getvalue()

    if has_any_matches:
        return file_content_string
    return ""
