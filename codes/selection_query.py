# selection.py
from typing import Dict, List, Tuple

from vllm import RequestOutput, SamplingParams

from .config import BATCH_SIZE, MAX_TOKENS, llm, tokenizer
from .utils import count_tokens, extract_file_query

reading_prompt = """
You will be implementing a git diff patch to solve an issue with the code repository.
You will first need to select files in the file directory.

This is the problem statement.

{problem_statement}

This is the file directory

<directory>
{directory_string}
</directory>

Which files should be inspected so that we can solve the problem?
When we inspect each file, what strings should be searched?

Return the strings to search in this format

(explanation)

<root>
    <entry>
        <filepath>filepath</filepath>  
        <strings_to_search>
            <string_to_search>string_to_search</string_to_search>
            ...
            <string_to_search>string_to_search</string_to_search>
        </strings_to_search>
    </entry>
    <entry>
        <filepath>filepath</filepath>
        <strings_to_search>
            <string_to_search>string_to_search</string_to_search>
            ...
            <string_to_search>string_to_search</string_to_search>
        </strings_to_search>
    </entry>
    ...
</root>
...

Notes:
- Make sure to encode each entry between <root> and </root>
- Return the FULL filepath - exactly as specified in <directory> and </directory>
    - Example: <filepath>repo/path/to/directory/file.py</filepath>
- If you are searching for a word instead of a substring, maybe add spaces or brackets before and after the string
    - For example, if you are searching for uses of the function `calculate`, use ` calculate(` as the search string instead of `calculate`
- Prefer searching longer strings
    - Avoid searching for strings that might appear in many parts of the codebase
- Search the test files as well to understand the feature behavior
    - Also search for the relevant function calls in the test files
""".strip()


def get_selection_query(directory_string: str, problem_statement: str) -> Tuple[List[str], List[Dict[str, List[str]]]]:
    sampling_params = SamplingParams(
        temperature=0.6,
        min_p=0.01,
        skip_special_tokens=True,
        max_tokens=MAX_TOKENS,
    )
    list_of_messages = [
        [
            {
                "role": "user",
                "content": reading_prompt.format(
                    problem_statement=problem_statement[:20_000],
                    directory_string=directory_string[:30_000],
                ),
            },
        ]
        for _ in range(BATCH_SIZE)
    ]
    prompt_texts = [
        tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True) + "<think>\n"
        for messages in list_of_messages
    ]
    print("get_selection_query", [count_tokens(text, tokenizer) for text in prompt_texts])
    request_outputs: List[RequestOutput] = llm.generate(prompt_texts, sampling_params=sampling_params)
    if not request_outputs:
        return [], []
    response_texts = [output.outputs[0].text for output in request_outputs]
    print("get_selection_query", [count_tokens(text, tokenizer) for text in response_texts])
    completion_texts = [pt + rt for pt, rt in zip(prompt_texts, response_texts)]
    file_queries = [extract_file_query(rt) for rt in response_texts]
    return completion_texts, file_queries
