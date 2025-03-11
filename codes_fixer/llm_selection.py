# selection.py
from typing import Dict, List, Tuple, Optional
import os
import re

import torch
from vllm import RequestOutput, SamplingParams, LLM

from .config import BATCH_SIZE, model_32b 
from .utils import count_tokens 

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

Return the strings to search in this format

(explanation)

<root>
    <entry>
        <filepath>filepath</filepath>  
    </entry>
    <entry>
        <filepath>filepath</filepath>
    </entry>
    ...
</root>
...

Notes:
- Make sure to encode each entry between <root> and </root>
- Return the FULL filepath - exactly as specified in <directory> and </directory>
    - Example: <filepath>repo/path/to/directory/file.py</filepath>
- Do not include test files in search
""".strip()

def extract_file_path(xml_content: str) -> Dict[str, List[str]]:
    import xml.etree.ElementTree as ET

    parsed_data: List[str] = []
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
                if filepath_text:
                    parsed_data.append(filepath_text)
        except Exception as e:
            logger.info(f"Error parsing output {e}")
            logger.info(xml_content)
            return [] 
    return parsed_data

def get_llm_selection(directory_string: str, problem_statement: str) -> Tuple[List[str], List[Dict[str, List[str]]]]:
    MAX_TOKENS: int = 4096
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
        model_32b["tokenizer"].apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True) + "<think>\n"
        for messages in list_of_messages
    ]
    logger.info(f"prompt_texts token length: {[count_tokens(text, model_32b['tokenizer']) for text in prompt_texts]}")
    request_outputs: List[RequestOutput] = model_32b["llm"].generate(prompt_texts, sampling_params=sampling_params)
    if not request_outputs:
        return [], []
    response_texts = [output.outputs[0].text for output in request_outputs]
    logger.info(f"response_texts token length: {[count_tokens(text, model_32b['tokenizer']) for text in response_texts]}")
    completion_texts = [pt + rt for pt, rt in zip(prompt_texts, response_texts)]
    extracted_files = [extract_file_path(rt) for rt in response_texts]
    return completion_texts, extracted_files
