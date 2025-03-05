# classify_difficulty.py
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd

from vllm import RequestOutput, SamplingParams

from .config import BATCH_SIZE, MAX_TOKENS, llm, tokenizer
from .utils import count_tokens

reading_prompt = """
You are an expert software engineer and AI assistant tasked with evaluating GitHub issues based on their clarity and difficulty. 

This is the problem statement.

{problem_statement}

This is the file directory

<directory>
{directory_string}
</directory>

## Task
Analyze it in detail and classify it into the following two categories:

1. Well-Specified:
If the problem statement is poorly specified, it can be significantly harder, or in some cases impossible, to generate a patch that solves the problem. 
You need to label the problem statement with these 4 possible labels:
- 1: The issue is well-specified and it is clear what is required for a successful solution.
- 2: There are some blanks to fill in about the issue, but there is a sensible interpretation of what is required for a successful solution.
- 3: The issue is vague and there is room for ambiguity. It is unclear what a successful solution would look like.
- 4: It is almost impossible to understand what you are being asked to do without further information.

2. Difficulty:
You need to estimate how much time it would take an experienced software engineer who has had a few hours to familiarize themselves with the codebase to write a patch solving the issue. 
There are 4 possible labels for difficulty:
- 1: <15 min fix (e.g., a trivial change adding some assertions to a function)
- 2: 15 min–1 hour (e.g., a small change that requires a bit of thought)
- 3: 1–4 hours (e.g., substantially rewriting a function or editing multiple files)
- 4: >4 hours (e.g., a very esoteric issue that clearly requires a substantial amount of research to fix, changing >100 lines of code)

## Output Format:
Provide your thought and then classification result in a structured XML format as follows:

<root>
    <well_specified>2</well_specified>
    <difficulty>3</difficulty>
</root>

Notes:
- Make sure to encode each classification result between <root> and </root>
""".strip()

def safe_int(s):
    if isinstance(s, str) and s.isdigit():
        return int(s)
    return None

def extract_classification(xml_content: str) -> Dict[str, List[str]]:
    import xml.etree.ElementTree as ET

    parsed_data: Dict[str, List[str]] = {}
    pattern: str = r"<root>(.*?)</root>"
    matches: List[str] = re.findall(pattern, xml_content, re.DOTALL)
    for match in matches:
        try:
            root = ET.fromstring("<root>" + match + "</root>")
            well_specified = root.find("well_specified")
            well_specified_classification: Optional[str] = (
                well_specified.text.strip() if well_specified is not None and well_specified.text is not None else None
            )
            difficulty = root.find("difficulty")
            difficulty_classification: Optional[str] = (
                difficulty.text.strip() if difficulty is not None and difficulty.text is not None else None
            )
            parsed_data["well_specified"] = safe_int(well_specified_classification)
            parsed_data["difficulty"] = safe_int(difficulty_classification)
        except Exception as e:
            print("Error parsing output", e)
            print(xml_content)
            return {}
    return parsed_data

def classify_difficulty(directory_string: str, problem_statement: str) -> Tuple[List[str], List[Dict[str, List[str]]]]:
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
    print("classify_difficulty", [count_tokens(text, tokenizer) for text in prompt_texts])
    request_outputs: List[RequestOutput] = llm.generate(prompt_texts, sampling_params=sampling_params)
    if not request_outputs:
        return [], []
    response_texts = [output.outputs[0].text for output in request_outputs]
    print("classify_difficulty", [count_tokens(text, tokenizer) for text in response_texts])
    completion_texts = [pt + rt for pt, rt in zip(prompt_texts, response_texts)]
    classification_result = [extract_classification(rt) for rt in response_texts]
    print("classification_result", classification_result)
    return completion_texts, classification_result

def skip_judge(
        classification_result: Dict[str, List[str]], 
        difficulty_threshold: float=2.5, 
        well_specified_threchold: float=1.5
        ) -> bool:
    result_mean = pd.DataFrame(classification_result).mean()
    return result_mean["difficulty"] > difficulty_threshold or result_mean["well_specified"] > well_specified_threchold