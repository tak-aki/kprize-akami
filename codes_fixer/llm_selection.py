# selection.py
from typing import Dict, List, Tuple, Optional
import os
import re

import torch
from vllm import RequestOutput, SamplingParams, LLM

from .config import BATCH_SIZE 
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

    ## Initialize LLM
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if os.getenv("KAGGLE_KERNEL_RUN_TYPE") or os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        llm_model_pth: str = "/kaggle/input/m/mtfall/deepseek-r1/transformers/deepseek-r1-distill-llama-70b-awq/1"
        num_gpus: int = 4
    else:
        llm_model_pth: str = "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ"
        num_gpus: int = torch.cuda.device_count()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))

    MAX_TOKENS: int = 4096

    MAX_NUM_SEQS: int = 6
    MAX_MODEL_LEN: int = 32_768


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
    logger.info(f"prompt_texts token length: {[count_tokens(text, tokenizer) for text in prompt_texts]}")
    request_outputs: List[RequestOutput] = llm.generate(prompt_texts, sampling_params=sampling_params)
    if not request_outputs:
        return [], []
    response_texts = [output.outputs[0].text for output in request_outputs]
    logger.info(f"response_texts token length: {[count_tokens(text, tokenizer) for text in response_texts]}")
    completion_texts = [pt + rt for pt, rt in zip(prompt_texts, response_texts)]
    extracted_files = [extract_file_path(rt) for rt in response_texts]
    return completion_texts, extracted_files
