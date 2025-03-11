# config.py
import os
import warnings

import torch
from vllm import LLM

# REPO_PATH: str = "repo"

warnings.simplefilter("ignore")

## Environment Variables Setting
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if os.getenv("KAGGLE_KERNEL_RUN_TYPE") or os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    num_gpus: int = 4
else:
    num_gpus: int = torch.cuda.device_count()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))

## Parameters
BATCH_SIZE: int = 4
VALIDATION_COPY_COUNT: int = 1

## LLM Initialization
### 32B Model
model_32b = {}
if os.getenv("KAGGLE_KERNEL_RUN_TYPE") or os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    model_32b["llm_model_pth"] = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-32b-awq/1"
else:
    model_32b["llm_model_pth"] = "inarikami/DeepSeek-R1-Distill-Qwen-32B-AWQ"
model_32b["MAX_TOKENS"] = 8192
model_32b["MAX_NUM_SEQS"] = 4
model_32b["MAX_MODEL_LEN"] = 32_768
model_32b["llm"] = LLM(
    model=model_32b["llm_model_pth"],
    max_num_seqs=model_32b["MAX_NUM_SEQS"],  # Maximum number of sequences per iteration. Default is 256
    max_model_len=model_32b["MAX_MODEL_LEN"],  # Model context length
    trust_remote_code=True,  # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
    tensor_parallel_size=num_gpus,  # The number of GPUs to use for distributed execution with tensor parallelism
    gpu_memory_utilization=0.5,  # The ratio (between 0 and 1) of GPU memory to reserve for the model
    enable_prefix_caching=True, 
    enable_lora=True,
    max_lora_rank=32,
    seed=2024,
)
model_32b["tokenizer"] = model_32b["llm"].get_tokenizer()

### 7B Model
model_7b = {}
if os.getenv("KAGGLE_KERNEL_RUN_TYPE") or os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    model_7b["llm_model_pth"] = "/kaggle/input/swe-fixer/transformers/swe-fixer-retriever-7b/1"
else:
    model_7b["llm_model_pth"] = "internlm/SWE-Fixer-Retriever-7B"
model_7b["MAX_TOKENS"] = 2048
model_7b["MAX_NUM_SEQS"] = 4
model_7b["MAX_MODEL_LEN"] = 65_536
model_7b["llm"] = LLM(
    model=model_7b["llm_model_pth"],
    max_num_seqs=model_7b["MAX_NUM_SEQS"],  # Maximum number of sequences per iteration. Default is 256
    max_model_len=model_7b["MAX_MODEL_LEN"],  # Model context length
    trust_remote_code=True,  # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
    tensor_parallel_size=num_gpus,  # The number of GPUs to use for distributed execution with tensor parallelism
    gpu_memory_utilization=0.4,  # The ratio (between 0 and 1) of GPU memory to reserve for the model
    enable_prefix_caching=True, 
    seed=2024,
)
model_7b["tokenizer"] = model_7b["llm"].get_tokenizer()