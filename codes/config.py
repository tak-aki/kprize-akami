# config.py
import os
import warnings

import torch
from vllm import LLM

REPO_PATH: str = "repo"

warnings.simplefilter("ignore")


## Initialize LLM
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if os.getenv("KAGGLE_KERNEL_RUN_TYPE") or os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    llm_model_pth: str = "/kaggle/input/m/mtfall/deepseek-r1/transformers/deepseek-r1-distill-llama-70b-awq/1"
    num_gpus: int = 4
else:
    llm_model_pth: str = "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ"
    num_gpus: int = torch.cuda.device_count()

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))

BATCH_SIZE: int = 6
VALIDATION_COPY_COUNT: int = 1
MAX_TOKENS: int = 8192

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
    cpu_offload_gb=16,
    seed=2024,
)

tokenizer = llm.get_tokenizer()
