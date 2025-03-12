# config.py
import os
import warnings

import torch

warnings.simplefilter("ignore")

# model_param = "70b"
model_param = "32b"

if os.getenv("KAGGLE_KERNEL_RUN_TYPE") or os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    num_gpus: int = 4
    if model_param == "70b":
        llm_model_path: str = "/kaggle/input/m/mtfall/deepseek-r1/transformers/deepseek-r1-distill-llama-70b-awq/1"
        difficulty_lora_path: str = (
                "/kaggle/input/kprize-akami-difficulty-model/output_train-exp004-70b_003-fold0-checkpoint-100"
            )
    elif model_param == "32b":
        llm_model_path: str = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-32b-awq/1"
        difficulty_lora_path: str = (
                "/kaggle/input/kprize-akami-difficulty-model/output_train-exp004-003-fold0-checkpoint-100"
            )
    retrieval_model_path = "/kaggle/input/swe-fixer/transformers/swe-fixer-retriever-7b/1"
else:
    num_gpus: int = torch.cuda.device_count()
    if model_param == "70b":
        llm_model_path: str = "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ"
        difficulty_lora_path: str = "output_train/output_train-exp004-70b_003-fold0-checkpoint-100"
    elif model_param == "32b":
        llm_model_path: str = "inarikami/DeepSeek-R1-Distill-Qwen-32B-AWQ"
        difficulty_lora_path: str = "output_train/output_train-exp004-003-fold0-checkpoint-100"
    retrieval_model_path = "internlm/SWE-Fixer-Retriever-7B"


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MAX_NUM_SEQS: int = 6
BATCH_SIZE: int = 6
VALIDATION_COPY_COUNT: int = 1
