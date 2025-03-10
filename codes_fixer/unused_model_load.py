import os

def load70b():
    from vllm import LLM
    import torch

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    MAX_MODEL_LEN: int = 32_768
    MAX_NUM_SEQS: int = 6

    if os.getenv("KAGGLE_KERNEL_RUN_TYPE") or os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        llm_model_pth: str = "/kaggle/input/m/mtfall/deepseek-r1/transformers/deepseek-r1-distill-llama-70b-awq/1"
        num_gpus: int = 4
    else:
        llm_model_pth: str = "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ"
        num_gpus: int = torch.cuda.device_count()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))
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
    return None

def load7b():
    from vllm import LLM
    import torch
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    MAX_MODEL_LEN: int = 65_536
    MAX_NUM_SEQS: int = 6

    if os.getenv("KAGGLE_KERNEL_RUN_TYPE") or os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        llm_model_pth: str = "/kaggle/input/swe-fixer/transformers/swe-fixer-retriever-7b/1" #TODO: Change this to the correct model
        num_gpus: int = 4
    else:
        llm_model_pth: str = "internlm/SWE-Fixer-Retriever-7B"
        num_gpus: int = torch.cuda.device_count()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))

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
    return None


def load_model():
    """
        時間のかかる初回ロードをpredictの外で行うための処理
    """
    load70b()
    load7b()
    return None