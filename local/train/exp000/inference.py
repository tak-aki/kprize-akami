import json
import logging
import math
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import hydra
import numpy as np
import pandas as pd
import torch
import vllm
from datasets import Dataset
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from transformers import AutoTokenizer, LogitsProcessor
from vllm import LLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

from local.src.utils import set_seed

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from .run import setup_swe_verified_data

CANDIDATE = ["1", "2", "3"]

# ログ設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
logger.propagate = False


####################
# Config定義
####################
@dataclass
class ExpConfig:
    debug: bool = False
    seed: int = 1029
    # 推論時に評価するfold（Stratified K-Foldの設定に合わせる）
    folds: List[int] = field(default_factory=lambda: [0])

    model_name: str = "inarikami/DeepSeek-R1-Distill-Qwen-32B-AWQ"
    lora_checkpoint: str = "output_train/exp000/default/fold0/checkpoint-25/adapter_model"
    max_length: int = 3072
    lora_r: int = 16
    prompt_template: str = textwrap.dedent("""
        Below is a GitHub-related issue.

        {problem_statement}

        Please review the problem statement and estimate how long it will take to resolve the issue. Choose the appropriate number from the options below:
            1.	<15 min fix
            2.	15 min - 1 hour
            3.	>1 hours

        Answer only the number (1, 2, or 3).
        """)


@dataclass
class EnvConfig:
    exp_output_dir: str = "output_infer"


@dataclass
class Config:
    env: EnvConfig = EnvConfig()
    exp: ExpConfig = ExpConfig()


cs = ConfigStore.instance()
cs.store(name="default", group="env", node=EnvConfig)
cs.store(name="default", group="exp", node=ExpConfig)


####################
# データセット準備
####################
def setup_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    return tokenizer


####################
# 推論用プロンプト作成
####################
def make_inference_prompt(cfg: Config, row, tokenizer):
    """
    学習時のプロンプトテンプレートを利用し、
    正解を付加しない「Answer:」までのプロンプトを作成します。
    """
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": cfg.exp.prompt_template.format(problem_statement=row["problem_statement"])},
    ]
    # 事前に定義したテンプレートをもとに、モデルの入力形式に合わせた文字列を作成
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not prompt.endswith("Answer:"):
        prompt += "Answer:"
    return prompt


def generate_text_vllm(cfg, prompt_text_list, tokenizer, lora_path):
    KEEP = []
    for x in CANDIDATE:
        c = tokenizer.encode(x, add_special_tokens=False)[0]
        KEEP.append(c)
    print(f"Force predictions to be tokens {KEEP} which are {CANDIDATE}.")

    class DigitLogitsProcessor(LogitsProcessor):
        def __init__(self, tokenizer):
            self.allowed_ids = KEEP

        def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
            scores[self.allowed_ids] += 100
            return scores

    logits_processors = [DigitLogitsProcessor(tokenizer)]
    llm = vllm.LLM(
        cfg.exp.model_name,
        quantization="awq" if "awq" in cfg.exp.model_name.lower() else None,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.80,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=4096,  # if "7b" in cfg.exp.model_name.lower() else (5840 if IS_KAGGLE else 8096),
        disable_log_stats=True,
        # enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=cfg.exp.lora_r,
    )
    sampling_params = vllm.SamplingParams(
        n=1,  # Number of output sequences to return for each prompt.
        top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
        temperature=0,  # randomness of the sampling
        seed=777,  # Seed for reprodicibility
        skip_special_tokens=True,  # Whether to skip special tokens in the output.
        max_tokens=1,  # Maximum number of tokens to generate per output sequence.
        logits_processors=logits_processors,
        logprobs=5,
    )
    responses = llm.generate(
        prompt_text_list,
        sampling_params=sampling_params,
        use_tqdm=True,
        lora_request=LoRARequest("sql_adapter", 1, lora_path),
    )

    results = []
    errors = 0

    for i, response in enumerate(responses):
        try:
            x = response.outputs[0].logprobs[0]
            logprobs = []
            for k in KEEP:
                if k in x:
                    logprobs.append(math.exp(x[k].logprob))
                else:
                    logprobs.append(0)
                    print(f"bad logits {i}")
            logprobs = np.array(logprobs)
            logprobs /= logprobs.sum()
            results.append(logprobs)
        except:
            # print(f"error {i}")
            results.append(np.array([0.25, 0.25, 0.25]))
            errors += 1
    pred_array = np.array(results).reshape((-1, 3))

    return pred_array


def main(cfg: Config) -> None:
    set_seed(cfg.exp.seed)

    exp_name = f"{Path(sys.argv[0]).parent.name}/{HydraConfig.get().runtime.choices.exp}"  # e.g. 000_sample/default
    output_dir = Path(cfg.env.exp_output_dir) / exp_name
    os.makedirs(output_dir, exist_ok=True)
    print(f"output_dir: {output_dir}")

    swe_bench_data = setup_swe_verified_data()
    df = pd.DataFrame(swe_bench_data)

    # difficultyでstratified k-fold
    df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=1029).split(df, df.difficulty)
    ):
        df.loc[val_idx, "fold"] = fold_idx

    # 推論処理

    # 学習時と同様のdifficulty->ラベルのマッピング（0,1,2）
    df["y_label"] = df["difficulty"].map({"<15 min fix": 1, "15 min - 1 hour": 2, "1-4 hours": 3, ">4 hours": 3})
    tokenizer = setup_tokenizer(cfg.exp.model_name)
    df["prompt"] = df.apply(lambda row: make_inference_prompt(cfg, row, tokenizer), axis=1)

    for fold in cfg.exp.folds:
        val_df = df[df["fold"] == fold].reset_index(drop=True)
        logger.info(f"Number of validation samples: {len(val_df)}")

        # 候補トークン「1,2,3」のtoken idを取得
        candidate_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in CANDIDATE]
        logger.info(f"Candidate token IDs: {candidate_ids}")

        scores = generate_text_vllm(cfg, val_df["prompt"], tokenizer, lora_path=cfg.exp.lora_checkpoint)

        # 結果をデータフレームに追加
        val_df["pred_label"] = np.argmax(scores, axis=1)

        y_label_zero = val_df["y_label"] - 1

        # 精度評価
        accuracy = (val_df["pred_label"] == y_label_zero).mean()
        logger.info(f"Validation Accuracy: {accuracy:.4f}")
        cm = confusion_matrix(y_label_zero, val_df["pred_label"])
        logger.info(f"Confusion Matrix:\n{cm}")

        # 結果の保存
        result_file = output_dir / "inference_results.csv"
        val_df.to_csv(result_file, index=False)
        logger.info(f"Results saved to {result_file}")


@hydra.main(version_base=None, config_path=".", config_name="config")
def run(cfg: Config) -> None:
    main(cfg)


if __name__ == "__main__":
    run()
