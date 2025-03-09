import json
import logging
import os
import shutil
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import hydra
import numpy as np
import pandas as pd
from datasets import Dataset
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM

from local.src.setup import setup_data
from local.src.utils import set_seed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
logger.propagate = False


# 3クラス分類
DIFFICULTY2TOKEN = {
    "<15 min fix": "easy",
    "15 min - 1 hour": "medium",
    "1-4 hours": "difficult",
    ">4 hours": "difficult",
}
TOKENS = ["easy", "medium", "difficult"]
TOKEN2LABEL = {token: i for i, token in enumerate(TOKENS)}


####################
# Config 設定
####################
@dataclass
class ExpConfig:
    debug: bool = False
    seed: int = 1029
    folds: List[int] = field(default_factory=lambda: [0])
    model_name: str = "inarikami/DeepSeek-R1-Distill-Qwen-32B-AWQ"
    max_length: int = 3072

    lora_r: int = 16
    lora_alpha: float = lora_r * 2
    lora_dropout: float = 0.05
    lora_bias: str = "none"

    n_epochs: int = 2
    warmup_steps: int = 20
    optim_type: str = "adamw_torch_fused"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    lr: float = 1e-4

    prompt_template: str = textwrap.dedent("""
        Below is a GitHub-related issue.

        {problem_statement}

        Please review the problem statement and estimate how long it will take to resolve the issue. Choose the appropriate word from the options below:
            easy: <15 min fix
            medium: 15 min - 1 hour
            difficult: >1 hour

        Answer only with one word (easy, medium, or difficult).
        """)


@dataclass
class EnvConfig:
    exp_output_dir: str = "output_train"


@dataclass
class Config:
    env: EnvConfig = EnvConfig()
    exp: ExpConfig = ExpConfig()


# hydra用にdefaultを設定
cs = ConfigStore.instance()
cs.store(name="default", group="env", node=EnvConfig)
cs.store(name="default", group="exp", node=ExpConfig)


####################
# 実験用コード
####################
def setup_swe_verified_data():
    split = "test"  # train, test, dev
    dataset_name = "princeton-nlp/SWE-bench_Verified"
    data_dir = Path("input/") / dataset_name.split("/")[-1].lower()
    cache_dir = data_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir = data_dir / "processed"
    data_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Setting up the dataset.")
    swe_bench_data = setup_data(cache_dir, data_output_dir, dataset_name=dataset_name, split=split)
    return swe_bench_data


def setup_model_and_tokenizer(exp_cfg: ExpConfig):
    """
    Set up the model and tokenizer for training.
    """
    tokenizer = AutoTokenizer.from_pretrained(exp_cfg.model_name)
    tokenizer.padding_side = "left"

    peft_config = LoraConfig(
        r=exp_cfg.lora_r,
        lora_alpha=exp_cfg.lora_alpha,
        lora_dropout=exp_cfg.lora_dropout,
        bias=exp_cfg.lora_bias,
        inference_mode=False,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = AutoModelForCausalLM.from_pretrained(
        exp_cfg.model_name,
        # device_map="cpu",
        pad_token_id=tokenizer.pad_token_id,
    )
    model.config.use_cache = False
    # model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


class CustomTokenizer:
    """
    Custom tokenizer wrapper for batch processing.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int, is_train: bool = True) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

    def __call__(self, batch: dict) -> dict:
        tokenized = self.tokenizer(batch["prompt"], max_length=self.max_length, truncation=True)
        if self.is_train:
            labels = batch["y_label"]
            return {**tokenized, "labels": labels}
        else:
            return {**tokenized}


def prepare_datasets(train: pd.DataFrame, tokenizer, cfg: Config, fold_idx: int):
    train_df = train[train.fold != fold_idx]
    if cfg.exp.debug:
        train_df = train_df.head()
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(train[(train.fold == fold_idx)])

    encode = CustomTokenizer(tokenizer, max_length=cfg.exp.max_length)

    train_ds = train_ds.map(encode, batched=True)
    val_ds = val_ds.map(encode, batched=True)

    return train_ds, val_ds


def setup_trainer(model, tokenizer, train_ds, val_ds, output_dir, cfg: Config, fold_idx: int):
    """
    Set up the Trainer for model training.
    """
    output_dir_fold = os.path.join(output_dir, f"fold{fold_idx}")
    data_collator = DataCollatorForCompletionOnlyLM("Answer:", tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir_fold,
        overwrite_output_dir=False,
        num_train_epochs=cfg.exp.n_epochs,
        per_device_train_batch_size=cfg.exp.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.exp.gradient_accumulation_steps,
        gradient_checkpointing=True,
        save_total_limit=None,
        save_strategy="epoch",
        logging_strategy="epoch",
        optim=cfg.exp.optim_type,
        fp16=True,
        learning_rate=cfg.exp.lr,
        warmup_steps=cfg.exp.warmup_steps,
        greater_is_better=True,
        report_to="none",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=cfg.exp.seed,
    )

    return Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        data_collator=data_collator,
    )


def make_prompt(cfg: Config, row, tokenizer):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {
            "role": "user",
            "content": cfg.exp.prompt_template.format(
                problem_statement=row["problem_statement"],
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def preprocess_df(df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase) -> pd.DataFrame:
    def preprocess_row(row: pd.Series, tokenizer: PreTrainedTokenizerBase) -> dict:
        item = tokenizer(row["prompt"], add_special_tokens=False, truncation=False)
        return item

    items = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        items.append(preprocess_row(row, tokenizer))

    df = pd.concat([df, pd.DataFrame(items)], axis=1)
    return df


def train(cfg: Config, output_dir: Path, df: pd.DataFrame):
    tokenizer = AutoTokenizer.from_pretrained(cfg.exp.model_name)

    df["y_word"] = df["difficulty"].map(DIFFICULTY2TOKEN)
    df["y_label"] = df["y_word"].map(TOKEN2LABEL)
    df["prompt"] = df.apply(lambda row: make_prompt(cfg, row, tokenizer), axis=1)
    df["prompt"] = df["prompt"] + "Answer:" + df["y_word"].astype(str)
    df_processed = preprocess_df(df, tokenizer)

    print(df_processed["input_ids"].apply(len).describe())

    for fold in cfg.exp.folds:
        model, tokenizer = setup_model_and_tokenizer(cfg.exp)
        train_ds, val_ds = prepare_datasets(df_processed, tokenizer, cfg, fold)
        trainer = setup_trainer(model, tokenizer, train_ds, val_ds, output_dir, cfg, fold)
        trainer.train()


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: Config) -> None:  # Duck typing: cfgは実際にはDictConfigだが、Configクラスのように扱える
    print(cfg)
    set_seed(cfg.exp.seed)

    exp_name = f"{Path(sys.argv[0]).parent.name}/{HydraConfig.get().runtime.choices.exp}"  # e.g. 000_sample/default
    output_dir = Path(cfg.env.exp_output_dir) / exp_name
    os.makedirs(output_dir, exist_ok=True)
    print(f"output_dir: {output_dir}")

    swe_bench_data = setup_swe_verified_data()
    df = pd.DataFrame(swe_bench_data)

    rng = np.random.default_rng(1029)
    valid_ids = rng.choice(np.arange(500), 100, replace=False)
    df["fold"] = 1
    df.loc[df.index.isin(valid_ids), "fold"] = 0

    train(cfg, output_dir, df)


if __name__ == "__main__":
    main()
