#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_sft_lora.py

Usage:
    python train_sft_lora.py \
        --train_path data/humaneval_train.jsonl \
        --model_name deepseek-ai/deepseek-coder-1.3b-base \
        --output_dir checkpoints/deepseek_1_3b_lora_sft
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


class HumanEvalSFTDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 1024,
    ):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        prompt = ex["prompt"]
        solution = ex["canonical_solution"]
        full_text = prompt.rstrip() + "\n\n" + solution.rstrip("\n")

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]

        prompt_ids = self.tokenizer(
            prompt.rstrip(),
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        prompt_len = len(prompt_ids)

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }



@dataclass
class DataCollatorForSFT:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        max_len = max(f["input_ids"].size(0) for f in features)

        def pad_tensor(x, pad_value):
            if x.size(0) == max_len:
                return x
            pad = x.new_full((max_len - x.size(0),), pad_value)
            return torch.cat([x, pad], dim=0)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            input_ids.append(pad_tensor(f["input_ids"], self.tokenizer.pad_token_id))
            attention_mask.append(pad_tensor(f["attention_mask"], 0))
            labels.append(pad_tensor(f["labels"], -100))

        batch["input_ids"] = torch.stack(input_ids, dim=0)
        batch["attention_mask"] = torch.stack(attention_mask, dim=0)
        batch["labels"] = torch.stack(labels, dim=0)

        return batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/humaneval_train.jsonl")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-coder-1.3b-base")
    parser.add_argument("--output_dir", type=str, default="checkpoints/deepseek_1_3b_lora_sft")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": 0} if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    base_model = prepare_model_for_kbit_training(base_model)
    base_model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_config)
    print("[INFO] LoRA parameters trainable:")
    model.print_trainable_parameters()

    dataset = HumanEvalSFTDataset(
        path=args.train_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    data_collator = DataCollatorForSFT(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=200, 
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=False,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        weight_decay=0.01,
        warmup_ratio=0.03,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("LoRA SFT finished, saved to", args.output_dir)


if __name__ == "__main__":
    main()
