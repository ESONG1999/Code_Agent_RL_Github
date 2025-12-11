#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_dpo_lora.py

DPO (Rafailov et al., 2023):
    L = -E[ log σ( β * ( (logπθ(y⁺|x) - logπθ(y⁻|x))
                        - (logπ_ref(y⁺|x) - logπ_ref(y⁻|x)) ) ) ]
"""

import os
import json
import argparse
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftConfig, PeftModel
from tqdm import tqdm



class DpoPairDataset(Dataset):

    def __init__(
        self,
        path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
    ):
        self.data: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

        print(f"[INFO] Loaded {len(self.data)} DPO pairs from {path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.data[idx]
        prompt = ex["prompt"]
        prompt_len_tokens = int(ex["prompt_len_tokens"])
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        # prompt + chosen
        chosen_text = prompt.rstrip() + "\n\n" + chosen.rstrip("\n")
        chosen_enc = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        chosen_ids = chosen_enc["input_ids"][0]
        chosen_attn = chosen_enc["attention_mask"][0]

        # prompt + rejected
        rejected_text = prompt.rstrip() + "\n\n" + rejected.rstrip("\n")
        rejected_enc = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        rejected_ids = rejected_enc["input_ids"][0]
        rejected_attn = rejected_enc["attention_mask"][0]

        return {
            "prompt": prompt,
            "prompt_len_tokens": prompt_len_tokens,
            "chosen_ids": chosen_ids,
            "chosen_attn": chosen_attn,
            "rejected_ids": rejected_ids,
            "rejected_attn": rejected_attn,
            "logp_ref_chosen": float(ex["logp_ref_chosen"]),
            "logp_ref_rejected": float(ex["logp_ref_rejected"]),
        }


def collate_dpo(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    max_len_chosen = max(x["chosen_ids"].size(0) for x in batch)
    max_len_rejected = max(x["rejected_ids"].size(0) for x in batch)

    chosen_ids_list = []
    chosen_attn_list = []
    rejected_ids_list = []
    rejected_attn_list = []
    prompt_lens = []
    logp_ref_chosen = []
    logp_ref_rejected = []

    for ex in batch:
        ids_c = ex["chosen_ids"]
        attn_c = ex["chosen_attn"]
        pad_c = max_len_chosen - ids_c.size(0)
        if pad_c > 0:
            ids_c = torch.cat(
                [ids_c, ids_c.new_full((pad_c,), pad_token_id)],
                dim=0,
            )
            attn_c = torch.cat(
                [attn_c, attn_c.new_zeros((pad_c,))],
                dim=0,
            )

        ids_r = ex["rejected_ids"]
        attn_r = ex["rejected_attn"]
        pad_r = max_len_rejected - ids_r.size(0)
        if pad_r > 0:
            ids_r = torch.cat(
                [ids_r, ids_r.new_full((pad_r,), pad_token_id)],
                dim=0,
            )
            attn_r = torch.cat(
                [attn_r, attn_r.new_zeros((pad_r,))],
                dim=0,
            )

        chosen_ids_list.append(ids_c)
        chosen_attn_list.append(attn_c)
        rejected_ids_list.append(ids_r)
        rejected_attn_list.append(attn_r)

        prompt_lens.append(ex["prompt_len_tokens"])
        logp_ref_chosen.append(ex["logp_ref_chosen"])
        logp_ref_rejected.append(ex["logp_ref_rejected"])

    batch_dict = {
        "chosen_input_ids": torch.stack(chosen_ids_list, dim=0),        # [B, Tc]
        "chosen_attention_mask": torch.stack(chosen_attn_list, dim=0),  # [B, Tc]
        "rejected_input_ids": torch.stack(rejected_ids_list, dim=0),    # [B, Tr]
        "rejected_attention_mask": torch.stack(rejected_attn_list, dim=0),
        "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long),     # [B]
        "logp_ref_chosen": torch.tensor(logp_ref_chosen, dtype=torch.float32),
        "logp_ref_rejected": torch.tensor(logp_ref_rejected, dtype=torch.float32),
    }
    return batch_dict

def compute_logp_sum(
    model: PeftModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits                              # [B, T, V]
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [B, T-1, V]
    target_ids = input_ids[:, 1:]                        # [B, T-1]

    B, T_minus1 = target_ids.shape
    logp_sums: List[torch.Tensor] = []

    for i in range(B):
        pl = int(prompt_lens[i].item())
        total_len = int(attention_mask[i].sum().item())
        comp_len = total_len - pl
        if comp_len <= 0:
            logp_sums.append(torch.tensor(0.0, device=input_ids.device))
            continue

        start_idx = pl - 1
        end_idx = min(pl - 1 + comp_len, T_minus1)

        lp_slice = logprobs[i, start_idx:end_idx, :]     # [comp_len, V]
        tid_slice = target_ids[i, start_idx:end_idx]     # [comp_len]

        selected = lp_slice.gather(
            dim=-1,
            index=tid_slice.unsqueeze(-1),
        ).squeeze(-1)                                    # [comp_len]

        logp_sums.append(selected.sum())

    return torch.stack(logp_sums, dim=0)                 # [B]



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dpo_data_path",
        type=str,
        default="data/humaneval_dpo_pairs_correct_only.jsonl",
    )
    parser.add_argument(
        "--sft_model",
        type=str,
        default="checkpoints/deepseek_1_3b_lora_sft_v2",
        help="用作 policy 初始化和 reference logp 的 SFT LoRA checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/deepseek_1_3b_lora_dpo",
    )
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--beta", type=float, default=0.05, help="DPO 温度 β")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()



def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    peft_config = PeftConfig.from_pretrained(args.sft_model)
    base_model_name = peft_config.base_model_name_or_path
    print("[INFO] Base model:", base_model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
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
        base_model_name,
        quantization_config=bnb_config,
        device_map={"": 0} if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False

    model = PeftModel.from_pretrained(
        base_model,
        args.sft_model,
        is_trainable=True,
    )
    model.train()
    print("[INFO] LoRA parameters trainable:")
    model.print_trainable_parameters()

    dataset = DpoPairDataset(
        path=args.dpo_data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_dpo(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )
    log_sigmoid = nn.LogSigmoid()

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_losses = []

        for batch in tqdm(dataloader, desc=f"DPO epoch {epoch}"):
            chosen_ids = batch["chosen_input_ids"].to(device)          # [B, Tc]
            chosen_attn = batch["chosen_attention_mask"].to(device)    # [B, Tc]
            rejected_ids = batch["rejected_input_ids"].to(device)      # [B, Tr]
            rejected_attn = batch["rejected_attention_mask"].to(device)
            prompt_lens = batch["prompt_lens"].to(device)              # [B]
            logp_ref_chosen = batch["logp_ref_chosen"].to(device)      # [B]
            logp_ref_rejected = batch["logp_ref_rejected"].to(device)  # [B]

            logp_chosen = compute_logp_sum(
                model,
                chosen_ids,
                chosen_attn,
                prompt_lens,
            )  # [B]
            logp_rejected = compute_logp_sum(
                model,
                rejected_ids,
                rejected_attn,
                prompt_lens,
            )  # [B]

            # Δθ & Δref
            delta_theta = logp_chosen - logp_rejected
            delta_ref = logp_ref_chosen - logp_ref_rejected

            # DPO logits: β * (Δθ - Δref)
            logits = args.beta * (delta_theta - delta_ref)

            # loss = -E[ log σ(logits) ]
            loss = -log_sigmoid(logits).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.detach().cpu().item())

        avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        print(f"[DPO] epoch {epoch} avg_loss = {avg_loss:.4f}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[INFO] DPO LoRA model saved to", args.output_dir)


if __name__ == "__main__":
    main()
