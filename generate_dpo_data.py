#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
generate_dpo_data.py
{
  "task_id": ...,
  "prompt": ...,
  "prompt_len_tokens": int,
  "chosen": "...",          
  "rejected": "...",        
  "logp_ref_chosen": float,
  "logp_ref_rejected": float,
  "base_reward_chosen": 1,
  "base_reward_rejected": 0,
  "gen_len_chosen": int,
  "gen_len_rejected": int
}
"""

import os
import json
import argparse
import random
import signal
from typing import Dict, Any, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftConfig, PeftModel
from tqdm import tqdm


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError()


signal.signal(signal.SIGALRM, _timeout_handler)


def load_humaneval(path: str) -> List[Dict[str, Any]]:
    """读取 HumanEval 风格 JSONL 数据，每行一个 task。"""
    examples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def run_humaneval_tests(
    prompt: str,
    completion: str,
    test_code: str,
    entry_point: str,
    time_limit: int = 5,
) -> float:
    prompt = prompt.rstrip()
    completion = completion.rstrip("\n")
    test_code = test_code.rstrip()

    full_code = prompt + "\n\n" + completion + "\n\n" + test_code

    glb: Dict[str, Any] = {}
    try:
        signal.alarm(time_limit)
        exec(full_code, glb)
        candidate = glb[entry_point]
        check_fn = glb["check"]
        check_fn(candidate)
        signal.alarm(0)
        return 1.0
    except TimeoutError:
        signal.alarm(0)
        return 0.0
    except Exception:
        signal.alarm(0)
        return 0.0


# ========== 参数解析 ==========

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/humaneval_train.jsonl",
        help="用于生成 DPO 偏好对的 HumanEval 训练集",
    )
    parser.add_argument(
        "--sft_model",
        type=str,
        default="checkpoints/deepseek_1_3b_lora_sft_v2",
        help="SFT LoRA checkpoint 路径",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/humaneval_dpo_pairs_correct_only.jsonl",
        help="输出 DPO 偏好对 JSONL 路径",
    )
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--per_task_samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--time_limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

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
        is_trainable=False,
    )
    model.eval()

    examples = load_humaneval(args.train_path)
    print(f"[INFO] Loaded {len(examples)} train tasks from {args.train_path}")

    out_f = open(args.output_path, "w", encoding="utf-8")

    total_pairs = 0
    kept_tasks = 0

    for ex in tqdm(examples, desc="Generating DPO pairs"):
        task_id = ex["task_id"]
        prompt = ex["prompt"]
        test_code = ex["test"]
        entry_point = ex["entry_point"]

        prompt_enc = tokenizer(
            prompt.rstrip(),
            return_tensors="pt",
            truncation=True,
            max_length=args.max_prompt_length,
            add_special_tokens=True,
        )
        prompt_ids = prompt_enc["input_ids"][0].to(device)
        prompt_len_tokens = prompt_ids.size(0)

        samples_for_task: List[Dict[str, Any]] = []

        for _ in range(args.per_task_samples):
            inputs = {
                "input_ids": prompt_ids.unsqueeze(0),
                "attention_mask": torch.ones_like(prompt_ids).unsqueeze(0),
            }

            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            full_ids = gen_out[0]          # [T]
            total_len = full_ids.size(0)
            if total_len <= prompt_len_tokens:
                continue

            completion_ids = full_ids[prompt_len_tokens:]
            gen_len = completion_ids.size(0)
            completion_text = tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
            )

            base_reward = run_humaneval_tests(
                prompt=prompt,
                completion=completion_text,
                test_code=test_code,
                entry_point=entry_point,
                time_limit=args.time_limit,
            )

            with torch.no_grad():
                full_ids_batch = full_ids.unsqueeze(0).to(device)  # [1, T]
                attn_mask = torch.ones_like(full_ids_batch)        # [1, T]

                outputs = model(
                    input_ids=full_ids_batch,
                    attention_mask=attn_mask,
                )
                logits = outputs.logits  # [1, T, V]
                logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [1, T-1, V]
                target_ids = full_ids_batch[:, 1:]                       # [1, T-1]

                start_idx = prompt_len_tokens - 1
                end_idx = total_len - 1

                if end_idx <= start_idx:
                    continue

                logprobs_slice = logprobs[:, start_idx:end_idx, :]   # [1, comp_len, V]
                target_slice = target_ids[:, start_idx:end_idx]      # [1, comp_len]

                selected = logprobs_slice.gather(
                    dim=-1,
                    index=target_slice.unsqueeze(-1),
                ).squeeze(-1)                                       # [1, comp_len]
                logp_ref_sum = selected.sum().item()

            samples_for_task.append(
                {
                    "task_id": task_id,
                    "prompt": prompt,
                    "prompt_len_tokens": int(prompt_len_tokens),
                    "completion": completion_text,
                    "base_reward": float(base_reward),
                    "gen_len": int(gen_len),
                    "logp_ref": float(logp_ref_sum),
                }
            )

        if len(samples_for_task) == 0:
            continue

        pos = [s for s in samples_for_task if s["base_reward"] == 1.0]
        neg = [s for s in samples_for_task if s["base_reward"] == 0.0]

        if len(pos) == 0 or len(neg) == 0:
            continue

        pos_sorted = sorted(
            pos,
            key=lambda s: (s["gen_len"], -s["logp_ref"]),
        )
        chosen = pos_sorted[0]

        neg_sorted = sorted(
            neg,
            key=lambda s: (-s["gen_len"], s["logp_ref"]),
        )
        rejected = neg_sorted[0]

        record = {
            "task_id": task_id,
            "prompt": prompt,
            "prompt_len_tokens": int(chosen["prompt_len_tokens"]),
            "chosen": chosen["completion"],
            "rejected": rejected["completion"],
            "logp_ref_chosen": float(chosen["logp_ref"]),
            "logp_ref_rejected": float(rejected["logp_ref"]),
            "base_reward_chosen": float(chosen["base_reward"]),
            "base_reward_rejected": float(rejected["base_reward"]),
            "gen_len_chosen": int(chosen["gen_len"]),
            "gen_len_rejected": int(rejected["gen_len"]),
        }
        out_f.write(json.dumps(record) + "\n")
        total_pairs += 1
        kept_tasks += 1

    out_f.close()
    print(f"[INFO] Saved {total_pairs} DPO pairs from {kept_tasks} tasks to {args.output_path}")


if __name__ == "__main__":
    main()
