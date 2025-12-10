#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_rl_lora.py

Usage:
    python train_rl_lora.py \
        --train_path data/humaneval_train.jsonl \
        --sft_model checkpoints/deepseek_1_3b_lora_sft \
        --output_dir checkpoints/deepseek_1_3b_lora_rl_alpha0.01 \
        --length_penalty_alpha 0.01
"""

import os
import json
import argparse
import random
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

from peft import PeftConfig, PeftModel

def load_humaneval(path: str) -> List[Dict[str, Any]]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def run_humaneval_tests(
    prompt: str,
    completion: str,
    test_code: str,
    entry_point: str,
) -> float:
    full_code = prompt.strip() + "\n\n" + completion.strip() + "\n\n" + test_code.strip()

    glb: Dict[str, Any] = {}
    try:
        exec(full_code, glb)
        candidate = glb[entry_point]
        check_fn = glb["check"]
        check_fn(candidate)
        return 1.0
    except Exception:
        return 0.0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/humaneval_train.jsonl")
    parser.add_argument(
        "--sft_model",
        type=str,
        required=True,
        help="Path to LoRA SFT checkpoint (e.g., checkpoints/deepseek_1_3b_lora_sft)",
    )
    parser.add_argument("--output_dir", type=str, default="checkpoints/deepseek_1_3b_lora_rl")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--length_penalty_alpha", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    peft_config = PeftConfig.from_pretrained(args.sft_model)
    base_model_name = peft_config.base_model_name_or_path
    print("Base model:", base_model_name)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    examples = load_humaneval(args.train_path)

    log_path = os.path.join(args.output_dir, "rl_train_log.jsonl")
    log_f = open(log_path, "w", encoding="utf-8")

    for step in tqdm(range(1, args.num_steps + 1), desc="RL-LoRA steps"):
        ex = random.choice(examples)
        prompt = ex["prompt"]
        test_code = ex["test"]
        entry_point = ex["entry_point"]

        model.eval()
        with torch.no_grad():
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length - args.max_new_tokens,
            ).to(device)
            prompt_ids = inputs["input_ids"][0]
            prompt_len = prompt_ids.size(0)

            gen_out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        full_ids = gen_out[0]      # [T]
        completion_ids = full_ids[prompt_len:]
        if completion_ids.numel() == 0:
            continue

        completion_text = tokenizer.decode(
            completion_ids,
            skip_special_tokens=True,
        )

        base_reward = run_humaneval_tests(
            prompt=prompt,
            completion=completion_text,
            test_code=test_code,
            entry_point=entry_point,
        )
        gen_len = completion_ids.numel()
        length_penalty = args.length_penalty_alpha * (gen_len / 100.0)
        reward_value = base_reward - length_penalty

        model.train() 

        full_ids = full_ids.unsqueeze(0).to(device)      # [1, T]
        attention_mask = torch.ones_like(full_ids, device=device)

        outputs = model(
            input_ids=full_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # [1, T, V]
        logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [1, T-1, V]

        target_ids = full_ids[:, 1:]  # [1, T-1]

        total_len = full_ids.size(1)
        gen_len = total_len - prompt_len
        if gen_len <= 0:
            continue

        start_idx = prompt_len - 1
        end_idx = total_len - 1

        logprobs_slice = logprobs[:, start_idx:end_idx, :]  # [1, gen_len, V]
        target_slice = target_ids[:, start_idx:end_idx]     # [1, gen_len]

        selected_logprobs = logprobs_slice.gather(
            dim=-1,
            index=target_slice.unsqueeze(-1),
        ).squeeze(-1)  # [1, gen_len]

        # sum over time
        logp = selected_logprobs.sum(dim=1)  # [1], requires_grad=True

        reward = torch.tensor(
            reward_value,
            dtype=logp.dtype,
            device=logp.device,
        )

        loss = -(reward * logp).mean()

        if not loss.requires_grad:
            raise RuntimeError("loss does not require grad, something is wrong with the graph.")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % args.log_every == 0:
            log_line = {
                "step": step,
                "task_id": ex["task_id"],
                "base_reward": float(base_reward),
                "length_penalty": float(length_penalty),
                "reward": float(reward_value),
                "gen_len": int(gen_len),
                "loss": float(loss.detach().cpu().item()),
            }
            log_f.write(json.dumps(log_line) + "\n")
            log_f.flush()

    log_f.close()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("RL-LoRA model saved to", args.output_dir)


if __name__ == "__main__":
    main()
