#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_ppo_lora.py

Usage:

    # 1) SFT LoRA
    python train_sft_lora.py \
        --train_path data/humaneval_train.jsonl \
        --model_name deepseek-ai/deepseek-coder-1.3b-base \
        --output_dir checkpoints/deepseek_1_3b_lora_sft

    # 2) SFT LoRA PPO
    python train_ppo_lora.py \
        --train_path data/humaneval_train.jsonl \
        --sft_model checkpoints/deepseek_1_3b_lora_sft \
        --output_dir checkpoints/deepseek_1_3b_lora_ppo_alpha0.01 \
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

import signal



class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError()


signal.signal(signal.SIGALRM, _timeout_handler)


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
    time_limit: int = 5,
) -> float:
    prompt = prompt.rstrip()
    test_code = test_code.rstrip()
    completion = completion.rstrip("\n")

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/humaneval_train.jsonl")
    parser.add_argument(
        "--sft_model",
        type=str,
        required=True,
        help="SFT LoRA checkpoint 路径, e.g. checkpoints/deepseek_1_3b_lora_sft",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/deepseek_1_3b_lora_ppo",
    )
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_updates", type=int, default=300)      
    parser.add_argument("--batch_size", type=int, default=4)         
    parser.add_argument("--ppo_epochs", type=int, default=4)         
    parser.add_argument("--clip_range", type=float, default=0.2)     
    parser.add_argument("--length_penalty_alpha", type=float, default=0.01)
    parser.add_argument("--time_limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()

def collect_trajectories(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    examples: List[Dict[str, Any]],
    args,
    device: torch.device,
) -> List[Dict[str, Any]]:
    model.eval()
    trajs = []

    for _ in range(args.batch_size):
        ex = random.choice(examples)
        prompt = ex["prompt"]
        test_code = ex["test"]
        entry_point = ex["entry_point"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length - args.max_new_tokens,
        ).to(device)
        prompt_ids = inputs["input_ids"][0]
        prompt_len = prompt_ids.size(0)

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

        full_ids = gen_out[0].to(device)           # [T]
        completion_ids = full_ids[prompt_len:]     # [T_gen]

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
            time_limit=args.time_limit,
        )
        gen_len = completion_ids.numel()
        length_penalty = args.length_penalty_alpha * (gen_len / 100.0)
        reward_value = base_reward - length_penalty

        with torch.no_grad():
            full_ids_batch = full_ids.unsqueeze(0)            # [1, T]
            attention_mask = torch.ones_like(full_ids_batch)  # [1, T]
            outputs = model(
                input_ids=full_ids_batch,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # [1, T, V]

            logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [1, T-1, V]
            target_ids = full_ids_batch[:, 1:]                       # [1, T-1]

            total_len = full_ids_batch.size(1)
            comp_len = total_len - prompt_len
            if comp_len <= 0:
                continue

            start_idx = prompt_len - 1
            end_idx = total_len - 1

            logprobs_slice = logprobs[:, start_idx:end_idx, :]   # [1, comp_len, V]
            target_slice = target_ids[:, start_idx:end_idx]      # [1, comp_len]

            selected = logprobs_slice.gather(
                dim=-1,
                index=target_slice.unsqueeze(-1),
            ).squeeze(-1)    # [1, comp_len]

            logp_old_sum = selected.sum().item()

        trajs.append({
            "task_id": ex["task_id"],
            "prompt": prompt,
            "input_ids_full": full_ids.detach().cpu(),
            "prompt_len": int(prompt_len),
            "completion_len": int(comp_len),
            "logp_old": float(logp_old_sum),
            "reward": float(reward_value),
            "base_reward": float(base_reward),
            "gen_len": int(gen_len),
        })

    return trajs

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    examples = load_humaneval(args.train_path)

    log_path = os.path.join(args.output_dir, "ppo_train_log.jsonl")
    log_f = open(log_path, "w", encoding="utf-8")

    for update in tqdm(range(1, args.num_updates + 1), desc="PPO-LoRA updates"):
        trajs = collect_trajectories(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            args=args,
            device=device,
        )
        if len(trajs) == 0:
            continue

        batch_size = len(trajs)
        rewards = torch.tensor(
            [t["reward"] for t in trajs],
            dtype=torch.float32,
            device=device,
        )

        advantages = rewards - rewards.mean()
        if advantages.std() > 1e-8:
            advantages = advantages / (advantages.std() + 1e-8)

        old_logps = torch.tensor(
            [t["logp_old"] for t in trajs],
            dtype=torch.float32,
            device=device,
        )

        for epoch in range(args.ppo_epochs):
            model.train()
            indices = torch.randperm(batch_size)

            all_losses = []

            for idx in indices:
                traj = trajs[int(idx)]

                full_ids = traj["input_ids_full"].to(device)  # [T]
                prompt_len = traj["prompt_len"]
                comp_len = traj["completion_len"]

                full_ids_batch = full_ids.unsqueeze(0)           # [1, T]
                attention_mask = torch.ones_like(full_ids_batch) # [1, T]

                outputs = model(
                    input_ids=full_ids_batch,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits  # [1, T, V]

                logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [1, T-1, V]
                target_ids = full_ids_batch[:, 1:]                       # [1, T-1]

                total_len = full_ids_batch.size(1)
                assert total_len - prompt_len == comp_len

                start_idx = prompt_len - 1
                end_idx = total_len - 1

                logprobs_slice = logprobs[:, start_idx:end_idx, :]   # [1, comp_len, V]
                target_slice = target_ids[:, start_idx:end_idx]      # [1, comp_len]

                selected = logprobs_slice.gather(
                    dim=-1,
                    index=target_slice.unsqueeze(-1),
                ).squeeze(-1)   # [1, comp_len]

                logp_new_sum = selected.sum(dim=1)  # [1]

                logp_old_sum = old_logps[int(idx)].unsqueeze(0)  # [1]
                adv = advantages[int(idx)].unsqueeze(0)          # [1]

                # ratio = exp(logp_new - logp_old)
                ratio = torch.exp(logp_new_sum - logp_old_sum)   # [1]

                # PPO-Clip surrogate
                unclipped = ratio * adv
                clipped = torch.clamp(
                    ratio,
                    1.0 - args.clip_range,
                    1.0 + args.clip_range,
                ) * adv
                loss_pi = -torch.min(unclipped, clipped)  # [1]

                all_losses.append(loss_pi)

            loss = torch.stack(all_losses).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if update % args.log_every == 0:
            avg_reward = float(rewards.mean().item())
            avg_len = float(
                sum(t["gen_len"] for t in trajs) / len(trajs)
            )
            avg_base_reward = float(
                sum(t["base_reward"] for t in trajs) / len(trajs)
            )

            log_line = {
                "update": update,
                "batch_size": batch_size,
                "avg_reward": avg_reward,
                "avg_base_reward": avg_base_reward,
                "avg_gen_len": avg_len,
            }
            log_f.write(json.dumps(log_line) + "\n")
            log_f.flush()
            print(f"[PPO] update={update} avg_reward={avg_reward:.4f} "
                  f"avg_base_reward={avg_base_reward:.4f} avg_len={avg_len:.2f}")

    log_f.close()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[INFO] PPO-LoRA model saved to", args.output_dir)


if __name__ == "__main__":
    main()
