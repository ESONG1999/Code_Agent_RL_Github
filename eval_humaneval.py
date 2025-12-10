#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eval_humaneval.py

output:
    results/<exp_name>_per_task.jsonl
    results/<exp_name>_metrics.json
"""

import os
import json
import argparse
from typing import Dict, Any, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from tqdm import tqdm

from peft import PeftConfig, PeftModel

import signal

def load_humaneval(path: str) -> List[Dict[str, Any]]:
    """加载 data/humaneval_test.jsonl"""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError()


signal.signal(signal.SIGALRM, _timeout_handler)


def run_humaneval_tests(
    prompt: str,
    completion: str,
    test_code: str,
    entry_point: str,
    debug: bool = False,
    time_limit: int = 5,  
) -> bool:

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
        return True

    except TimeoutError:
        if debug:
            print("==== TIMEOUT ====")
            print("entry_point:", entry_point)
        signal.alarm(0)
        return False

    except Exception as e:
        if debug:
            print("==== ERROR ====")
            print("entry_point:", entry_point)
            print("exception:", repr(e))
            print("----- PROMPT (first 20 lines) -----")
            print("\n".join(prompt.splitlines()[:20]))
            print("----- COMPLETION (first 40 lines) -----")
            print("\n".join(completion.splitlines()[:40]))
        signal.alarm(0)
        return False



def load_model_and_tokenizer(model_path: str, device: torch.device):
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_lora = os.path.exists(adapter_config_path)

    if is_lora:
        print(f"[INFO] Detected LoRA adapter at {model_path}, loading as LoRA-QLoRA model...")

        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_name = peft_config.base_model_name_or_path
        print(f"[INFO] Base model name: {base_model_name}")

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
        base_model.config.use_cache = True 

        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
        return model, tokenizer, True

    else:
        print(f"[INFO] No adapter_config.json found, loading {model_path} as a base model...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        model.to(device)
        model.eval()
        return model, tokenizer, False



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/humaneval_test.jsonl",
        help="Path to HumanEval test split JSONL",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "Base model name (e.g. deepseek-ai/deepseek-coder-1.3b-base) "
            "or LoRA checkpoint directory"
        ),
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Name for this experiment (used in results file names)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Max total sequence length (prompt + completion)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens to generate per task",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="If >0, use sampling; if =0, use greedy decoding",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p for nucleus sampling when temperature>0",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, e.g. 'cuda:0' or 'cpu'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation",
    )
    parser.add_argument(
        "--debug_first_n",
        type=int,
        default=0,
        help="前 N 个 task 打印异常和代码，便于调试",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=5,
        help="Max seconds allowed per HumanEval task (exec + tests).",
    )

    return parser.parse_args()



def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("[INFO] Using device:", device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer, is_lora = load_model_and_tokenizer(args.model_path, device)
    print(f"[INFO] Model loaded. LoRA: {is_lora}")

    examples = load_humaneval(args.data_path)
    total = len(examples)
    print(f"[INFO] Loaded {total} HumanEval tasks from {args.data_path}")

    per_task_path = os.path.join("results", f"{args.exp_name}_per_task.jsonl")
    metrics_path = os.path.join("results", f"{args.exp_name}_metrics.json")

    n_pass = 0
    lengths: List[int] = []

    with open(per_task_path, "w", encoding="utf-8") as f_out:
        for i, ex in enumerate(tqdm(examples, desc=f"Evaluating {args.exp_name}")):
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
                    do_sample=(args.temperature > 0.0),
                    temperature=args.temperature if args.temperature > 0 else 1.0,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            full_ids = gen_out[0]
            completion_ids = full_ids[prompt_len:]

            completion_text = tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
            )
            gen_len = completion_ids.numel()

            ok = run_humaneval_tests(
                prompt=prompt,
                completion=completion_text,
                test_code=test_code,
                entry_point=entry_point,
                debug=(i < args.debug_first_n),
                time_limit=args.time_limit,
            )

            if ok:
                n_pass += 1
            lengths.append(gen_len)

            record = {
                "task_id": ex["task_id"],
                "passed": bool(ok),
                "gen_len": int(gen_len),
                "completion": completion_text,
            }
            f_out.write(json.dumps(record) + "\n")

    pass_at_1 = n_pass / total if total > 0 else 0.0
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0

    metrics = {
        "exp_name": args.exp_name,
        "model_path": args.model_path,
        "is_lora": is_lora,
        "num_tasks": total,
        "num_pass": n_pass,
        "pass_at_1": pass_at_1,
        "avg_gen_len": avg_len,
    }
    with open(metrics_path, "w", encoding="utf-8") as f_metrics:
        json.dump(metrics, f_metrics, indent=2)

    print("[RESULT]")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
