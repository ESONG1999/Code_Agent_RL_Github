#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download and split the OpenAI HumanEval dataset into train / test JSONL files.

Output:
    data/humaneval_train.jsonl
    data/humaneval_test.jsonl

Each line is a JSON object with:
    {
        "task_id": str,
        "prompt": str,
        "canonical_solution": str,
        "test": str,
        "entry_point": str
    }
"""

import os
import json
import random
from datasets import load_dataset


def main(seed: int = 42, train_ratio: float = 0.8):
    os.makedirs("data", exist_ok=True)

    ds = load_dataset("openai/openai_humaneval", split="test")  # 164 tasks 
    n = len(ds)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    train_cut = int(train_ratio * n)
    train_idx = set(indices[:train_cut])

    train_path = os.path.join("data", "humaneval_train.jsonl")
    test_path = os.path.join("data", "humaneval_test.jsonl")

    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(test_path, "w", encoding="utf-8") as f_test:

        for i in range(n):
            ex = ds[i]
            record = {
                "task_id": ex["task_id"],
                "prompt": ex["prompt"],
                "canonical_solution": ex["canonical_solution"],
                "test": ex["test"],
                "entry_point": ex["entry_point"],
            }
            line = json.dumps(record, ensure_ascii=False)
            if i in train_idx:
                f_train.write(line + "\n")
            else:
                f_test.write(line + "\n")

    print(f"Saved {train_path} and {test_path}")


if __name__ == "__main__":
    main()
