# RL-Code-Agent: SFT + RL/PPO Fine-tuning of DeepSeek-Coder on HumanEval

This repo implements a **full SFT + RL pipeline** on top of the
[`deepseek-ai/deepseek-coder-1.3b-base`](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base)
model for **Python code generation** evaluated on HumanEval-style tasks.

The project is designed to be **small-GPU friendly** (16–24GB), using
**4-bit QLoRA** + **LoRA adapters** for both supervised fine-tuning (SFT)
and reinforcement learning (vanilla policy gradient and PPO).

> TL;DR: On a 1.3B code model, we improve HumanEval pass@1 from **15.2% → 36.4% (SFT)** and
> further to **48.5% (RL)** while also **reducing average solution length by ~40%**.

---

## 1. Motivation

Large code LMs like DeepSeek-Coder already achieve strong performance on benchmarks such as
HumanEval, but they are not explicitly optimized for **“pass unit tests with minimal and clean code”**.

This repo explores:

1. How far we can push a **1.3B open-source code model** with
   - lightweight **LoRA-based SFT** on HumanEval-like tasks, and  
   - **RL objectives** based on unit-test rewards and length penalties.

2. Whether **PPO-style policy optimization** provides additional gains and better
   length–accuracy trade-offs vs. vanilla REINFORCE.

3. How to build a **robust evaluation harness** that:
   - is indentation-safe for HumanEval canonical solutions, and  
   - gracefully handles infinite loops / very slow solutions via per-task timeouts.

---

## 2. Methods

### 2.1 Base model

- **Backbone**: [`deepseek-ai/deepseek-coder-1.3b-base`](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base)
- **Tokenizer**: HuggingFace `AutoTokenizer` with `trust_remote_code=True`
- **Quantization**: 4-bit NF4 via `bitsandbytes` (QLoRA)
- **Parameter-efficient tuning**: `peft` LoRA adapters trained on top of the quantized base

All experiments fit comfortably on a **single 16–24GB GPU**.

### 2.2 Data

We use a HumanEval-style dataset with the following fields:

- `task_id`: unique identifier per problem  
- `prompt`: function signature + docstring (no function body)  
- `canonical_solution`: reference implementation (function body, including indentation)  
- `test`: unit test code defining `check(candidate)`  
- `entry_point`: name of the function to be evaluated

We split tasks into a small train split (`humaneval_train.jsonl`) and an eval split
(`humaneval_test.jsonl`) for ablations.

### 2.3 Supervised Fine-tuning (SFT)

**Script:** `train_sft_lora.py`

- Objective: causal LM loss on **`prompt + canonical_solution`**, but we **mask out the prompt tokens**
  (only the solution body contributes to the loss).
- Implementation details:
  - 4-bit QLoRA (`BitsAndBytesConfig`) + LoRA adapters on key projection layers  
  - Only LoRA parameters are trainable; base model is frozen.
  - We preserve indentation in `canonical_solution` (no `.lstrip()`!) to avoid teaching the model
    invalid Python.
- Typical hyperparameters:
  - `r=8`, `lora_alpha=16`, `lora_dropout=0.05`
  - `learning_rate=5e-5`, `num_train_epochs=1.0`
  - `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`

This stage produces a **SFT LoRA checkpoint**, e.g.:

```bash
checkpoints/deepseek_1_3b_lora_sft_v2
```

### 2.4 RL with unit-test rewards (REINFORCE)

**Script:** `train_rl_lora.py`

We first implement **vanilla policy gradient (REINFORCE)** as a baseline:

- **Policy**: DeepSeek-Coder-1.3B + SFT LoRA adapter  
  (only LoRA parameters are trainable; the base model is frozen).
- For each HumanEval-style task, we:
  1. Sample a completion from the current policy.
  2. Execute `prompt + completion + test` in a sandbox.
  3. Define a scalar reward:
     ```text
     r = 1.0   if all unit tests pass
         0.0   otherwise

     reward = r - alpha * (generated_length / 100)
     ```
- The loss per trajectory is:
  ```text
  L(θ) = - reward · log p_θ(completion | prompt)
  ```
- We use a **Unix `signal.alarm` timeout** to avoid infinite loops and extremely slow code paths
  (e.g., buggy implementations like `prime_fib` that never terminate).

This stage produces RL LoRA checkpoints such as:

- `checkpoints/deepseek_1_3b_lora_rl_alpha0.01_100`  
- `checkpoints/deepseek_1_3b_lora_rl_alpha0.03`

---

### 2.5 PPO with clipped surrogate objective

**Script:** `train_ppo_lora.py`

We then upgrade the RL stage to a **PPO-Clip** objective, which uses a clipped policy ratio
to stabilize updates while allowing multiple epochs per batch of trajectories.

For each PPO update:

1. **Rollout / trajectory collection**

   - Sample a batch of tasks from `humaneval_train.jsonl`.
   - For each task:
     - Encode the `prompt` and generate a completion from the current policy.
     - Compute `reward` using the same unit-test + length-penalty scheme.
     - Recompute the log-probability of the sampled completion and store:
       - `input_ids_full`
       - `prompt_len`, `completion_len`
       - `logp_old` (sum over completion tokens)
       - `reward`

2. **Advantage estimation**

   - For a batch of size `B`:
     ```text
     rewards    = [r₁, …, r_B]
     advantages = (rewards - mean(rewards)) / (std(rewards) + ε)
     ```

3. **PPO-Clip objective**

   - For each trajectory:
     ```text
     logp_new  = sum log π_θ(a_t | s_t)
     ratio     = exp(logp_new - logp_old)
     unclipped = ratio · advantage
     clipped   = clip(ratio, 1 - ε, 1 + ε) · advantage

     loss_π = - min(unclipped, clipped)
     ```
   - We only update **LoRA adapter parameters**, keeping the 1.3B base frozen.
   - We typically run several PPO epochs per batch (e.g., 4 epochs) and explore
     different length penalties (`alpha`) and batch sizes (`batch_size ∈ {4, 8}`).

Resulting PPO checkpoints include:

- `checkpoints/deepseek_1_3b_lora_ppo_alpha0.01`  
- `checkpoints/deepseek_1_3b_lora_ppo_alpha0.03`  
- `checkpoints/deepseek_1_3b_lora_ppo_alpha0.01_8`  
- `checkpoints/deepseek_1_3b_lora_ppo_alpha0.03_8`

---

## 3. Evaluation

**Script:** `eval_humaneval.py`

The evaluation harness is:

- **Model-agnostic**: it can evaluate both:
  - base HuggingFace models (e.g., `deepseek-ai/deepseek-coder-1.3b-base`), and  
  - LoRA adapter directories (containing `adapter_config.json`).
- **Indentation-safe**: it preserves indentation in canonical solutions and generated code.
- **Robust**: it uses per-task timeouts to prevent infinite loops from hanging evaluation.

For each task:

1. Encode the `prompt` with the tokenizer.
2. Generate a completion with `max_new_tokens = 256`.
3. Construct:
   ```text
   full_code = prompt + "\n\n" + completion + "\n\n" + test
   ```
4. Execute `full_code` with a `signal.alarm` timeout.
5. Retrieve the function `entry_point` and run `check(candidate)`.
6. Record whether all tests pass.

We report:

- `pass_at_1 = num_pass / num_tasks`
- `avg_gen_len`: average number of generated tokens per completion.

---

## 4. Results

All experiments are run on **33 HumanEval-style tasks** using a single 16–24GB GPU.

### 4.1 Overall performance

| Model / Checkpoint                           | Method                      | pass@1 | #passed / #tasks | Avg. length |
|---------------------------------------------|-----------------------------|:------:|:----------------:|:-----------:|
| `deepseek-ai/deepseek-coder-1.3b-base`      | Base                        | 0.152  | 5 / 33           | 177.27      |
| `deepseek_1_3b_lora_sft_v2`                 | SFT (QLoRA + LoRA)          | 0.364  | 12 / 33          | 136.33      |
| `deepseek_1_3b_lora_rl_alpha0.01_100`       | RL (REINFORCE, α = 0.01)    | **0.485** | 16 / 33       | 107.55      |
| `deepseek_1_3b_lora_rl_alpha0.03`           | RL (REINFORCE, α = 0.03)    | 0.424  | 14 / 33          | 136.21      |
| `deepseek_1_3b_lora_ppo_alpha0.01`          | PPO (α = 0.01, batch = 4)   | 0.455  | 15 / 33          | 89.64       |
| `deepseek_1_3b_lora_ppo_alpha0.03`          | PPO (α = 0.03, batch = 4)   | 0.424  | 14 / 33          | 66.33       |
| `deepseek_1_3b_lora_ppo_alpha0.01_8`        | PPO (α = 0.01, batch = 8)   | 0.394  | 13 / 33          | 80.27       |
| `deepseek_1_3b_lora_ppo_alpha0.03_8`        | PPO (α = 0.03, batch = 8)   | 0.455  | 15 / 33          | 92.91       |

**Key observations:**

- **SFT alone** improves HumanEval pass@1 from **15.2% → 36.4%** (+21.2 points),
  while shortening average outputs from ~177 to ~136 tokens.
- **RL with unit-test rewards + length penalty (α = 0.01)** further boosts pass@1
  to **48.5%** (+33.3 points over base) and reduces average length to ~108 tokens.
- **PPO** achieves similar correctness (up to 45.5% pass@1) with even more aggressive
  length reduction (down to 66–90 tokens), illustrating a clear **length–accuracy trade-off**.

---

## 5. Quick start

### 5.1 Environment

```bash
conda create -n rl-code-agent python=3.10 -y
conda activate rl-code-agent

# PyTorch (modify cu121 -> cu118 if needed)
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1

pip install   "transformers==4.45.2"   "peft==0.13.0"   "accelerate==1.1.0"   "bitsandbytes==0.45.2"   "datasets==3.1.0"   "tqdm"   "numpy<2"
```

### 5.2 Training and evaluation

#### 1) Supervised fine-tuning (SFT) & REINFORCE Train & Evaluation

```bash
sh train_SFT_RL.sh
```

#### 2) PPO Train & Evaluation

```bash
sh train_PPO.sh
```

---

## 6. Future work: DPO and preference-based alignment
