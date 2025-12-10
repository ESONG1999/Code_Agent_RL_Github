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

#### 1) Supervised fine-tuning (SFT)

```bash
python train_sft_lora.py   --train_path data/humaneval_train.jsonl   --model_name deepseek-ai/deepseek-coder-1.3b-base   --output_dir checkpoints/deepseek_1_3b_lora_sft_v2
```

#### 2) RL (REINFORCE)

```bash
python train_rl_lora.py   --train_path data/humaneval_train.jsonl   --sft_model checkpoints/deepseek_1_3b_lora_sft_v2   --output_dir checkpoints/deepseek_1_3b_lora_rl_alpha0.01_100   --length_penalty_alpha 0.01
```

#### 3) PPO

```bash
python train_ppo_lora.py   --train_path data/humaneval_train.jsonl   --sft_model checkpoints/deepseek_1_3b_lora_sft_v2   --output_dir checkpoints/deepseek_1_3b_lora_ppo_alpha0.01   --length_penalty_alpha 0.01   --num_updates 300   --batch_size 4   --ppo_epochs 4
```

#### 4) Evaluation (base + SFT + RL/PPO)

```bash
# Base model
python eval_humaneval.py   --data_path data/humaneval_test.jsonl   --model_path deepseek-ai/deepseek-coder-1.3b-base   --exp_name base_deepseek_1_3b

# SFT LoRA
python eval_humaneval.py   --data_path data/humaneval_test.jsonl   --model_path checkpoints/deepseek_1_3b_lora_sft_v2   --exp_name sft_v2_deepseek_1_3b_lora

# RL
python eval_humaneval.py   --data_path data/humaneval_test.jsonl   --model_path checkpoints/deepseek_1_3b_lora_rl_alpha0.01_100   --exp_name rl_deepseek_1_3b_lora_alpha0_01_100

# PPO
python eval_humaneval.py   --data_path data/humaneval_test.jsonl   --model_path checkpoints/deepseek_1_3b_lora_ppo_alpha0.01   --exp_name ppo_deepseek_1_3b_lora_alpha0_01
```

---

## 6. Future work: DPO and preference-based alignment

As a next step, we plan to:

- Sample multiple candidate solutions per prompt and **convert scalar rewards into pairwise preferences**.
- Train a **Direct Preference Optimization (DPO)** model on these synthetic preference pairs.
- Compare **offline DPO** against online RL/PPO in terms of correctness and code quality.

Stay tuned 👀
