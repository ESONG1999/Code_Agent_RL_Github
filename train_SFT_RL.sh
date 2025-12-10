# SFT
python eval_humaneval.py \
  --data_path data/humaneval_test.jsonl \
  --model_path deepseek-ai/deepseek-coder-1.3b-base \
  --exp_name base_deepseek_1_3b

python train_sft_lora.py \
  --train_path data/humaneval_train.jsonl \
  --model_name deepseek-ai/deepseek-coder-1.3b-base \
  --output_dir checkpoints/deepseek_1_3b_lora_sft_v2

python eval_humaneval.py \
  --data_path data/humaneval_test.jsonl \
  --model_path checkpoints/deepseek_1_3b_lora_sft_v2 \
  --exp_name sft_v2_deepseek_1_3b_lora \
  --time_limit 60

# RL Penalty-0.01
python train_rl_lora.py \
  --train_path data/humaneval_train.jsonl \
  --sft_model checkpoints/deepseek_1_3b_lora_sft_v2 \
  --output_dir checkpoints/deepseek_1_3b_lora_rl_alpha0.01_100 \
  --length_penalty_alpha 0.01 \
  --num_steps 100

python eval_humaneval.py \
  --data_path data/humaneval_test.jsonl \
  --model_path checkpoints/deepseek_1_3b_lora_rl_alpha0.01_100 \
  --exp_name rl_deepseek_1_3b_lora_alpha0_01_100 \
  --time_limit 60

# RL Penalty-0.03
python train_rl_lora.py \
  --train_path data/humaneval_train.jsonl \
  --sft_model checkpoints/deepseek_1_3b_lora_sft_v2 \
  --output_dir checkpoints/deepseek_1_3b_lora_rl_alpha0.03 \
  --length_penalty_alpha 0.03 \
  --num_steps 100

python eval_humaneval.py \
  --data_path data/humaneval_test.jsonl \
  --model_path checkpoints/deepseek_1_3b_lora_rl_alpha0.03 \
  --exp_name rl_deepseek_1_3b_lora_alpha0_03 \
  --time_limit 60