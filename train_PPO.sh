# PPO Penalty-0.01 batch_size-4
python train_ppo_lora.py \
  --train_path data/humaneval_train.jsonl \
  --sft_model checkpoints/deepseek_1_3b_lora_sft_v2 \
  --output_dir checkpoints/deepseek_1_3b_lora_ppo_alpha0.01 \
  --length_penalty_alpha 0.01 \
  --num_updates 100 \
  --batch_size 4 \
  --ppo_epochs 4

python eval_humaneval.py \
  --data_path data/humaneval_test.jsonl \
  --model_path checkpoints/deepseek_1_3b_lora_ppo_alpha0.01 \
  --exp_name ppo_deepseek_1_3b_lora_alpha0_01
  
# PPO Penalty-0.01 batch_size-8
python train_ppo_lora.py \
  --train_path data/humaneval_train.jsonl \
  --sft_model checkpoints/deepseek_1_3b_lora_sft_v2 \
  --output_dir checkpoints/deepseek_1_3b_lora_ppo_alpha0.01_8 \
  --length_penalty_alpha 0.01 \
  --num_updates 100 \
  --batch_size 8 \
  --ppo_epochs 4

python eval_humaneval.py \
  --data_path data/humaneval_test.jsonl \
  --model_path checkpoints/deepseek_1_3b_lora_ppo_alpha0.01_8 \
  --exp_name ppo_deepseek_1_3b_lora_alpha0_01_8

# PPO Penalty-0.03 batch_size-4
python train_ppo_lora.py \
  --train_path data/humaneval_train.jsonl \
  --sft_model checkpoints/deepseek_1_3b_lora_sft_v2 \
  --output_dir checkpoints/deepseek_1_3b_lora_ppo_alpha0.03 \
  --length_penalty_alpha 0.03 \
  --num_updates 100 \
  --batch_size 4 \
  --ppo_epochs 4

python eval_humaneval.py \
  --data_path data/humaneval_test.jsonl \
  --model_path checkpoints/deepseek_1_3b_lora_ppo_alpha0.03 \
  --exp_name ppo_deepseek_1_3b_lora_alpha0_03

# PPO Penalty-0.03 batch_size-8
python train_ppo_lora.py \
  --train_path data/humaneval_train.jsonl \
  --sft_model checkpoints/deepseek_1_3b_lora_sft_v2 \
  --output_dir checkpoints/deepseek_1_3b_lora_ppo_alpha0.03_8 \
  --length_penalty_alpha 0.03 \
  --num_updates 100 \
  --batch_size 8 \
  --ppo_epochs 4

python eval_humaneval.py \
  --data_path data/humaneval_test.jsonl \
  --model_path checkpoints/deepseek_1_3b_lora_ppo_alpha0.03_8 \
  --exp_name ppo_deepseek_1_3b_lora_alpha0_03_8