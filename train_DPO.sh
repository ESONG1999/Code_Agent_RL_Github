# python generate_dpo_data.py \
#   --train_path data/humaneval_train.jsonl \
#   --sft_model checkpoints/deepseek_1_3b_lora_sft_v2 \
#   --output_path data/humaneval_dpo_pairs_correct_only.jsonl \
#   --per_task_samples 16

python train_dpo_lora.py \
  --dpo_data_path data/humaneval_dpo_pairs_correct_only.jsonl \
  --sft_model checkpoints/deepseek_1_3b_lora_sft_v2 \
  --output_dir checkpoints/deepseek_1_3b_lora_dpo_beta0.05_ep1 \
  --per_device_train_batch_size 2 \
  --num_epochs 1 \
  --beta 0.05 \
  --learning_rate 2e-6

python eval_humaneval.py \
  --data_path data/humaneval_test.jsonl \
  --model_path checkpoints/deepseek_1_3b_lora_dpo_beta0.05_ep1 \
  --exp_name dpo_deepseek_1_3b_lora_beta0.05_ep1
