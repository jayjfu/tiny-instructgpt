#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${USE_TRL:=true}"
if [[ "$USE_TRL" = true ]]; then SUFFIX="_trl"; else SUFFIX=""; fi

# Step 1: supervised fine-tuning
python "src/tiny_instructgpt/train_sft${SUFFIX}.py" --batch_size 4 --learning_rate 2e-4 --num_epochs 2 --max_length 512 --save_steps 2_000 --no_resume

# Step 2: reward model training
python "src/tiny_instructgpt/train_rm${SUFFIX}.py" --batch_size 4 --learning_rate 1e-5 --num_epochs 1 --max_length 512 --save_steps 5_000 --no_resume

# Step 3: reinforcement learning from human feedback (PPO)
python src/tiny_instructgpt/train_ppo.py --batch_size 2 --learning_rate 1e-6 --num_epochs 1 --ppo_epochs 4 --max_length 512 --max_new_tokens 512 --clip_range 0.2 --kl_coef 0.05 --logging_steps 10