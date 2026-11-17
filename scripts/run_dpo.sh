#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${USE_TRL:=true}"
if [[ "$USE_TRL" = true ]]; then SUFFIX="_trl"; else SUFFIX=""; fi

# Supervised fine-tuning
python "src/tiny_instructgpt/train_sft${SUFFIX}.py" --batch_size 4 --learning_rate 2e-4 --num_epochs 2 --max_length 512 --save_steps 2_000 --no_resume

# Direct preference optimization (DPO)
python "src/tiny_instructgpt/train_dpo${SUFFIX}.py" --batch_size 2 --learning_rate 5e-7 --num_epochs 1 --max_length 512 --logging_steps 1_000 --save_steps 5_000 --eval_steps 2_000 --no_resume