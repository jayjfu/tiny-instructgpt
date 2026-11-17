#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="data/oasst2"
DATA_FILE="2023-11-05_oasst2_ready.trees.jsonl"

mkdir -p "$DATA_DIR"
if [[ ! -f "$DATA_DIR/$DATA_FILE" ]]; then
  wget -P "$DATA_DIR" "https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/main/${DATA_FILE}.gz"
  gunzip "$DATA_DIR/${DATA_FILE}.gz"
fi

python src/tiny_instructgpt/prepare_oasst2.py