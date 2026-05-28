#!/usr/bin/env bash
# Sanity check: load first 500 training records, run 5 steps to verify
# - model loads correctly
# - data is consumed without format errors
# - loss is finite and starts decreasing
# Usage: bash scripts/sanity_check_sft.sh

set -e
cd "$(dirname "$0")/.."

SANITY_DATA="data/sanity_500.jsonl"
SANITY_OUT="output/sanity-check"

echo "=== Preparing 500-sample sanity dataset ==="
head -n 500 data/train_data.jsonl > "$SANITY_DATA"
echo "Written: $SANITY_DATA"

echo ""
echo "=== Starting sanity check (max_steps=5) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True swift sft configs/sft_config.yaml \
  --dataset          "$SANITY_DATA" \
  --val_dataset      data/val_data.jsonl \
  --output_dir       "$SANITY_OUT" \
  --max_steps        5 \
  --logging_steps    1 \
  --save_steps       999999 \
  --eval_steps       999999 \
  --num_train_epochs 1

echo ""
echo "=== Sanity check passed. Check loss in $SANITY_OUT/trainer_log.jsonl ==="
echo "If loss is finite and decreasing, run full training with:"
echo "  bash scripts/train_full.sh"
