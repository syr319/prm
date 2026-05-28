#!/usr/bin/env bash
# Full SFT training: OpenPRM Qwen2.5-VL-7B + LoRA
# Usage: bash scripts/train_full.sh
# Estimated time: ~12-16h on single H20 97GB

set -e
cd "$(dirname "$0")/.."

echo "=== OpenPRM Full SFT Training ==="
echo "Config: configs/sft_config.yaml"
echo "Output: output/openprm-7b-lora"
echo ""

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True swift sft configs/sft_config.yaml

echo ""
echo "=== Training complete. Checkpoints saved to output/openprm-7b-lora ==="
