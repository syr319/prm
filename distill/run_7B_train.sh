#!/usr/bin/env bash
# Train DistillPRM-7B with adaptive_t3 on 8 GPUs.
#
# Student backbone : models/Qwen2.5-Math-7B
# Loss mode        : adaptive_t3 (difficulty_temperature=3.0)
# Output           : models/DistillPRM-7B/adaptive_t3/
#
# Memory notes (H20 96GB per GPU):
#   7B bf16 model  ≈ 14 GB
#   Optimizer (m+v bf16) ≈ 28 GB
#   Gradients      ≈ 14 GB
#   Activations (gradient_checkpointing) ≈ 2–4 GB
#   Total          ≈ 58–60 GB  → safe on 96 GB
#
# Effective batch = 4 (per-GPU) × 8 (grad_accum) × 8 (GPUs) = 256
#
# Usage:
#   bash distill/run_7B_train.sh

set -e
cd "$(dirname "$0")/.."   # project root

echo "========================================================"
echo "DistillPRM-7B Training  adaptive_t3  $(date)"
echo "========================================================"

export PYTORCH_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=8 \
    distill/step5_train_distillpRM.py \
    --mode            adaptive_t3 \
    --student_model   models/Qwen2.5-Math-7B \
    --output_dir      models/DistillPRM-7B \
    --gradient_checkpointing \
    --epochs          3 \
    --batch_size      4 \
    --grad_accum      8 \
    --backbone_lr     1e-5 \
    --head_lr         1e-4 \
    --max_length      1024 \
    --lambda_error    0.1 \
    --focal_gamma     2.0 \
    --weight_scheme   sqrt_inv_freq \
    --log_every       100

echo ""
echo "========================================================"
echo "Training complete.  $(date)"
echo "Model: models/DistillPRM-7B/adaptive_t3/best_model.pt"
echo "========================================================"
