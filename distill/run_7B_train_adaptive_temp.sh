#!/usr/bin/env bash
# Train DistillPRM-7B with adaptive temperature (方向2)

set -e
cd "$(dirname "$0")/.."

echo "========================================================"
echo "DistillPRM-7B Training  adaptive_t3 + adaptive temperature  $(date)"
echo "========================================================"

export PYTORCH_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=8 \
    distill/step5_train_distillpRM.py \
    --mode            adaptive_temp \
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
    --log_every       100 \
    --adaptive_temperature \
    --t_min           1.0 \
    --t_max           5.0

echo "========================================================"
echo "Training complete.  $(date)"
echo "========================================================"