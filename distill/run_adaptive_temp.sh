#!/usr/bin/env bash
# Train three difficulty-temperature variants of adaptive DistillPRM on 8 GPUs.
# Modes:
#   adaptive_t2 — difficulty_temperature=2.0
#   adaptive_t3 — difficulty_temperature=3.0
#   adaptive_t5 — difficulty_temperature=5.0
#
# Usage:
#   bash distill/run_adaptive_temp.sh

set -e
cd "$(dirname "$0")/.."   # project root

N_GPUS=8

COMMON_ARGS="--epochs 3 --batch_size 16 --grad_accum 4 \
  --backbone_lr 1e-5 --head_lr 1e-4 --max_length 1024 \
  --lambda_error 0.1 --focal_gamma 2.0 --weight_scheme sqrt_inv_freq \
  --log_every 100"

train_mode() {
    local mode=$1
    echo "========================================================"
    echo "Starting training: mode=${mode}  $(date)"
    echo "Log: models/DistillPRM-1.5B/${mode}/train.log"
    echo "========================================================"
    torchrun --nproc_per_node=${N_GPUS} \
        distill/step5_train_distillpRM.py \
        --mode "${mode}" \
        ${COMMON_ARGS}
    echo "Finished mode=${mode}  $(date)"
}

train_mode adaptive_t2
train_mode adaptive_t3
train_mode adaptive_t5

echo ""
echo "All temperature experiments complete. $(date)"
echo "Models saved under models/DistillPRM-1.5B/{adaptive_t2,adaptive_t3,adaptive_t5}/"
