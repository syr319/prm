#!/usr/bin/env bash
# Run all three training modes sequentially on 8 GPUs: ce → kl → adaptive
# Each mode trains for 3 epochs.
# Logs go to models/DistillPRM-1.5B/{mode}/train.log
#
# Usage:
#   bash distill/run_all_training.sh

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

# Skip ce if already finished
if [ -f "models/DistillPRM-1.5B/ce/final_model.pt" ]; then
    echo "ce already complete, skipping."
else
    train_mode ce
fi

train_mode kl
train_mode adaptive

echo ""
echo "All training modes complete. $(date)"
echo "Models saved under models/DistillPRM-1.5B/{ce,kl,adaptive}/"
