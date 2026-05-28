#!/usr/bin/env bash
# Ablation: DistillPRM-1.5B trained WITHOUT error head contribution.
#
# Baseline to compare against: DistillPRM-1.5B-Adaptive(T=3) — avg F1 = 52.38%
#
# Only difference from adaptive_t3:
#   --lambda_error 0        (error head loss weight = 0; backbone learns only from score loss)
#
# All other hyperparameters are identical to adaptive_t3:
#   backbone_lr=1e-5  head_lr=1e-4  epochs=3  eff_batch=512
#   difficulty_temperature=3.0  focal_gamma=2.0  weight_scheme=sqrt_inv_freq
#
# Effective batch = 32 (per-GPU) × 16 (grad_accum) × 1 (GPU) = 512  ← same as 8-GPU run
#
# Runs on single GPU (CUDA_VISIBLE_DEVICES=0).
# Expected training time: ~14–18 h on H20 96 GB.
#
# Usage:
#   bash distill/run_ablation_no_error_head.sh

set -e
cd "$(dirname "$0")/.."   # project root


OUTPUT_BASE="models/DistillPRM-1.5B"
MODE="ablation_no_error_head"
LOG_DIR="${OUTPUT_BASE}/${MODE}"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo "Ablation: no error head  $(date)"
echo "Output : ${OUTPUT_BASE}/${MODE}/"
echo "Log    : ${LOG_DIR}/train.log"
echo "========================================================"
PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 python3 distill/step5_train_distillpRM.py \
    --mode          ${MODE} \
    --student_model models/Qwen2.5-Math-1.5B \
    --output_dir    ${OUTPUT_BASE} \
    --epochs        3 \
    --batch_size    4 \
    --grad_accum    128 \
    --backbone_lr   1e-5 \
    --head_lr       1e-4 \
    --max_length    1024 \
    --lambda_error  0 \
    --focal_gamma   2.0 \
    --weight_scheme sqrt_inv_freq \
    --log_every     100 \
    2>&1 | tee "${LOG_DIR}/train.log"

echo ""
echo "========================================================"
echo "Training complete.  $(date)"
echo "Model: ${OUTPUT_BASE}/${MODE}/best_model.pt"
echo "========================================================"
