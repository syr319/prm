# #!/usr/bin/env bash
# # Iterative distillation round 2 — DistillPRM-7B-Instruct continued training.
# #
# # Prerequisite: run mine_hard_steps.py first to produce hard_steps_from_instruct.json:
# #   python3 distill/mine_hard_steps.py \
# #       --checkpoint outputs/distillprm-7b-instruct-adaptive-t3/adaptive_t3/best_model.pt \
# #       --student_model models/Qwen2.5-Math-7B \
# #       --output data/hard_steps_from_instruct.json
# #
# # Training setup:
# #   Resumes from  : outputs/distillprm-7b-instruct-adaptive-t3/adaptive_t3/best_model.pt
# #   Combined data : 171K original (weight=1) + ~N hard steps (weight=3)
# #   backbone_lr   : 2e-6  (= 1e-5 / 5, conservative fine-tuning LR)
# #   head_lr       : 2e-5  (= 1e-4 / 5)
# #   epochs        : 1
# #   eff_batch     : 4 (per-GPU) × 8 (grad_accum) × 8 (GPUs) = 256
# #
# # Output: outputs/distillprm-7b-instruct-iter2/best_model.pt
# #
# # Usage:
# #   bash distill/run_iter2_train.sh

# set -e
# cd "$(dirname "$0")/.."   # project root

# echo "========================================================"
# echo "DistillPRM-7B iter2  $(date)"
# echo "========================================================"

# export PYTORCH_ALLOC_CONF=expandable_segments:True

# # torchrun --nproc_per_node=8 \
# #     distill/step7_iter2_train.py \
# #     --student_model       models/Qwen2.5-Math-7B \
# #     --resume_checkpoint   models/DistillPRM-7B/adaptive_t3/best_model.pt \
# #     --data_path           data/genprm_math_steps_final.json \
# #     --hard_data_path      data/hard_steps_from_7b.json \
# #     --output_dir          outputs/distillprm-7b-iter2 \
# #     --mode                adaptive_t3 \
# #     --gradient_checkpointing \
# #     --epochs              1 \
# #     --batch_size          4 \
# #     --grad_accum          8 \
# #     --backbone_lr         2e-6 \
# #     --head_lr             2e-5 \
# #     --max_length          1024 \
# #     --lambda_error        0.1 \
# #     --focal_gamma         2.0 \
# #     --weight_scheme       sqrt_inv_freq \
# #     --log_every           100

# echo ""
# echo "========================================================"
# echo "Training complete.  $(date)"
# echo "Model: outputs/distillprm-7b-iter2/best_model.pt"
# echo "========================================================"
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

echo "========================================================"
echo "DistillPRM-7B iter2-combined  $(date)"
echo "========================================================"

export PYTORCH_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=8 \
    distill/step7_iter2_train.py \
    --student_model       models/Qwen2.5-Math-7B \
    --resume_checkpoint   models/DistillPRM-7B/adaptive_t3/best_model.pt \
    --data_path           data/genprm_math_steps_final.json \
    --hard_data_path      data/hard_steps_from_7b_combined.json \
    --output_dir          outputs/distillprm-7b-iter2-combined \
    --mode                adaptive_t3 \
    --gradient_checkpointing \
    --epochs              1 \
    --batch_size          4 \
    --grad_accum          8 \
    --backbone_lr         2e-6 \
    --head_lr             2e-5 \
    --max_length          1024 \
    --lambda_error        0.1 \
    --focal_gamma         2.0 \
    --weight_scheme       sqrt_inv_freq \
    --log_every           100

echo ""
echo "========================================================"
echo "Training complete.  $(date)"
echo "Model: outputs/distillprm-7b-iter2-combined/best_model.pt"
echo "========================================================"