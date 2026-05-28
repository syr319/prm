#!/bin/bash
# Full LLaVA-Bench pipeline
set -e
cd /mnt/user/shenyiran3/PRM
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export $(grep -v '^#' .claude/settings.env | xargs)

LOG=logs/llava_bench.log
mkdir -p logs results/llava_bench/bon results/llava_bench/eval

echo "=== [$(date)] Step 1: Generate LLaVA-Bench candidates ===" | tee -a $LOG
python3 scripts/02_generate_candidates_llavabench.py 2>&1 | tee -a $LOG
echo "=== [$(date)] Step 1 done ===" | tee -a $LOG

echo "=== [$(date)] Step 2a: VisualPRM scoring ===" | tee -a $LOG
python3 scripts/03b_score_visualprm.py --dataset llava_bench 2>&1 | tee -a $LOG
echo "=== [$(date)] Step 2a done ===" | tee -a $LOG

echo "=== [$(date)] Step 2b: OpenPRM scoring ===" | tee -a $LOG
python3 scripts/03c_score_openprm.py --dataset llava_bench 2>&1 | tee -a $LOG
echo "=== [$(date)] Step 2b done ===" | tee -a $LOG

echo "=== [$(date)] Step 3: Evaluate with Qwen-VL-Max ===" | tee -a $LOG
python3 scripts/05_evaluate_llava_bench.py 2>&1 | tee -a $LOG
echo "=== [$(date)] Step 3 done ===" | tee -a $LOG

echo "=== [$(date)] ALL DONE ===" | tee -a $LOG
