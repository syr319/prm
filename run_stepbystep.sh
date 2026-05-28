#!/bin/bash
# Full MM-Vet step-by-step pipeline
set -e
cd /mnt/user/shenyiran3/PRM
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export $(grep -v '^#' .claude/settings.env | xargs)

LOG=logs/stepbystep.log
mkdir -p logs results/mmvet/bon results/mmvet/eval

echo "=== [$(date)] Step 1: Generate step-by-step candidates ===" | tee -a $LOG
python3 scripts/02_generate_candidates.py \
  --output results/mmvet/bon/candidates_stepbystep.json \
  --stepbystep 2>&1 | tee -a $LOG
echo "=== [$(date)] Step 1 done ===" | tee -a $LOG

echo "=== [$(date)] Step 2a: LLaVA-Critic scoring ===" | tee -a $LOG
python3 scripts/03a_score_llava_critic.py \
  --input candidates_stepbystep.json \
  --output llava_critic_bon_stepbystep.json 2>&1 | tee -a $LOG
echo "=== [$(date)] Step 2a done ===" | tee -a $LOG

echo "=== [$(date)] Step 2b: VisualPRM scoring ===" | tee -a $LOG
python3 scripts/03b_score_visualprm.py \
  --input candidates_stepbystep.json \
  --output visualprm_bon_stepbystep.json 2>&1 | tee -a $LOG
echo "=== [$(date)] Step 2b done ===" | tee -a $LOG

echo "=== [$(date)] Step 2c: OpenPRM scoring ===" | tee -a $LOG
python3 scripts/03c_score_openprm.py \
  --input candidates_stepbystep.json \
  --output openprm_bon_stepbystep.json 2>&1 | tee -a $LOG
echo "=== [$(date)] Step 2c done ===" | tee -a $LOG

echo "=== [$(date)] Step 3: Evaluate all 4 methods ===" | tee -a $LOG
python3 scripts/04_evaluate_mmvet.py --suffix stepbystep 2>&1 | tee -a $LOG
echo "=== [$(date)] Step 3 done ===" | tee -a $LOG

echo "=== [$(date)] ALL DONE ===" | tee -a $LOG
