#!/bin/bash
# Score both MM-Vet and LLaVA-Bench with checkpoint-2658, plus LLaVA-Bench with checkpoint-2100
set -e
cd /mnt/user/shenyiran3/PRM
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export $(grep -v '^#' .claude/settings.env | xargs)

LOG=logs/ckpt2658.log
mkdir -p logs

echo "=== [$(date)] Waiting for ckpt-2658 merge to complete ===" | tee -a $LOG
# Wait until merged model directory has model files
while [ ! -f output/openprm-7b-merged-2658/model.safetensors ] && \
      [ ! -f output/openprm-7b-merged-2658/pytorch_model.bin ] && \
      [ "$(ls output/openprm-7b-merged-2658/*.safetensors 2>/dev/null | wc -l)" -eq 0 ]; do
    sleep 10
done
echo "=== [$(date)] Merge complete ===" | tee -a $LOG

echo "=== [$(date)] Step 1: Score MM-Vet with ckpt2658 ===" | tee -a $LOG
python3 scripts/03c_score_openprm.py \
    --model-dir output/openprm-7b-merged-2658 \
    --output openprm_bon_ckpt2658.json 2>&1 | tee -a $LOG
echo "=== [$(date)] Step 1 done ===" | tee -a $LOG

echo "=== [$(date)] Step 2: Score LLaVA-Bench with ckpt2100 ===" | tee -a $LOG
python3 scripts/03c_score_openprm.py \
    --dataset llava_bench \
    --output llava_bench_openprm_bon_2100.json 2>&1 | tee -a $LOG
echo "=== [$(date)] Step 2 done ===" | tee -a $LOG

echo "=== [$(date)] Step 3: Score LLaVA-Bench with ckpt2658 ===" | tee -a $LOG
python3 scripts/03c_score_openprm.py \
    --dataset llava_bench \
    --model-dir output/openprm-7b-merged-2658 \
    --output llava_bench_openprm_bon_2658.json 2>&1 | tee -a $LOG
echo "=== [$(date)] Step 3 done ===" | tee -a $LOG

echo "=== [$(date)] Step 4: Evaluate MM-Vet (all methods incl. ckpt2658) ===" | tee -a $LOG
# Delete old ckpt2658 eval to force re-run
rm -f results/mmvet/eval/eval_openprm_bon_ckpt2658.json
python3 scripts/04_evaluate_mmvet.py 2>&1 | tee -a $LOG
echo "=== [$(date)] Step 4 done ===" | tee -a $LOG

echo "=== [$(date)] Step 5: Evaluate LLaVA-Bench (both ckpts) ===" | tee -a $LOG
python3 scripts/05_evaluate_llava_bench.py 2>&1 | tee -a $LOG
echo "=== [$(date)] Step 5 done ===" | tee -a $LOG

echo "=== [$(date)] ALL DONE ===" | tee -a $LOG
