#!/usr/bin/env bash
# Evaluate all three DistillPRM models on ProcessBench + val split in parallel.
# Each model runs on a separate GPU (GPU 0/1/2).
#
# Usage: bash distill/run_eval.sh

set -e
cd "$(dirname "$0")/.."   # project root

mkdir -p distill/eval_results

PB_DIR="data/ProcessBench"
EVAL_SCRIPT="distill/step6_evaluate.py"

echo "========================================================"
echo "DistillPRM Evaluation  $(date)"
echo "========================================================"

run_eval() {
    local mode=$1
    local gpu=$2
    local model_path="models/DistillPRM-1.5B/${mode}/best_model.pt"
    local pb_out="distill/eval_results/${mode}_processbench.json"
    local val_out="distill/eval_results/${mode}_val.json"

    echo "  [GPU ${gpu}] Starting ${mode} ..."

    # ProcessBench evaluation
    CUDA_VISIBLE_DEVICES=${gpu} python3 ${EVAL_SCRIPT} \
        --model_path  "${model_path}" \
        --processbench "${PB_DIR}" \
        --batch_size  64 \
        --max_length  1024 \
        --output      "${pb_out}" \
        > "distill/eval_results/${mode}_processbench.log" 2>&1

    # Val-split evaluation (step-level + localization metrics)
    CUDA_VISIBLE_DEVICES=${gpu} python3 ${EVAL_SCRIPT} \
        --model_path  "${model_path}" \
        --data_path   "data/genprm_math_steps_final.json" \
        --batch_size  64 \
        --max_length  1024 \
        --output      "${val_out}" \
        >> "distill/eval_results/${mode}_processbench.log" 2>&1

    echo "  [GPU ${gpu}] ${mode} done."
}

# Run sequentially on single GPU
run_eval ce          0
run_eval kl          0
run_eval adaptive    0
run_eval adaptive_t2 0
run_eval adaptive_t3 0
run_eval adaptive_t5 0

echo ""
echo "========================================================"
echo "All evaluations complete.  $(date)"
echo "Results in distill/eval_results/"
echo "========================================================"

# Print summary
python3 - <<'PYEOF'
import json
from pathlib import Path

modes = ["ce", "kl", "adaptive", "adaptive_t2", "adaptive_t3", "adaptive_t5"]
splits = ["gsm8k", "math", "olympiadbench", "omnimath", "average"]

print("\n=== ProcessBench F1 (official metric) ===")
print(f"{'mode':<14}", end="")
for s in splits:
    print(f"{s:>14}", end="")
print()
print("-" * (14 + 14*len(splits)))

for mode in modes:
    p = Path(f"distill/eval_results/{mode}_processbench.json")
    if not p.exists():
        print(f"{mode:<14}  (missing)")
        continue
    data = json.load(open(p))
    print(f"{mode:<14}", end="")
    for s in splits:
        f1 = data.get(s, {}).get("f1", float("nan"))
        print(f"{f1:>14.4f}", end="")
    print()

print("\n=== Val-split Metrics ===")
print(f"{'mode':<14} {'acc':>8} {'auc':>8} {'ece':>8} {'loc_acc':>10} {'macro_f1':>10}")
print("-" * 62)
for mode in modes:
    p = Path(f"distill/eval_results/{mode}_val.json")
    if not p.exists():
        print(f"{mode:<12}  (missing)")
        continue
    m = json.load(open(p))
    print(f"{mode:<14}"
          f" {m.get('accuracy',0):>8.4f}"
          f" {m.get('auc_roc',0):>8.4f}"
          f" {m.get('ece',0):>8.4f}"
          f" {m.get('localization_acc',0):>10.4f}"
          f" {m.get('error_macro_f1',0):>10.4f}")
PYEOF
