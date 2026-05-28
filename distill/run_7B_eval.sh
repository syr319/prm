#!/usr/bin/env bash
# Evaluate DistillPRM-7B (adaptive_t3) on ProcessBench + val split.
# Runs on single GPU 0.
#
# Usage:
#   bash distill/run_7B_eval.sh

set -e
cd "$(dirname "$0")/.."   # project root

mkdir -p distill/eval_results

MODEL_PATH="models/DistillPRM-7B/adaptive_t3/epoch_02.pt"
STUDENT="models/Qwen2.5-Math-7B"
PB_DIR="data/ProcessBench"
EVAL_SCRIPT="distill/step6_evaluate.py"
PB_OUT="distill/eval_results/7B_adaptive_t3_processbench.json"
VAL_OUT="distill/eval_results/7B_adaptive_t3_val.json"
LOG="distill/eval_results/7B_adaptive_t3_processbench.log"

echo "========================================================"
echo "DistillPRM-7B Evaluation  $(date)"
echo "========================================================"

echo "  [GPU 0] ProcessBench ..."
CUDA_VISIBLE_DEVICES=0 python3 ${EVAL_SCRIPT} \
    --model_path    "${MODEL_PATH}" \
    --student_model "${STUDENT}" \
    --processbench  "${PB_DIR}" \
    --batch_size    32 \
    --max_length    1024 \
    --output        "${PB_OUT}" \
    > "${LOG}" 2>&1

echo "  [GPU 0] Val-split ..."
CUDA_VISIBLE_DEVICES=0 python3 ${EVAL_SCRIPT} \
    --model_path    "${MODEL_PATH}" \
    --student_model "${STUDENT}" \
    --data_path     "data/genprm_math_steps_final.json" \
    --batch_size    32 \
    --max_length    1024 \
    --output        "${VAL_OUT}" \
    >> "${LOG}" 2>&1

echo ""
echo "========================================================"
echo "Evaluation complete.  $(date)"
echo "========================================================"

# Print results alongside 1.5B models
python3 - <<'PYEOF'
import json
from pathlib import Path

models = {
    "1.5B-CE":           "ce",
    "1.5B-KL":           "kl",
    "1.5B-Adap(T=1)":    "adaptive",
    "1.5B-Adap(T=3)":    "adaptive_t3",
    "7B-Adap(T=3)":      "7B_adaptive_t3",
}
splits = ["gsm8k", "math", "olympiadbench", "omnimath", "average"]

print("\n=== ProcessBench F1 ===")
print(f"{'model':<18}", end="")
for s in splits:
    print(f"{s:>14}", end="")
print()
print("-" * (18 + 14*len(splits)))

for label, key in models.items():
    p = Path(f"distill/eval_results/{key}_processbench.json")
    if not p.exists():
        print(f"{label:<18}  (missing)")
        continue
    data = json.load(open(p))
    print(f"{label:<18}", end="")
    for s in splits:
        f1 = data.get(s, {}).get("f1", float("nan"))
        print(f"{f1*100:>14.2f}", end="")
    print()

print("\n=== Val-split Metrics ===")
print(f"{'model':<18} {'acc':>8} {'auc':>8} {'ece':>8} {'loc_acc':>10}")
print("-" * 57)
for label, key in models.items():
    p = Path(f"distill/eval_results/{key}_val.json")
    if not p.exists():
        print(f"{label:<18}  (missing)")
        continue
    m = json.load(open(p))
    print(f"{label:<18}"
          f" {m.get('accuracy',0)*100:>7.2f}%"
          f" {m.get('auc_roc',0):>8.4f}"
          f" {m.get('ece',0):>8.4f}"
          f" {m.get('localization_acc',0)*100:>9.2f}%")
PYEOF
