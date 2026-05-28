"""
Error analysis for DistillPRM-7B-Instruct (adaptive_t3) on ProcessBench.

Usage:
  cd /mnt/user/shenyiran3/PRM
  CUDA_VISIBLE_DEVICES=0 python3 distill/error_analysis.py \
      --model_path  outputs/distillprm-7b-instruct-adaptive-t3/adaptive_t3/best_model.pt \
      --student_model models/Qwen2.5-Math-7B-Instruct \
      --processbench  data/ProcessBench \
      --output        distill/eval_results/7B_instruct_error_analysis.json
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "distill"))

from step4_build_student_model import DistillPRM, STUDENT_MODEL_PATH
from step5_train_distillpRM import DistillPRMDataset, collate_fn
from step6_evaluate import load_model, _compute_f1_at_threshold
from transformers import AutoTokenizer


# ─── Inference ───────────────────────────────────────────────────────────────

def run_split(model, tokenizer, pb_data, batch_size, max_length, device):
    """Return flat list of (prob_idx, step_idx, score) for one split."""
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    flat_records = []
    index_map    = []   # (prob_idx, step_idx) for each flat record

    for prob_idx, prob in enumerate(pb_data):
        context_parts = []
        for step_idx, step_text in enumerate(prob["steps"]):
            flat_records.append({
                "question":     prob["problem"],
                "context":      "\n\n".join(context_parts),
                "current_step": step_text,
                "hard_label":   1,
                "verification_cot": "",
            })
            index_map.append((prob_idx, step_idx))
            context_parts.append(step_text)

    dataset = DistillPRMDataset(flat_records, tokenizer, max_length=max_length)
    loader  = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers  = 4,
        collate_fn  = lambda b: collate_fn(b, pad_token_id=pad_id),
        pin_memory  = True,
    )

    all_scores = []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            score, _ = model(ids, mask)
            all_scores.extend(score.cpu().float().tolist())

    return index_map, all_scores


# ─── Step-level ground truth ─────────────────────────────────────────────────

def step_true_label(label, step_idx):
    """
    ProcessBench label: -1 = all correct, k>=0 = first wrong step index.
    Returns:
      1  — step is correct (before first error, or problem is fully correct)
      0  — step is the first error
      -1 — step is after the first error (excluded from step-level analysis)
    """
    if label == -1:
        return 1
    if step_idx < label:
        return 1
    if step_idx == label:
        return 0
    return -1   # post-error steps: ambiguous, exclude


# ─── Threshold tuning ────────────────────────────────────────────────────────

def tune_threshold(pb_data, problem_map, scores_arr):
    """Find tau* that maximises ProcessBench F1 on this split."""
    thresholds = np.linspace(0.01, 0.99, 200)
    best_tau, best_f1 = 0.5, -1.0
    for tau in thresholds:
        f1, _, _ = _compute_f1_at_threshold(pb_data, problem_map, scores_arr, tau)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
    return best_tau, best_f1


# ─── Analysis ────────────────────────────────────────────────────────────────

def analyse(all_preds):
    """Print and return full error analysis."""

    # ── 1. Per-split FP/FN counts ────────────────────────────────────────────
    print("\n" + "="*65)
    print("1. FALSE POSITIVE / FALSE NEGATIVE by split")
    print("="*65)
    print(f"  {'split':<16} {'total':>7} {'wrong':>7} {'FP':>7} {'FN':>7} "
          f"{'FP%':>7} {'FN%':>7}")
    print("-"*65)

    split_stats = {}
    for split in ["gsm8k", "math", "olympiadbench", "omnimath"]:
        rows = [p for p in all_preds if p["split"] == split and p["true_label"] != -1]
        wrong = [p for p in rows if p["is_wrong"]]
        fp = [p for p in wrong if p["true_label"] == 1]   # correct → predicted wrong
        fn = [p for p in wrong if p["true_label"] == 0]   # error   → predicted correct
        split_stats[split] = {"total": len(rows), "wrong": len(wrong),
                              "fp": len(fp), "fn": len(fn)}
        fp_pct = 100 * len(fp) / max(len(wrong), 1)
        fn_pct = 100 * len(fn) / max(len(wrong), 1)
        print(f"  {split:<16} {len(rows):>7} {len(wrong):>7} {len(fp):>7} {len(fn):>7} "
              f"{fp_pct:>6.1f}% {fn_pct:>6.1f}%")

    # Overall
    all_step = [p for p in all_preds if p["true_label"] != -1]
    all_wrong = [p for p in all_step if p["is_wrong"]]
    fp_all = [p for p in all_wrong if p["true_label"] == 1]
    fn_all = [p for p in all_wrong if p["true_label"] == 0]
    print("-"*65)
    print(f"  {'ALL':<16} {len(all_step):>7} {len(all_wrong):>7} "
          f"{len(fp_all):>7} {len(fn_all):>7} "
          f"{100*len(fp_all)/max(len(all_wrong),1):>6.1f}% "
          f"{100*len(fn_all)/max(len(all_wrong),1):>6.1f}%")

    # ── 2a. Step position distribution of errors ─────────────────────────────
    print("\n" + "="*65)
    print("2a. STEP POSITION distribution of wrong predictions")
    print("="*65)
    pos_counter = Counter(p["step_idx"] + 1 for p in all_wrong)
    total_wrong = len(all_wrong)
    print(f"  {'position':>10}  {'count':>8}  {'%':>7}")
    for pos in sorted(pos_counter):
        print(f"  {'step '+str(pos):>10}  {pos_counter[pos]:>8}  "
              f"{100*pos_counter[pos]/total_wrong:>6.1f}%")

    # Summarise: step 1 vs rest
    step1_wrong = pos_counter.get(1, 0)
    step2plus   = total_wrong - step1_wrong
    print(f"\n  Step 1 errors : {step1_wrong}  ({100*step1_wrong/max(total_wrong,1):.1f}%)")
    print(f"  Step 2+ errors: {step2plus}  ({100*step2plus/max(total_wrong,1):.1f}%)")

    # ── 2b. Step text length distribution ────────────────────────────────────
    print("\n" + "="*65)
    print("2b. STEP TEXT LENGTH distribution (chars)")
    print("="*65)
    bins = [(0, 100), (100, 300), (300, 600), (600, 1200), (1200, 99999)]
    labels = ["0-100", "100-300", "300-600", "600-1200", "1200+"]

    wrong_lens  = [len(p["step_text"]) for p in all_wrong]
    all_lens    = [len(p["step_text"]) for p in all_step]

    print(f"  {'len bucket':>12}  {'wrong':>8}  {'all steps':>10}  {'error rate':>11}")
    for (lo, hi), lbl in zip(bins, labels):
        w = sum(1 for l in wrong_lens if lo <= l < hi)
        a = sum(1 for l in all_lens   if lo <= l < hi)
        er = 100 * w / max(a, 1)
        print(f"  {lbl:>12}  {w:>8}  {a:>10}  {er:>10.1f}%")

    print(f"\n  Median step length (wrong) : {int(np.median(wrong_lens)) if wrong_lens else 0} chars")
    print(f"  Median step length (all)   : {int(np.median(all_lens)) if all_lens else 0} chars")

    # ── 2c. FP/FN breakdown per split (difficulty gradient) ──────────────────
    print("\n" + "="*65)
    print("2c. FP vs FN breakdown per split (difficulty gradient)")
    print("="*65)
    print(f"  {'split':<16} {'FP rate':>10} {'FN rate':>10}  "
          f"(FP=correct→wrong, FN=error→correct)")
    print("-"*65)
    for split in ["gsm8k", "math", "olympiadbench", "omnimath"]:
        rows = [p for p in all_preds if p["split"] == split and p["true_label"] != -1]
        correct_steps = [p for p in rows if p["true_label"] == 1]
        error_steps   = [p for p in rows if p["true_label"] == 0]
        fp = sum(1 for p in correct_steps if p["is_wrong"])
        fn = sum(1 for p in error_steps   if p["is_wrong"])
        fp_rate = 100 * fp / max(len(correct_steps), 1)
        fn_rate = 100 * fn / max(len(error_steps),   1)
        print(f"  {split:<16} {fp_rate:>9.1f}% {fn_rate:>9.1f}%  "
              f"({fp}/{len(correct_steps)} correct steps misclassified, "
              f"{fn}/{len(error_steps)} error steps missed)")

    # ── 3. 20 random olympiadbench wrong examples ─────────────────────────────
    print("\n" + "="*65)
    print("3. 20 RANDOM WRONG PREDICTIONS from olympiadbench")
    print("="*65)
    olym_wrong = [p for p in all_preds
                  if p["split"] == "olympiadbench"
                  and p["is_wrong"]
                  and p["true_label"] != -1]
    random.seed(42)
    samples = random.sample(olym_wrong, min(20, len(olym_wrong)))

    for i, s in enumerate(samples, 1):
        label_str  = "CORRECT step" if s["true_label"] == 1 else "ERROR step"
        pred_str   = "predicted WRONG" if s["pred_label"] == 0 else "predicted CORRECT"
        error_type = "FALSE POSITIVE" if s["true_label"] == 1 else "FALSE NEGATIVE"
        print(f"\n[{i:02d}] {error_type}  (step {s['step_idx']+1})  "
              f"score={s['pred_score']:.3f}  true={label_str}")
        q = s["question"][:200].replace("\n", " ")
        t = s["step_text"][:400].replace("\n", " ")
        print(f"  Q: {q}{'...' if len(s['question'])>200 else ''}")
        print(f"  Step: {t}{'...' if len(s['step_text'])>400 else ''}")

    return split_stats


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   required=True)
    parser.add_argument("--student_model", default=STUDENT_MODEL_PATH)
    parser.add_argument("--processbench", required=True)
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--max_length",   type=int, default=1024)
    parser.add_argument("--output",       default="distill/eval_results/error_analysis.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pb_dir = Path(args.processbench)

    print(f"Loading model from {args.model_path} ...")
    model     = load_model(args.model_path, device, student_model_path=args.student_model)
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    split_names = ["gsm8k", "math", "olympiadbench", "omnimath"]
    split_raw   = {}   # split → (pb_data, problem_map, scores_arr) for threshold tuning

    all_preds = []

    # ── Inference per split ──────────────────────────────────────────────────
    for split in split_names:
        path = pb_dir / f"{split}.json"
        if not path.exists():
            print(f"  Skipping {split}: not found")
            continue
        pb_data = json.load(open(path))
        print(f"\n[{split}] {len(pb_data)} problems")

        # Also build problem_map for threshold tuning (matches step6 format)
        flat_records_thr = []
        problem_map      = []
        for prob_idx, prob in enumerate(pb_data):
            context_parts = []
            start = len(flat_records_thr)
            for step_text in prob["steps"]:
                flat_records_thr.append({
                    "question": prob["problem"],
                    "context":  "\n\n".join(context_parts),
                    "current_step": step_text,
                    "hard_label": 1,
                    "verification_cot": "",
                })
                context_parts.append(step_text)
            problem_map.append((prob_idx, start, len(flat_records_thr)))

        index_map, scores = run_split(
            model, tokenizer, pb_data,
            args.batch_size, args.max_length, device
        )
        split_raw[split] = (pb_data, problem_map, np.array(scores))

        # Store raw scores with step metadata (threshold applied later)
        for (prob_idx, step_idx), score in zip(index_map, scores):
            prob  = pb_data[prob_idx]
            label = prob["label"]
            tl    = step_true_label(label, step_idx)
            all_preds.append({
                "split":      split,
                "prob_idx":   prob_idx,
                "step_idx":   step_idx,
                "question":   prob["problem"],
                "step_text":  prob["steps"][step_idx],
                "true_label": tl,
                "pred_score": float(score),
                "pred_label": None,   # fill after threshold
                "is_wrong":   None,
            })

    # ── Tune threshold on GSM8K ──────────────────────────────────────────────
    ref = "gsm8k" if "gsm8k" in split_raw else next(iter(split_raw))
    tau, f1_ref = tune_threshold(*split_raw[ref])
    print(f"\nOptimal threshold τ* = {tau:.4f}  (GSM8K F1 = {f1_ref:.4f})")

    # ── Apply threshold ──────────────────────────────────────────────────────
    for p in all_preds:
        p["pred_label"] = 1 if p["pred_score"] >= tau else 0
        p["is_wrong"]   = (p["pred_label"] != p["true_label"]) and (p["true_label"] != -1)

    # ── Run analysis ─────────────────────────────────────────────────────────
    analyse(all_preds)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_preds, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_preds)} step predictions → {out_path}")


if __name__ == "__main__":
    main()
