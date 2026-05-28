"""
Temperature-scaling calibration for DistillPRM ProcessBench predictions.

Reads per-step predictions from error_analysis JSON (no GPU needed),
applies sigmoid(logit(score) / T) for various T values, then evaluates
ProcessBench F1 (threshold tuned on GSM8K, applied to all splits).

Usage:
  cd /mnt/user/shenyiran3/PRM
  python3 distill/temperature_calibration.py \
      --preds distill/eval_results/7B_math_error_analysis.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# ─── ProcessBench F1 (problem-level, first-error localisation) ───────────────

def reconstruct_problems(records):
    """
    Group per-step records by (split, prob_idx) and reconstruct problem metadata.

    Returns:
      problems: {split: [{prob_idx, label, step_scores_orig}]}
        label  = first error step index (-1 if all correct)
        step_scores_orig = list of original pred_score, one per step in order
    """
    # split → prob_idx → {steps: {step_idx: score}, label}
    raw = defaultdict(lambda: defaultdict(lambda: {"steps": {}, "label": None}))

    for r in records:
        split    = r["split"]
        prob_idx = r["prob_idx"]
        step_idx = r["step_idx"]
        raw[split][prob_idx]["steps"][step_idx] = r["pred_score"]

        tl = r["true_label"]
        if tl == 0:
            raw[split][prob_idx]["label"] = step_idx   # first error step
        elif raw[split][prob_idx]["label"] is None:
            pass  # will stay None → correct problem

    problems = {}
    for split, prob_dict in raw.items():
        split_probs = []
        for prob_idx in sorted(prob_dict.keys()):
            entry  = prob_dict[prob_idx]
            label  = entry["label"] if entry["label"] is not None else -1
            # build ordered score list (step 0, 1, 2, ...)
            steps  = entry["steps"]
            n_steps = max(steps.keys()) + 1
            scores  = [steps.get(i, 1.0) for i in range(n_steps)]
            split_probs.append({
                "prob_idx":    prob_idx,
                "label":       label,
                "step_scores": scores,
            })
        problems[split] = split_probs
    return problems


def calibrate(score, T):
    """sigmoid(logit(score) / T)"""
    score  = np.clip(score, 1e-7, 1 - 1e-7)
    logit  = np.log(score / (1.0 - score))
    return 1.0 / (1.0 + np.exp(-logit / T))


def compute_f1(problems_split, tau, T):
    """
    ProcessBench F1 for one split at given (T, tau).
    F1 = harmonic_mean(acc_error, acc_correct)
    """
    correct_erroneous = 0
    total_erroneous   = 0
    correct_clean     = 0
    total_clean       = 0

    for prob in problems_split:
        label       = prob["label"]
        cal_scores  = [calibrate(s, T) for s in prob["step_scores"]]

        # First step the model predicts as wrong (score < tau)
        first_wrong_pred = next(
            (i for i, s in enumerate(cal_scores) if s < tau), None
        )

        if label == -1:
            # Clean problem: model should predict ALL steps as correct
            total_clean += 1
            if first_wrong_pred is None:
                correct_clean += 1
        else:
            # Erroneous problem: model should flag exactly step `label` first
            total_erroneous += 1
            if first_wrong_pred == label:
                correct_erroneous += 1

    acc_err = correct_erroneous / max(total_erroneous, 1)
    acc_cor = correct_clean     / max(total_clean, 1)
    f1      = 2 * acc_err * acc_cor / max(acc_err + acc_cor, 1e-8)
    return f1, acc_err, acc_cor


def tune_threshold(problems_gsm8k, T, n_steps=200):
    """Find tau* that maximises GSM8K F1 for temperature T."""
    best_tau, best_f1 = 0.5, -1.0
    for tau in np.linspace(0.01, 0.99, n_steps):
        f1, _, _ = compute_f1(problems_gsm8k, float(tau), T)
        if f1 > best_f1:
            best_f1  = f1
            best_tau = float(tau)
    return best_tau, best_f1


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds",   required=True,
                        help="Path to error_analysis JSON with per-step pred_score")
    parser.add_argument("--temps",   default="0.3,0.5,0.7,1.0,1.5,2.0",
                        help="Comma-separated temperature values")
    args = parser.parse_args()

    print(f"Loading predictions from {args.preds} ...")
    records = json.load(open(args.preds))
    print(f"  {len(records):,} step records loaded")

    temperatures = [float(t) for t in args.temps.split(",")]
    splits = ["gsm8k", "math", "olympiadbench", "omnimath"]

    problems = reconstruct_problems(records)
    for s in splits:
        n = len(problems.get(s, []))
        print(f"  {s}: {n} problems reconstructed")

    # ── Run calibration for each T ────────────────────────────────────────────
    print("\n" + "="*75)
    print("TEMPERATURE SCALING CALIBRATION RESULTS")
    print("="*75)

    header = f"{'T':>6}  {'tau*':>6}  " + "  ".join(f"{s[:8]:>10}" for s in splits) + "  {'avg_F1':>8}"
    print(header)
    print("-"*75)

    results = []
    for T in temperatures:
        # Tune threshold on GSM8K
        tau, f1_gsm = tune_threshold(problems.get("gsm8k", []), T)

        # Evaluate all splits
        f1s = {}
        for split in splits:
            if split not in problems:
                f1s[split] = float("nan")
                continue
            f1, acc_err, acc_cor = compute_f1(problems[split], tau, T)
            f1s[split] = f1

        avg_f1 = float(np.nanmean(list(f1s.values())))
        results.append({"T": T, "tau": tau, "f1s": f1s, "avg_f1": avg_f1})

        row = (f"{T:>6.1f}  {tau:>6.4f}  " +
               "  ".join(f"{f1s.get(s, float('nan'))*100:>9.2f}%" for s in splits) +
               f"  {avg_f1*100:>7.2f}%")
        print(row)

    # ── Best T ────────────────────────────────────────────────────────────────
    print("-"*75)
    best = max(results, key=lambda r: r["avg_f1"])
    print(f"\nBest T = {best['T']}  (avg F1 = {best['avg_f1']*100:.2f}%,  tau* = {best['tau']:.4f})")
    print("\nPer-split F1 at best T:")
    for s in splits:
        print(f"  {s:<16}: {best['f1s'].get(s, float('nan'))*100:.2f}%")

    # ── Detailed breakdown at each T (acc_error / acc_correct) ───────────────
    print("\n" + "="*75)
    print("DETAILED: acc_error / acc_correct at each T (tuned tau)")
    print("="*75)
    for T in temperatures:
        tau, _ = tune_threshold(problems.get("gsm8k", []), T)
        print(f"\nT={T:.1f}  tau*={tau:.4f}")
        print(f"  {'split':<16} {'F1':>8} {'acc_err':>10} {'acc_cor':>10}")
        for split in splits:
            if split not in problems:
                continue
            f1, acc_err, acc_cor = compute_f1(problems[split], tau, T)
            print(f"  {split:<16} {f1*100:>7.2f}%  {acc_err*100:>8.2f}%  {acc_cor*100:>8.2f}%")


if __name__ == "__main__":
    main()
