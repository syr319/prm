"""
Step 6: Evaluate DistillPRM on held-out data or ProcessBench.

Metrics:
  - Step accuracy (binary: predicted >= 0.5 matches hard_label)
  - AUC-ROC
  - Expected Calibration Error (ECE)
  - First-error-localization accuracy: for each multi-step solution,
    does the model flag the correct step as the FIRST wrong step?
  - Error type classification F1 (macro)
  - Per-difficulty-bucket accuracy breakdown

Usage:
  # Evaluate on training data's held-out split (fast sanity check)
  python step6_evaluate.py --model_path models/DistillPRM-1.5B/ce/best_model.pt

  # Evaluate on custom JSON dataset (same format as training data)
  python step6_evaluate.py --model_path ... --data_path data/processbench.json

  # Evaluate all three trained modes at once
  python step6_evaluate.py --compare --output_dir models/DistillPRM-1.5B
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "distill"))

from step4_build_student_model import (
    DistillPRM,
    build_input_text,
    NUM_ERROR_TYPES,
    STUDENT_MODEL_PATH,
)
from step5_train_distillpRM import (
    DistillPRMDataset,
    collate_fn,
    extract_error_type,
    compute_ece,
)

# ─── Paths ────────────────────────────────────────────────────────────────────

TRAINING_DATA_PATH = ROOT / "data" / "genprm_math_steps_with_soft_scores.json"
OUTPUT_DIR         = ROOT / "models" / "DistillPRM-1.5B"
EVAL_DIR           = ROOT / "distill" / "eval_results"


# ─── Evaluation helpers ───────────────────────────────────────────────────────

def load_model(
    model_path:        str,
    device:            torch.device,
    student_model_path: str = STUDENT_MODEL_PATH,
) -> DistillPRM:
    """Load a DistillPRM checkpoint."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    print(f"Loading model from {path} ...")
    model = DistillPRM(model_name_or_path=student_model_path)

    state = torch.load(path, map_location=device, weights_only=True)
    # Handle both bare state_dict and full checkpoint
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    return model


@torch.no_grad()
def run_inference(
    model:      DistillPRM,
    records:    List[dict],
    tokenizer,
    device:     torch.device,
    batch_size: int = 32,
    max_length: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on all records.

    Returns:
      scores        : [N]  float, predicted P(correct)
      hard_labels   : [N]  int,   binary ground truth
      error_pred    : [N]  int,   predicted error type class
      error_true    : [N]  int,   ground-truth error type (keyword-matched)
    """
    from transformers import AutoTokenizer
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    dataset = DistillPRMDataset(records, tokenizer, max_length=max_length)
    loader  = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 4,
        collate_fn  = lambda b: collate_fn(b, pad_token_id=pad_id),
        pin_memory  = True,
    )

    all_scores       = []
    all_hard_labels  = []
    all_error_pred   = []
    all_error_true   = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        student_score, error_logits = model(input_ids, attention_mask)

        all_scores.extend(student_score.cpu().float().tolist())
        all_hard_labels.extend(batch["hard_label"].int().tolist())
        all_error_pred.extend(error_logits.argmax(dim=-1).cpu().tolist())
        all_error_true.extend(batch["error_label"].tolist())

    return (
        np.array(all_scores),
        np.array(all_hard_labels),
        np.array(all_error_pred),
        np.array(all_error_true),
    )


# ─── Metric computation ───────────────────────────────────────────────────────

def compute_step_metrics(
    scores:      np.ndarray,
    hard_labels: np.ndarray,
) -> dict:
    """Step-level binary classification metrics."""
    preds_binary = (scores >= 0.5).astype(int)
    acc  = accuracy_score(hard_labels, preds_binary)
    ece  = compute_ece(scores, hard_labels)
    auc  = roc_auc_score(hard_labels, scores)

    # Precision / recall on wrong steps (class 0 = hard_label 0)
    tp = ((preds_binary == 0) & (hard_labels == 0)).sum()
    fp = ((preds_binary == 0) & (hard_labels == 1)).sum()
    fn = ((preds_binary == 1) & (hard_labels == 0)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy":            float(acc),
        "auc_roc":             float(auc),
        "ece":                 float(ece),
        "wrong_step_precision": float(precision),
        "wrong_step_recall":   float(recall),
        "wrong_step_f1":       float(f1),
    }


def compute_error_type_metrics(
    error_pred: np.ndarray,
    error_true: np.ndarray,
) -> dict:
    """Error type classification metrics (on wrong steps only)."""
    wrong_mask = error_true != 0  # wrong steps
    if wrong_mask.sum() == 0:
        return {"error_macro_f1": 0.0}

    ep_wrong = error_pred[wrong_mask]
    et_wrong = error_true[wrong_mask]

    macro_f1 = f1_score(
        et_wrong, ep_wrong, average="macro", zero_division=0
    )
    return {"error_macro_f1": float(macro_f1)}


def compute_error_localization(
    records:  List[dict],
    scores:   np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    First-error-localization accuracy.

    For each conversation, find the first step where:
      (a) model predicts wrong (score < threshold)
    vs. the actual first wrong step according to hard_label.

    A prediction is "correct" if:
      - The solution has no error AND model doesn't flag any step, OR
      - The solution has an error AND model flags the SAME step as the first error.
    """
    # Group records by conversation (conv_idx)
    convs: Dict[int, List[dict]] = defaultdict(list)
    for i, r in enumerate(records):
        conv_id = r.get("conv_idx", i)
        convs[conv_id].append((r["step_index"], r["hard_label"], scores[i]))

    n_correct = 0
    n_total   = 0
    n_has_error = 0
    n_localized = 0

    for conv_id, steps in convs.items():
        # Sort by step_index
        steps = sorted(steps, key=lambda x: x[0])
        if not steps:
            continue

        labels_ordered = [s[1] for s in steps]
        scores_ordered = [s[2] for s in steps]

        # Find actual first error
        first_error_true = next(
            (i for i, l in enumerate(labels_ordered) if l == 0), None
        )
        # Find predicted first error
        first_error_pred = next(
            (i for i, sc in enumerate(scores_ordered) if sc < threshold), None
        )

        n_total += 1
        if first_error_true is None and first_error_pred is None:
            n_correct += 1   # both say correct
        elif first_error_true is not None:
            n_has_error += 1
            if first_error_pred == first_error_true:
                n_correct += 1
                n_localized += 1

    loc_acc   = n_correct / n_total if n_total > 0 else 0.0
    error_loc = n_localized / n_has_error if n_has_error > 0 else 0.0

    return {
        "localization_acc":         float(loc_acc),   # overall: correct + error localization
        "error_localization_acc":   float(error_loc),  # on erroneous solutions only
        "n_conversations":          n_total,
        "n_with_error":             n_has_error,
    }


def compute_difficulty_bucket_accuracy(
    scores:       np.ndarray,
    hard_labels:  np.ndarray,
    soft_scores:  Optional[np.ndarray] = None,
) -> dict:
    """
    Accuracy breakdown by step difficulty bucket.

    Difficulty is defined by teacher_score if available,
    otherwise by score spread (uncertain = difficult).
    """
    if soft_scores is None:
        # Use model's uncertainty as proxy
        difficulty = 1.0 - 2.0 * np.abs(scores - 0.5)
    else:
        difficulty = 1.0 - 2.0 * np.abs(soft_scores - 0.5)

    preds_binary = (scores >= 0.5).astype(int)
    correct = (preds_binary == hard_labels).astype(float)

    buckets = {
        "easy":   difficulty <= 0.3,
        "medium": (difficulty > 0.3) & (difficulty <= 0.7),
        "hard":   difficulty > 0.7,
    }

    result = {}
    for name, mask in buckets.items():
        if mask.sum() > 0:
            result[f"{name}_n"]   = int(mask.sum())
            result[f"{name}_acc"] = float(correct[mask].mean())

    return result


# ─── Main evaluation function ──────────────────────────────────────────────────

def evaluate_model(
    model_path:         str,
    data_path:          str,
    device:             torch.device,
    batch_size:         int = 32,
    max_length:         int = 1024,
    val_frac:           float = 0.05,
    seed:               int = 42,
    use_val_split:      bool = True,
    student_model_path: str = STUDENT_MODEL_PATH,
) -> dict:
    """Run full evaluation and return metrics dict."""
    from transformers import AutoTokenizer

    # Load data
    print(f"Loading data from {data_path} ...")
    with open(data_path) as f:
        records = json.load(f)
    records = [r for r in records if r.get("hard_label") in (0, 1)]
    print(f"Loaded {len(records):,} records")

    if use_val_split:
        # Use same val split as training (reproducible with seed)
        n_val  = max(1, int(len(records) * val_frac))
        rng    = np.random.default_rng(seed)
        idx    = rng.permutation(len(records))
        val_idx = idx[-n_val:]
        records = [records[i] for i in val_idx]
        print(f"Using {len(records):,} validation records")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        student_model_path, trust_remote_code=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = load_model(model_path, device, student_model_path=student_model_path)

    # Run inference
    print("Running inference ...")
    scores, hard_labels, error_pred, error_true = run_inference(
        model, records, tokenizer, device, batch_size, max_length
    )

    # Extract soft scores if available
    soft_scores = None
    if records[0].get("soft_score") is not None:
        soft_scores = np.array([r.get("soft_score", r["hard_label"]) for r in records])

    # Compute metrics
    metrics = {}
    metrics.update(compute_step_metrics(scores, hard_labels))
    metrics.update(compute_error_type_metrics(error_pred, error_true))
    metrics.update(compute_error_localization(records, scores))
    metrics.update(compute_difficulty_bucket_accuracy(scores, hard_labels, soft_scores))

    # Score statistics
    correct_scores = scores[hard_labels == 1]
    wrong_scores   = scores[hard_labels == 0]
    metrics["mean_score_correct"] = float(correct_scores.mean()) if len(correct_scores) > 0 else 0.0
    metrics["mean_score_wrong"]   = float(wrong_scores.mean())   if len(wrong_scores)   > 0 else 0.0

    return metrics


# ─── ProcessBench evaluation ──────────────────────────────────────────────────

def _compute_f1_at_threshold(
    pb_data:    List[dict],
    problem_map: List[Tuple[int, int, int]],
    scores_arr: np.ndarray,
    threshold:  float,
) -> Tuple[float, float, float]:
    """
    Compute F1 at a given threshold for one split.

    Official ProcessBench F1 = harmonic mean of:
      acc_error   = accuracy on erroneous problems (label >= 0)
      acc_correct = accuracy on correct problems   (label == -1)

    Returns (f1, acc_error, acc_correct).
    """
    n_error_total   = 0
    n_error_correct = 0
    n_clean_total   = 0
    n_clean_correct = 0

    for prob_idx, start, end in problem_map:
        true_label  = pb_data[prob_idx]["label"]  # -1 or step index
        step_scores = scores_arr[start:end]
        first_pred  = next(
            (i for i, sc in enumerate(step_scores) if sc < threshold), -1
        )

        if true_label >= 0:
            n_error_total += 1
            if first_pred == true_label:
                n_error_correct += 1
        else:
            n_clean_total += 1
            if first_pred == -1:
                n_clean_correct += 1

    acc_error   = n_error_correct / n_error_total   if n_error_total   > 0 else 0.0
    acc_correct = n_clean_correct / n_clean_total   if n_clean_total   > 0 else 0.0
    f1 = (2 * acc_error * acc_correct / (acc_error + acc_correct)
          if (acc_error + acc_correct) > 0 else 0.0)
    return float(f1), float(acc_error), float(acc_correct)


# def evaluate_processbench(
#     model_path:         str,
#     pb_dir:             str,
#     device:             torch.device,
#     batch_size:         int = 64,
#     max_length:         int = 1024,
#     student_model_path: str = STUDENT_MODEL_PATH,
# ) -> dict:
def evaluate_processbench(
    model_path:         str,
    pb_dir:             str,
    device:             torch.device,
    batch_size:         int = 64,
    max_length:         int = 1024,
    student_model_path: str = STUDENT_MODEL_PATH,
    save_step_preds:    str = None,
) -> dict:
    """
    Evaluate DistillPRM on ProcessBench using the official F1 metric.

    Official ProcessBench F1 = harmonic mean of:
      acc_error   : accuracy on erroneous problems (label >= 0)
      acc_correct : accuracy on correct problems   (label == -1)

    The score threshold is tuned on the GSM8K subset (maximises F1),
    then applied to all other subsets — matching the protocol in the
    ProcessBench paper (Zheng et al., 2024) and GenPRM Table 1.
    """
    from transformers import AutoTokenizer

    pb_dir = Path(pb_dir)
    split_names = ["gsm8k", "math", "olympiadbench", "omnimath"]
    split_paths = {s: pb_dir / f"{s}.json" for s in split_names}

    tokenizer = AutoTokenizer.from_pretrained(
        student_model_path, trust_remote_code=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    model = load_model(model_path, device, student_model_path=student_model_path)

    # ── Collect per-step scores for each split ───────────────────────────────
    split_data: dict = {}   # split_name → (pb_data, problem_map, scores_arr)

    for split_name in split_names:
        split_path = split_paths[split_name]
        if not split_path.exists():
            print(f"  Skipping {split_name}: {split_path} not found")
            continue

        with open(split_path) as f:
            pb_data = json.load(f)
        print(f"\n  {split_name}: {len(pb_data)} problems")

        flat_records: List[dict] = []
        problem_map: List[Tuple[int, int, int]] = []

        for prob_idx, prob in enumerate(pb_data):
            problem   = prob["problem"]
            steps     = prob["steps"]
            start_idx = len(flat_records)

            context_parts: List[str] = []
            for step_text in steps:
                flat_records.append({
                    "question":     problem,
                    "context":      "\n\n".join(context_parts),
                    "current_step": step_text,
                    "hard_label":   1,
                    "verification_cot": "",
                })
                context_parts.append(step_text)

            problem_map.append((prob_idx, start_idx, len(flat_records)))

        print(f"    Running inference on {len(flat_records):,} steps ...")
        dataset = DistillPRMDataset(flat_records, tokenizer, max_length=max_length)
        loader  = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = 4,
            collate_fn  = lambda b: collate_fn(b, pad_token_id=pad_id),
            pin_memory  = True,
        )

        all_scores: List[float] = []
        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                student_score, _ = model(input_ids, attention_mask)
                all_scores.extend(student_score.cpu().float().tolist())

        split_data[split_name] = (pb_data, problem_map, np.array(all_scores))

    if not split_data:
        return {}

    # ── Tune threshold on GSM8K ───────────────────────────────────────────────
    ref_split = "gsm8k" if "gsm8k" in split_data else next(iter(split_data))
    ref_pb, ref_map, ref_scores = split_data[ref_split]

    thresholds = np.linspace(0.01, 0.99, 200)
    best_tau, best_f1_ref = 0.5, -1.0
    for tau in thresholds:
        f1, _, _ = _compute_f1_at_threshold(ref_pb, ref_map, ref_scores, tau)
        if f1 > best_f1_ref:
            best_f1_ref = f1
            best_tau    = float(tau)

    print(f"\n  Optimal threshold (tuned on {ref_split}): τ* = {best_tau:.4f}  "
          f"→ {ref_split} F1 = {best_f1_ref:.4f}")

    # ── Apply τ* to all splits ────────────────────────────────────────────────
    results = {}
    f1_values = []

    for split_name, (pb_data, problem_map, scores_arr) in split_data.items():
        f1, acc_err, acc_cor = _compute_f1_at_threshold(
            pb_data, problem_map, scores_arr, best_tau
        )
        results[split_name] = {
            "f1":                   f1,
            "acc_error":            acc_err,
            "acc_correct":          acc_cor,
            "threshold":            best_tau,
            "n_total":              len(pb_data),
            "n_error_problems":     sum(1 for p in pb_data if p["label"] >= 0),
        }
        f1_values.append(f1)
        print(f"    {split_name:15s}: F1={f1:.4f}  "
              f"acc_error={acc_err:.4f}  acc_correct={acc_cor:.4f}")

    # avg_f1 = float(np.mean(f1_values))
    # results["average"] = {"f1": avg_f1}
    # print(f"\n  Average F1 across splits: {avg_f1:.4f}")

    # return results
    avg_f1 = float(np.mean(f1_values))
    results["average"] = {"f1": avg_f1}
    print(f"\n  Average F1 across splits: {avg_f1:.4f}")

    # ── Save step-level predictions if requested ──────────────────────────────
    if save_step_preds is not None:
        step_preds = []
        for split_name, (pb_data, problem_map, scores_arr) in split_data.items():
            for prob_idx, start, end in problem_map:
                label = pb_data[prob_idx]["label"]
                steps = pb_data[prob_idx]["steps"]
                for step_idx, score in enumerate(scores_arr[start:end]):
                    if label == -1:
                        true_label = 1
                    elif step_idx < label:
                        true_label = 1
                    elif step_idx == label:
                        true_label = 0
                    else:
                        true_label = -1
                    pred_label = 1 if score >= best_tau else 0
                    is_wrong = (pred_label != true_label) if true_label != -1 else False
                    step_preds.append({
                        "split":      split_name,
                        "prob_idx":   prob_idx,
                        "step_idx":   step_idx,
                        "step_text":  steps[step_idx],
                        "pred_score": float(score),
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "is_wrong":   is_wrong,
                    })
        out_path = Path(save_step_preds)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(step_preds, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved {len(step_preds)} step predictions → {save_step_preds}")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DistillPRM on held-out data."
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--data_path", default=str(TRAINING_DATA_PATH),
                        help="Path to evaluation data JSON")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--val_frac",   type=float, default=0.05,
                        help="Fraction for val split (if using training data)")
    parser.add_argument("--full_data",  action="store_true",
                        help="Evaluate on full dataset (not just val split)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output",     default=None,
                        help="Save metrics JSON to this path")
    parser.add_argument("--processbench", default=None,
                        help="Run ProcessBench evaluation. Pass path to ProcessBench dir.")
    # parser.add_argument("--student_model", default=STUDENT_MODEL_PATH,
    #                     help="Path to student backbone used when the checkpoint was trained")
    # args = parser.parse_args()
    parser.add_argument("--student_model", default=STUDENT_MODEL_PATH,
                        help="Path to student backbone used when the checkpoint was trained")
    parser.add_argument("--save_step_preds", default=None,
                        help="Save step-level predictions to this JSON path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.processbench:
        print(f"\nProcessBench evaluation: {args.processbench}")
        # pb_metrics = evaluate_processbench(
        #     model_path         = args.model_path,
        #     pb_dir             = args.processbench,
        #     device             = device,
        #     batch_size         = args.batch_size,
        #     max_length         = args.max_length,
        #     student_model_path = args.student_model,
        # )
        pb_metrics = evaluate_processbench(
            model_path         = args.model_path,
            pb_dir             = args.processbench,
            device             = device,
            batch_size         = args.batch_size,
            max_length         = args.max_length,
            student_model_path = args.student_model,
            save_step_preds    = args.save_step_preds,
        )
        print("\n" + "=" * 60)
        print("PROCESSBENCH RESULTS (official F1)")
        print("=" * 60)
        for split, m in pb_metrics.items():
            print(f"  {split:20s}: F1={m['f1']:.4f}", end="")
            if "acc_error" in m:
                print(f"  (acc_error={m['acc_error']:.4f}  acc_correct={m['acc_correct']:.4f})", end="")
            print()
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(pb_metrics, f, indent=2)
            print(f"\nSaved to {out_path}")
        return

    metrics = evaluate_model(
        model_path         = args.model_path,
        data_path          = args.data_path,
        device             = device,
        batch_size         = args.batch_size,
        max_length         = args.max_length,
        val_frac           = args.val_frac,
        seed               = args.seed,
        use_val_split      = not args.full_data,
        student_model_path = args.student_model,
    )

    # Pretty-print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nStep-level classification:")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  AUC-ROC           : {metrics['auc_roc']:.4f}")
    print(f"  ECE (↓)           : {metrics['ece']:.4f}")
    print(f"  Wrong step P/R/F1 : "
          f"{metrics['wrong_step_precision']:.3f} / "
          f"{metrics['wrong_step_recall']:.3f} / "
          f"{metrics['wrong_step_f1']:.3f}")

    print(f"\nError type classification:")
    print(f"  Macro F1          : {metrics['error_macro_f1']:.4f}")

    print(f"\nFirst-error localization:")
    print(f"  Overall acc       : {metrics['localization_acc']:.4f}")
    print(f"  Error-only acc    : {metrics['error_localization_acc']:.4f}")
    print(f"  N conversations   : {metrics['n_conversations']:,}")
    print(f"  N with error      : {metrics['n_with_error']:,}")

    print(f"\nDifficulty breakdown:")
    for bucket in ("easy", "medium", "hard"):
        n_key   = f"{bucket}_n"
        acc_key = f"{bucket}_acc"
        if n_key in metrics:
            print(f"  {bucket:8s}: N={metrics[n_key]:>7,}  acc={metrics[acc_key]:.4f}")

    print(f"\nScore distribution:")
    print(f"  Mean score (correct steps) : {metrics['mean_score_correct']:.4f}")
    print(f"  Mean score (wrong steps)   : {metrics['mean_score_wrong']:.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {out_path}")

    return metrics


if __name__ == "__main__":
    main()
