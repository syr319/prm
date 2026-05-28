"""
Mine hard reasoning steps from the training set using DistillPRM-7B-Instruct.

Scores all records in the training data, retains those where the model is
uncertain (student_score ∈ [--low, --high], default [0.3, 0.7]), and saves
them to a JSON file with 'student_score' and 'sample_weight' fields for
iter2 training.

Usage:
    python3 distill/mine_hard_steps.py \\
        --checkpoint outputs/distillprm-7b-instruct-adaptive-t3/adaptive_t3/best_model.pt \\
        --student_model models/Qwen2.5-Math-7B \\
        --output data/hard_steps_from_instruct.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "distill"))

from step4_build_student_model import DistillPRM
from step5_train_distillpRM import DistillPRMDataset, collate_fn


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def load_checkpoint(checkpoint: Path, student_model: Path, device: torch.device) -> DistillPRM:
    model = DistillPRM(model_name_or_path=str(student_model))
    state = torch.load(str(checkpoint), map_location=device, weights_only=True)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Loaded {n/1e6:.1f}M-param model from {checkpoint}")
    return model


@torch.no_grad()
def score_all(
    model:      DistillPRM,
    records:    list,
    tokenizer,
    device:     torch.device,
    batch_size: int,
    max_length: int,
) -> list:
    ds     = DistillPRMDataset(records, tokenizer, max_length=max_length)
    pad_id = tokenizer.pad_token_id
    loader = DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 4,
        collate_fn  = lambda b: collate_fn(b, pad_token_id=pad_id),
        pin_memory  = True,
    )
    scores = []
    done   = 0
    for batch in loader:
        s, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        scores.extend(s.cpu().float().tolist())
        done += len(s)
        if done % 10000 < batch_size:
            print(f"  {done:>7,} / {len(records):,} scored", flush=True)
    return scores

# def main():
#     parser = argparse.ArgumentParser(description="Mine hard steps with DistillPRM-7B.")
#     parser.add_argument("--checkpoint",
#                         default="models/DistillPRM-7B/adaptive_t3/best_model.pt")
#     parser.add_argument("--student_model",  default="models/Qwen2.5-Math-7B")
#     parser.add_argument("--data_path",      default="data/genprm_math_steps_final.json")
#     parser.add_argument("--output",         default="data/hard_steps_from_7b.json")
#     parser.add_argument("--all_scores_out", default="data/all_scores_7b.json",
#                         help="Path to save all scores for distribution analysis")
#     parser.add_argument("--batch_size",     type=int,   default=16)
#     parser.add_argument("--max_length",     type=int,   default=1024)
#     parser.add_argument("--top_k_ratio",    type=float, default=0.25,
#                         help="Fraction of most uncertain samples to keep (default: 25%)")
#     parser.add_argument("--sample_weight",  type=float, default=3.0,
#                         help="Weight assigned to hard steps in iter2 training")
#     args = parser.parse_args()

#     checkpoint    = _resolve(args.checkpoint)
#     student_model = _resolve(args.student_model)
#     data_path     = _resolve(args.data_path)
#     output        = _resolve(args.output)
#     all_scores_out = _resolve(args.all_scores_out)
#     device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     print(f"Device        : {device}")
#     print(f"Checkpoint    : {checkpoint}")
#     print(f"Student model : {student_model}")
#     print(f"Data          : {data_path}")
#     print(f"Top-K ratio   : {args.top_k_ratio}  sample_weight={args.sample_weight}")

#     tokenizer = AutoTokenizer.from_pretrained(str(student_model), trust_remote_code=True)
#     tokenizer.padding_side = "right"
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     model = load_checkpoint(checkpoint, student_model, device)

#     print(f"\nLoading {data_path} ...")
#     with open(data_path, encoding="utf-8") as f:
#         records = json.load(f)
#     records = [r for r in records if r.get("hard_label") in (0, 1)]
#     print(f"Total records: {len(records):,}")

#     print(f"\nScoring (batch_size={args.batch_size}) ...")
#     scores = score_all(model, records, tokenizer, device, args.batch_size, args.max_length)

#     # ============ 全量分数分布统计 ============
#     import numpy as np
#     scores_arr = np.array(scores)

#     print(f"\n{'='*50}")
#     print(f"全量分数分布统计 (n={len(scores_arr):,})")
#     print(f"{'='*50}")
#     print(f"  Mean:   {scores_arr.mean():.4f}")
#     print(f"  Median: {np.median(scores_arr):.4f}")
#     print(f"  Std:    {scores_arr.std():.4f}")
#     print(f"  Min:    {scores_arr.min():.4f}  Max: {scores_arr.max():.4f}")
#     print()
#     print(f"  [0.0, 0.1]: {((scores_arr>=0.0)&(scores_arr<0.1)).sum():>7,}")
#     print(f"  [0.1, 0.2]: {((scores_arr>=0.1)&(scores_arr<0.2)).sum():>7,}")
#     print(f"  [0.2, 0.3]: {((scores_arr>=0.2)&(scores_arr<0.3)).sum():>7,}")
#     print(f"  [0.3, 0.4]: {((scores_arr>=0.3)&(scores_arr<0.4)).sum():>7,}")
#     print(f"  [0.4, 0.5]: {((scores_arr>=0.4)&(scores_arr<0.5)).sum():>7,}")
#     print(f"  [0.5, 0.6]: {((scores_arr>=0.5)&(scores_arr<0.6)).sum():>7,}")
#     print(f"  [0.6, 0.7]: {((scores_arr>=0.6)&(scores_arr<0.7)).sum():>7,}")
#     print(f"  [0.7, 0.8]: {((scores_arr>=0.7)&(scores_arr<0.8)).sum():>7,}")
#     print(f"  [0.8, 0.9]: {((scores_arr>=0.8)&(scores_arr<0.9)).sum():>7,}")
#     print(f"  [0.9, 1.0]: {((scores_arr>=0.9)&(scores_arr<=1.0)).sum():>7,}")

#     # 保存全量分数
#     all_scores_data = [
#         {"student_score": round(s, 6), "hard_label": r.get("hard_label")}
#         for r, s in zip(records, scores)
#     ]
#     all_scores_out.parent.mkdir(parents=True, exist_ok=True)
#     with open(all_scores_out, "w", encoding="utf-8") as f:
#         json.dump(all_scores_data, f)
#     print(f"\n全量分数已保存 → {all_scores_out}")

#     # 画直方图
#     try:
#         import matplotlib
#         matplotlib.use('Agg')
#         import matplotlib.pyplot as plt
#         plt.figure(figsize=(10, 6))
#         plt.hist(scores_arr, bins=50, edgecolor='black', alpha=0.7)
#         plt.xlabel('Student Score')
#         plt.ylabel('Count')
#         plt.title('Score Distribution (n={})'.format(len(scores_arr)))
#         plt.legend()
#         plt.tight_layout()
#         fig_path = output.parent / "score_distribution_7b.png"
#         plt.savefig(str(fig_path), dpi=150)
#         print(f"直方图已保存 → {fig_path}")
#     except Exception as e:
#         print(f"画图失败(不影响): {e}")

#     # ============ Top-K 筛选 hard steps ============
#     uncertainty = 1.0 - np.abs(2.0 * scores_arr - 1.0)  # 越接近0.5越不确定
#     top_k = int(len(scores) * args.top_k_ratio)
#     hard_indices = np.argsort(uncertainty)[-top_k:]  # 取 uncertainty 最高的

#     hard = []
#     for idx in hard_indices:
#         out = dict(records[int(idx)])
#         out["student_score"] = round(scores[int(idx)], 6)
#         out["sample_weight"] = args.sample_weight
#         hard.append(out)

#     label0 = sum(1 for r in hard if r["hard_label"] == 0)
#     label1 = sum(1 for r in hard if r["hard_label"] == 1)
#     hard_scores = np.array([r["student_score"] for r in hard])

#     print(f"\n{'='*50}")
#     print(f"Hard steps (Top-K {args.top_k_ratio*100:.0f}%): {len(hard):,} / {len(records):,}")
#     print(f"{'='*50}")
#     print(f"  label=0: {label0:,}")
#     print(f"  label=1: {label1:,}")
#     print(f"  score range: [{hard_scores.min():.4f}, {hard_scores.max():.4f}]")
#     print(f"  score mean:  {hard_scores.mean():.4f}")

#     output.parent.mkdir(parents=True, exist_ok=True)
#     with open(output, "w", encoding="utf-8") as f:
#         json.dump(hard, f, ensure_ascii=False)
#     print(f"\nSaved → {output}")


# if __name__ == "__main__":
#     main()
def main():
    parser = argparse.ArgumentParser(description="Mine hard steps with DistillPRM-7B.")
    parser.add_argument("--checkpoint",
                        default="models/DistillPRM-7B/adaptive_t3/best_model.pt")
    parser.add_argument("--student_model",  default="models/Qwen2.5-Math-7B")
    parser.add_argument("--data_path",      default="data/genprm_math_steps_final.json")
    parser.add_argument("--output",         default="data/hard_steps_from_7b.json")
    parser.add_argument("--all_scores_out", default="data/all_scores_7b.json",
                        help="Path to save all scores for distribution analysis")
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--max_length",     type=int,   default=1024)
    parser.add_argument("--threshold",      type=float, default=0.5,
                        help="Decision threshold for pred (default: 0.5)")
    parser.add_argument("--sample_weight",  type=float, default=2.0,
                        help="Weight assigned to hard steps in iter2 training")
    args = parser.parse_args()

    checkpoint    = _resolve(args.checkpoint)
    student_model = _resolve(args.student_model)
    data_path     = _resolve(args.data_path)
    output        = _resolve(args.output)
    all_scores_out = _resolve(args.all_scores_out)
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device        : {device}")
    print(f"Checkpoint    : {checkpoint}")
    print(f"Student model : {student_model}")
    print(f"Data          : {data_path}")
    print(f"Strategy      : misclassified samples (threshold={args.threshold})")
    print(f"sample_weight : {args.sample_weight}")

    tokenizer = AutoTokenizer.from_pretrained(str(student_model), trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_checkpoint(checkpoint, student_model, device)

    print(f"\nLoading {data_path} ...")
    with open(data_path, encoding="utf-8") as f:
        records = json.load(f)
    records = [r for r in records if r.get("hard_label") in (0, 1)]
    print(f"Total records: {len(records):,}")

    print(f"\nScoring (batch_size={args.batch_size}) ...")
    scores = score_all(model, records, tokenizer, device, args.batch_size, args.max_length)

    # ============ 全量分数统计 ============
    import numpy as np
    scores_arr = np.array(scores)

    print(f"\n{'='*50}")
    print(f"全量分数分布统计 (n={len(scores_arr):,})")
    print(f"{'='*50}")
    print(f"  Mean:   {scores_arr.mean():.4f}")
    print(f"  Median: {np.median(scores_arr):.4f}")
    print(f"  Std:    {scores_arr.std():.4f}")
    print(f"  Min:    {scores_arr.min():.4f}  Max: {scores_arr.max():.4f}")

    # 保存全量分数
    all_scores_data = [
        {"student_score": round(s, 6), "hard_label": r.get("hard_label")}
        for r, s in zip(records, scores)
    ]
    all_scores_out.parent.mkdir(parents=True, exist_ok=True)
    with open(all_scores_out, "w", encoding="utf-8") as f:
        json.dump(all_scores_data, f)
    print(f"全量分数已保存 → {all_scores_out}")

    # ============ 方案B：选模型判错的样本 ============
    hard = []
    # 统计
    tp, fp, tn, fn = 0, 0, 0, 0

    for rec, score in zip(records, scores):
        label = rec["hard_label"]
        pred = 1 if score >= args.threshold else 0

        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:  # pred == 0 and label == 1
            fn += 1

        # 预测错误的样本 = hard sample
        if pred != label:
            out = dict(rec)
            out["student_score"] = round(score, 6)
            out["sample_weight"] = args.sample_weight
            hard.append(out)

    total = len(records)
    correct = tp + tn
    wrong = fp + fn
    acc = correct / total * 100

    print(f"\n{'='*50}")
    print(f"模型在训练集上的预测情况 (threshold={args.threshold})")
    print(f"{'='*50}")
    print(f"  Accuracy: {acc:.2f}% ({correct:,} / {total:,})")
    print(f"  TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")
    print(f"  错误预测(hard samples): {wrong:,} ({wrong/total*100:.1f}%)")

    label0 = sum(1 for r in hard if r["hard_label"] == 0)
    label1 = sum(1 for r in hard if r["hard_label"] == 1)
    hard_scores = np.array([r["student_score"] for r in hard])

    print(f"\n{'='*50}")
    print(f"Hard steps (misclassified): {len(hard):,} / {len(records):,} ({len(hard)/len(records)*100:.1f}%)")
    print(f"{'='*50}")
    print(f"  label=0 (FP, 实际错但模型判对): {label0:,}")
    print(f"  label=1 (FN, 实际对但模型判错): {label1:,}")
    if len(hard_scores) > 0:
        print(f"  score range: [{hard_scores.min():.4f}, {hard_scores.max():.4f}]")
        print(f"  score mean:  {hard_scores.mean():.4f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(hard, f, ensure_ascii=False)
    print(f"\nSaved → {output}")


if __name__ == "__main__":
    main()