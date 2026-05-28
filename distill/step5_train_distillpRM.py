"""
Step 5: Train DistillPRM student model using GenPRM soft labels.

Training modes (--mode):
  ce        → CE-only baseline (no knowledge distillation)
  kl        → CE + KL with fixed 50/50 blend (standard KD)
  adaptive  → Difficulty-adaptive distillation (default, full DistillPRM)

All modes include the auxiliary error type classification head with focal loss
and class-frequency weighting to handle the reasoning_error class imbalance.

Distributed training:
  Launch with torchrun:
    torchrun --nproc_per_node=8 distill/step5_train_distillpRM.py --mode adaptive

  Single-GPU fallback (no torchrun):
    python3 distill/step5_train_distillpRM.py --mode ce

Input data: data/genprm_math_steps_final.json  (preferred)
  Each record: question, context, current_step, verification_cot,
               hard_label, step_index, total_steps, soft_score, error_type
  Falls back to data/genprm_math_steps_with_soft_scores.json if final not found.

Output: models/DistillPRM-1.5B/{mode}/  (checkpoints + final model)
"""

import argparse
import datetime
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "distill"))

from step4_build_student_model import (
    DistillPRM,
    adaptive_score_loss,
    error_type_focal_loss,
    build_input_text,
    NUM_ERROR_TYPES,
    STUDENT_MODEL_PATH,
)

# ─── Paths ────────────────────────────────────────────────────────────────────

_FINAL_DATA = ROOT / "data" / "genprm_math_steps_final.json"
_FALLBACK   = ROOT / "data" / "genprm_math_steps_with_soft_scores.json"
DATA_PATH   = _FINAL_DATA if _FINAL_DATA.exists() else _FALLBACK
OUTPUT_DIR  = ROOT / "models" / "DistillPRM-1.5B"


# ─── Error type extraction (fallback for records without error_type field) ────

_ERROR_KEYWORDS: Dict[int, List[str]] = {
    1: [
        "calculat", "arithmet", "numer", "computational", "digit",
        "multipl", "divid", "summ", "added", "subtracted",
        "wrong number", "incorrect number", "arithmetic error",
    ],
    2: [
        "algebr", "simplif", "factor", "expand", "distribut",
        "coefficient", "polynomial", "expression", "substitut",
        "algebraic manipulation", "equation",
    ],
    3: [
        "missing step", "skip", "jump", "gap", "not shown", "not proven",
        "incomplete", "without justif", "no justif", "logical leap",
        "direct claim", "does not follow",
    ],
    4: [
        "previous step", "prior step", "earlier step", "from step",
        "incorrect value from", "wrong value from", "uses the result",
        "referenced", "relied on",
    ],
    5: [
        "formula", "theorem", "definition", "concept", "principle",
        "wrong formula", "incorrect formula", "misappl", "conceptual",
        "rule", "law", "misunderstood",
    ],
    6: [
        "irrelevant", "unrelated", "does not contribute",
        "not needed", "unnecessary", "not related",
    ],
}

_OLD_TO_NEW: Dict[int, int] = {
    0: 0, 1: 1, 2: 1, 3: 3, 4: 2, 5: 3, 6: 3,
}


def extract_error_type(cot_text: str, hard_label: int) -> int:
    """Fallback: returns 4-class label (0–3) from CoT text."""
    if hard_label == 1:
        return 0
    if not cot_text:
        return 1
    low = cot_text.lower()
    scores = {e: sum(1 for kw in kws if kw in low)
              for e, kws in _ERROR_KEYWORDS.items()}
    best = max(scores, key=lambda k: scores[k])
    return _OLD_TO_NEW[best if scores[best] > 0 else 1]


# ─── Dataset ──────────────────────────────────────────────────────────────────

class DistillPRMDataset(Dataset):
    def __init__(self, records: List[dict], tokenizer, max_length: int = 1024):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        text = build_input_text(
            question     = r.get("question", ""),
            context      = r.get("context", ""),
            current_step = r.get("current_step", ""),
        )
        enc = self.tokenizer(
            text,
            max_length     = self.max_length,
            truncation     = True,
            padding        = False,
            return_tensors = "pt",
        )
        hard_label  = float(r["hard_label"])
        soft_score  = float(r.get("soft_score", hard_label))
        if "error_type" in r:
            error_label = int(r["error_type"])
        else:
            error_label = extract_error_type(
                r.get("verification_cot", ""), int(r["hard_label"])
            )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "hard_label":     torch.tensor(hard_label,  dtype=torch.float32),
            "soft_score":     torch.tensor(soft_score,  dtype=torch.float32),
            "error_label":    torch.tensor(error_label, dtype=torch.long),
             # 新增字段
            "step_index":     torch.tensor(r.get("step_index", 0),   dtype=torch.float32),
            "total_steps":    torch.tensor(r.get("total_steps", 1),   dtype=torch.float32),
            "step_length":    torch.tensor(len(r.get("current_step", "")), dtype=torch.float32),
        }


def collate_fn(batch, pad_token_id: int = 0):
    """Right-pad sequences in batch to the same length."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_list, attention_mask_list = [], []
    hard_labels, soft_scores, error_labels = [], [], []
    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids_list.append(F.pad(item["input_ids"],      (0, pad_len), value=pad_token_id))
        attention_mask_list.append(F.pad(item["attention_mask"], (0, pad_len), value=0))
        hard_labels.append(item["hard_label"])
        soft_scores.append(item["soft_score"])
        error_labels.append(item["error_label"])
    return {
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "hard_label":     torch.stack(hard_labels),
        "soft_score":     torch.stack(soft_scores),
        "error_label":    torch.stack(error_labels),
    }


# ─── Class weight computation ─────────────────────────────────────────────────

def compute_class_weights(
    records:   List[dict],
    n_classes: int,
    scheme:    str,
    device:    torch.device,
) -> Optional[torch.Tensor]:
    """
    Compute per-class weights for the error_type focal loss.

    Schemes:
      none          — no weighting (plain CE)
      inv_freq      — w_k = N / (K * n_k)
      sqrt_inv_freq — w_k = sqrt(N / n_k), normalised to mean = 1
    """
    if scheme == "none":
        return None
    counts = [0] * n_classes
    for r in records:
        et = int(r.get("error_type", 0))
        if 0 <= et < n_classes:
            counts[et] += 1
    total = sum(counts)
    if scheme == "inv_freq":
        weights = [total / (n_classes * max(c, 1)) for c in counts]
    elif scheme == "sqrt_inv_freq":
        weights = [math.sqrt(total / max(c, 1)) for c in counts]
        mean_w  = sum(weights) / len(weights)
        weights = [w / mean_w for w in weights]
    else:
        raise ValueError(f"Unknown weight scheme: {scheme}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ─── Loss helper ──────────────────────────────────────────────────────────────

_ADAPTIVE_MODES = {"adaptive", "adaptive_t2", "adaptive_t3", "adaptive_t5",
                   "ablation_no_error_head"}

# Map mode name → difficulty temperature (1.0 = original linear formula)
_MODE_TEMPERATURE: Dict[str, float] = {
    "adaptive":               1.0,
    "adaptive_t2":            2.0,
    "adaptive_t3":            3.0,
    "adaptive_t5":            5.0,
    "ablation_no_error_head": 3.0,   # same T as adaptive_t3; error head disabled via lambda_error=0
    "adaptive_multidim":      3.0,   # 方向1
    "adaptive_temp":          3.0,   # 方向2
}


def compute_score_loss(
    mode:                   str,
    student_score:          torch.Tensor,
    soft_score:             torch.Tensor,
    hard_label:             torch.Tensor,
    difficulty_temperature: float = 1.0,
) -> torch.Tensor:
    if mode == "ce":
        return F.binary_cross_entropy(student_score, hard_label)
    elif mode == "kl":
        return adaptive_score_loss(student_score, soft_score, hard_label,
                                   alpha_easy=0.5, alpha_hard=0.5)
    else:  # adaptive / adaptive_t2 / adaptive_t3 / adaptive_t5
        return adaptive_score_loss(student_score, soft_score, hard_label,
                                   alpha_easy=0.9, alpha_hard=0.1,
                                   difficulty_temperature=difficulty_temperature)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_ece(preds: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (preds >= lo) & (preds < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(preds[mask].mean() - labels[mask].mean())
    return float(ece / len(preds))


def _compute_error_f1(pred: np.ndarray, true: np.ndarray, n_classes: int) -> dict:
    result = {}
    f1s = []
    for c in range(n_classes):
        tp = ((pred == c) & (true == c)).sum()
        fp = ((pred == c) & (true != c)).sum()
        fn = ((pred != c) & (true == c)).sum()
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        result[f"error_f1_c{c}"] = float(f1)
        result[f"error_n_c{c}"]  = int((true == c).sum())
        f1s.append(f1)
    result["error_macro_f1"] = float(np.mean(f1s))
    return result


# ─── Optimizer ────────────────────────────────────────────────────────────────

def build_optimizer(
    model:           nn.Module,
    backbone_lr:     float,
    head_lr:         float,
    weight_decay:    float,
    freeze_backbone: bool,
) -> torch.optim.Optimizer:
    """Build AdamW with different LRs for backbone and heads. Handles DDP wrapper."""
    m = model.module if hasattr(model, "module") else model
    if freeze_backbone:
        for p in m.backbone.parameters():
            p.requires_grad = False
        head_params = list(m.score_head.parameters()) + list(m.error_head.parameters())
        return torch.optim.AdamW(head_params, lr=head_lr, weight_decay=weight_decay)
    param_groups = [
        {"params": m.backbone.parameters(),                                        "lr": backbone_lr},
        {"params": list(m.score_head.parameters()) + list(m.error_head.parameters()), "lr": head_lr},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:                  nn.Module,
    loader:                 DataLoader,
    device:                 torch.device,
    mode:                   str,
    lambda_error:           float,
    class_weights:          Optional[torch.Tensor] = None,
    focal_gamma:            float = 2.0,
    difficulty_temperature: float = 1.0,
) -> dict:
    """Run one evaluation pass and return metrics dict. Called on rank 0 only."""
    model.eval()
    total_loss = total_score_loss = total_err_loss = 0.0
    all_scores:      List[float] = []
    all_hard_labels: List[int]   = []
    all_error_pred:  List[int]   = []
    all_error_true:  List[int]   = []
    n_batches = 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        hard_label     = batch["hard_label"].to(device)
        soft_score     = batch["soft_score"].to(device)
        error_label    = batch["error_label"].to(device)

        student_score, error_logits = model(input_ids, attention_mask)

        l_score = compute_score_loss(mode, student_score, soft_score, hard_label,
                                     difficulty_temperature=difficulty_temperature)
        l_error = error_type_focal_loss(error_logits, error_label,
                                        class_weights=class_weights, gamma=focal_gamma)
        l_total = l_score + lambda_error * l_error

        total_loss       += l_total.item()
        total_score_loss += l_score.item()
        total_err_loss   += l_error.item()
        n_batches        += 1

        all_scores.extend(student_score.cpu().float().tolist())
        all_hard_labels.extend(hard_label.cpu().int().tolist())
        all_error_pred.extend(error_logits.argmax(dim=-1).cpu().tolist())
        all_error_true.extend(error_label.cpu().tolist())

    preds  = np.array(all_scores)
    labels = np.array(all_hard_labels)
    acc    = ((preds >= 0.5).astype(int) == labels).mean()
    ece    = compute_ece(preds, labels)

    ep = np.array(all_error_pred)
    et = np.array(all_error_true)
    error_metrics = _compute_error_f1(ep, et, NUM_ERROR_TYPES)

    result = {
        "loss":       total_loss       / n_batches,
        "loss_score": total_score_loss / n_batches,
        "loss_error": total_err_loss   / n_batches,
        "accuracy":   float(acc),
        "ece":        ece,
    }
    result.update(error_metrics)
    return result


# ─── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(
    model:                  nn.Module,
    loader:                 DataLoader,
    optimizer:              torch.optim.Optimizer,
    scheduler,
    device:                 torch.device,
    mode:                   str,
    lambda_error:           float,
    grad_accum:             int,
    grad_clip:              float,
    epoch:                  int,
    logger:                 logging.Logger,
    log_every:              int = 50,
    class_weights:          Optional[torch.Tensor] = None,
    focal_gamma:            float = 2.0,
    rank:                   int = 0,
    is_dist:                bool = False,
    difficulty_temperature: float = 1.0,
) -> dict:
    """Train for one epoch across all DDP ranks. Returns averaged metrics."""
    model.train()
    total_loss = total_score_loss = total_err_loss = 0.0
    n_steps = 0
    t0 = time.time()
    optimizer.zero_grad()

    for step_idx, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        hard_label     = batch["hard_label"].to(device)
        soft_score     = batch["soft_score"].to(device)
        error_label    = batch["error_label"].to(device)

        student_score, error_logits = model(input_ids, attention_mask)

        l_score = compute_score_loss(mode, student_score, soft_score, hard_label,
                                     difficulty_temperature=difficulty_temperature)
        l_error = error_type_focal_loss(error_logits, error_label,
                                        class_weights=class_weights, gamma=focal_gamma)
        l_total = l_score + lambda_error * l_error

        (l_total / grad_accum).backward()

        total_loss       += l_total.item()
        total_score_loss += l_score.item()
        total_err_loss   += l_error.item()
        n_steps          += 1

        if (step_idx + 1) % grad_accum == 0:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if rank == 0 and (step_idx + 1) % log_every == 0:
            avg_loss = total_loss / n_steps
            lr_now   = scheduler.get_last_lr()[0]
            elapsed  = time.time() - t0
            logger.info(
                f"Epoch {epoch}  step {step_idx+1}/{len(loader)}"
                f"  loss={avg_loss:.4f}"
                f"  l_score={total_score_loss/n_steps:.4f}"
                f"  l_error={total_err_loss/n_steps:.4f}"
                f"  lr={lr_now:.2e}"
                f"  elapsed={elapsed:.0f}s"
            )

    # Flush remaining gradients in incomplete accumulation window
    remaining = len(loader) % grad_accum
    if remaining != 0:
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # All-reduce train metrics so rank 0 logs the global average
    if is_dist and dist.is_initialized():
        loss_tensor = torch.tensor(
            [total_loss, total_score_loss, total_err_loss, float(n_steps)],
            device=device,
        )
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss       = loss_tensor[0].item()
        total_score_loss = loss_tensor[1].item()
        total_err_loss   = loss_tensor[2].item()
        n_steps          = loss_tensor[3].item()

    return {
        "loss":       total_loss       / n_steps,
        "loss_score": total_score_loss / n_steps,
        "loss_error": total_err_loss   / n_steps,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train DistillPRM student model.")
    parser.add_argument("--mode",         default="adaptive",
                        choices=["ce", "kl", "adaptive",
                                 "adaptive_t2", "adaptive_t3", "adaptive_t5",
                                 "ablation_no_error_head", "adaptive_multidim", "adaptive_temp"],)
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--batch_size",   type=int,   default=16, help="Per-device batch size")
    parser.add_argument("--grad_accum",   type=int,   default=4,  help="Gradient accumulation steps")
    parser.add_argument("--backbone_lr",  type=float, default=1e-5)
    parser.add_argument("--head_lr",      type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_length",   type=int,   default=1024)
    parser.add_argument("--lambda_error", type=float, default=0.1)
    parser.add_argument("--focal_gamma",  type=float, default=2.0,
                        help="Focal loss gamma (0 = plain CE, 2 = standard focal)")
    parser.add_argument("--weight_scheme", default="sqrt_inv_freq",
                        choices=["none", "inv_freq", "sqrt_inv_freq"])
    parser.add_argument("--val_frac",     type=float, default=0.05)
    parser.add_argument("--grad_clip",    type=float, default=1.0)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--data_path",    default=str(DATA_PATH))
    parser.add_argument("--output_dir",   default=str(OUTPUT_DIR))
    parser.add_argument("--student_model", default=STUDENT_MODEL_PATH,
                        help="Path to student backbone (default: Qwen2.5-Math-1.5B)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing (saves VRAM, ~30%% slower)")
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--resume_model_only", action="store_true",
                        help="Resume model weights only, skip optimizer/scheduler (avoids OOM)")
    parser.add_argument("--log_every",    type=int,   default=50)
    # 方向1参数
    parser.add_argument("--w_score",    type=float, default=1.0,
                        help="Weight for score-based difficulty (default=1.0, original behavior)")
    parser.add_argument("--w_position", type=float, default=0.0,
                        help="Weight for position-based difficulty (0=disabled)")
    parser.add_argument("--w_length",   type=float, default=0.0,
                        help="Weight for length-based difficulty (0=disabled)")

    # 方向2参数
    parser.add_argument("--adaptive_temperature", action="store_true",
                        help="Use per-sample adaptive temperature instead of fixed T")
    parser.add_argument("--t_min", type=float, default=1.0)
    parser.add_argument("--t_max", type=float, default=5.0)
    args = parser.parse_args()

    # Derive difficulty temperature from mode name (adaptive_tN → N.0)
    diff_temp = _MODE_TEMPERATURE.get(args.mode, 1.0)

    # ── Distributed setup ──────────────────────────────────────────────────────
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device     = torch.device("cuda", local_rank)
        is_dist    = True
    else:
        local_rank = 0
        rank       = 0
        world_size = 1
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_dist    = False

    # ── Logging (rank 0 only writes to file/stdout) ───────────────────────────
    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"

    handlers: list = [logging.FileHandler(log_path)] if rank == 0 else []
    if rank == 0:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level   = logging.INFO if rank == 0 else logging.WARNING,
        format  = "%(asctime)s [%(levelname)s] %(message)s",
        handlers= handlers,
    )
    logger = logging.getLogger(__name__)

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if rank == 0:
        logger.info(f"Device: {device}  world_size={world_size}")
        logger.info(f"Mode: {args.mode}  difficulty_temperature={diff_temp}")
        logger.info(
            f"Effective batch size: {args.batch_size} (per-GPU)"
            f" × {args.grad_accum} (grad_accum)"
            f" × {world_size} (GPUs)"
            f" = {args.batch_size * args.grad_accum * world_size}"
        )

    # ── Load data ─────────────────────────────────────────────────────────────
    data_path = Path(args.data_path)
    if not data_path.exists():
        if rank == 0:
            logger.error(f"Data file not found: {data_path}. Run step3 first.")
        sys.exit(1)

    if rank == 0:
        logger.info(f"Loading data from {data_path} ...")
    with open(data_path, encoding="utf-8") as f:
        records = json.load(f)
    records = [r for r in records if r.get("hard_label") in (0, 1)]
    if rank == 0:
        n_soft = sum(1 for r in records if "soft_score" in r)
        logger.info(
            f"Loaded {len(records):,} records  "
            f"({n_soft:,} with soft_score, {n_soft/len(records)*100:.1f}%)"
        )

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    if rank == 0:
        logger.info(f"Loading tokenizer from {args.student_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # ── Dataset split ─────────────────────────────────────────────────────────
    dataset = DistillPRMDataset(records, tokenizer, max_length=args.max_length)
    n_val   = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    if rank == 0:
        logger.info(f"Train: {n_train:,}  Val: {n_val:,}")

    _collate = lambda b: collate_fn(b, pad_token_id=pad_id)

    # Train loader: use DistributedSampler when running DDP
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_dist else None
    train_loader  = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = (train_sampler is None),
        sampler     = train_sampler,
        num_workers = args.num_workers,
        collate_fn  = _collate,
        pin_memory  = True,
    )

    # Val loader: only rank 0 evaluates on the full val set
    val_loader = None
    if rank == 0:
        val_loader = DataLoader(
            val_ds,
            batch_size  = args.batch_size * 2,
            shuffle     = False,
            num_workers = args.num_workers,
            collate_fn  = _collate,
            pin_memory  = True,
        )

    # ── Class weights for focal loss ──────────────────────────────────────────
    # Use training record indices from random_split for accurate counts.
    train_indices  = set(train_ds.indices)  # type: ignore[attr-defined]
    train_records  = [records[i] for i in train_indices]
    class_weights  = compute_class_weights(
        train_records, NUM_ERROR_TYPES, args.weight_scheme, device
    )
    if rank == 0:
        if class_weights is not None:
            counts = [sum(1 for r in train_records if int(r.get("error_type", 0)) == c)
                      for c in range(NUM_ERROR_TYPES)]
            logger.info(
                f"Error class weights ({args.weight_scheme}):\n"
                + "\n".join(
                    f"  class {c}: n={counts[c]:>7,}  weight={class_weights[c]:.3f}"
                    for c in range(NUM_ERROR_TYPES)
                )
            )
        logger.info(f"Focal gamma: {args.focal_gamma}")

    # ── Model ─────────────────────────────────────────────────────────────────
    if rank == 0:
        logger.info(f"Loading DistillPRM backbone from {args.student_model} ...")
    model = DistillPRM(model_name_or_path=args.student_model)
    model.to(device)

    if args.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable()
        if rank == 0:
            logger.info("Gradient checkpointing enabled.")

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {n_params/1e6:.1f}M")

    # ── Optimizer & scheduler (before DDP wrap, but after to(device)) ─────────
    optimizer = build_optimizer(
        model,
        backbone_lr     = args.backbone_lr,
        head_lr         = args.head_lr,
        weight_decay    = args.weight_decay,
        freeze_backbone = args.freeze_backbone,
    )

    total_opt_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps    = int(total_opt_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_opt_steps,
    )
    if rank == 0:
        logger.info(f"Optimizer steps: {total_opt_steps}  Warmup: {warmup_steps}")

    # ── Resume checkpoint ─────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")
    best_ckpt     = output_dir / "best_model.pt"
    metrics_path  = output_dir / "metrics.json"
    all_metrics:  List[dict] = []
    
    if args.resume or args.resume_model_only:
        if args.resume_model_only:
            # 只加载模型权重，不加载optimizer/scheduler state，避免OOM
            # 优先用最新的epoch_*.pt，fallback到best_model.pt
            _epoch_ckpts = sorted(output_dir.glob("epoch_*.pt"))
            _resume_candidate = _epoch_ckpts[-1] if _epoch_ckpts else (
                output_dir / "best_model.pt" if (output_dir / "best_model.pt").exists() else None
            )
        else:
            # 完整resume：优先resume.pt，fallback到最新epoch_*.pt
            _resume_candidate = output_dir / "resume.pt"
            if not _resume_candidate.exists():
                _epoch_ckpts = sorted(output_dir.glob("epoch_*.pt"))
                _resume_candidate = _epoch_ckpts[-1] if _epoch_ckpts else None

        if _resume_candidate is not None:
            if rank == 0:
                logger.info(f"Resuming from {_resume_candidate} ...")
            ckpt = torch.load(_resume_candidate, map_location=device)
            model.load_state_dict(ckpt["model"])
            if args.resume_model_only:
                if rank == 0:
                    logger.info("Model-only resume: optimizer/scheduler start fresh.")
            else:
                if "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])
                else:
                    if rank == 0:
                        logger.warning("Checkpoint has no optimizer state — optimizer starts fresh.")
                if "scheduler" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch   = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
            if rank == 0 and metrics_path.exists():
                with open(metrics_path) as f:
                    all_metrics = json.load(f)
            if rank == 0:
                logger.info(f"Resumed from epoch {start_epoch - 1}  "
                            f"best_val_loss={best_val_loss:.4f}")
        elif rank == 0:
            logger.warning("No checkpoints found; starting from scratch.")
    # if args.resume:
    #     # Prefer resume.pt (has optimizer/scheduler state).
    #     # Fall back to latest epoch_*.pt (model only — optimizer restarts fresh).
    #     _resume_candidate = output_dir / "resume.pt"
    #     if not _resume_candidate.exists():
    #         _epoch_ckpts = sorted(output_dir.glob("epoch_*.pt"))
    #         _resume_candidate = _epoch_ckpts[-1] if _epoch_ckpts else None

    #     if _resume_candidate is not None:
    #         if rank == 0:
    #             logger.info(f"Resuming from {_resume_candidate} ...")
    #         ckpt = torch.load(_resume_candidate, map_location=device)
    #         model.load_state_dict(ckpt["model"])
    #         if "optimizer" in ckpt:
    #             optimizer.load_state_dict(ckpt["optimizer"])
    #         else:
    #             if rank == 0:
    #                 logger.warning("Checkpoint has no optimizer state — optimizer starts fresh.")
    #         if "scheduler" in ckpt:
    #             scheduler.load_state_dict(ckpt["scheduler"])
    #         start_epoch   = ckpt["epoch"] + 1
    #         best_val_loss = ckpt.get("best_val_loss", float("inf"))
    #         if rank == 0 and metrics_path.exists():
    #             with open(metrics_path) as f:
    #                 all_metrics = json.load(f)
    #         if rank == 0:
    #             logger.info(f"Resumed from epoch {start_epoch - 1}.")
    #     elif rank == 0:
    #         logger.warning("No checkpoints found; starting from scratch.")

    # ── Wrap with DDP after potential checkpoint load ─────────────────────────
    if is_dist:
        model = DDP(model, device_ids=[local_rank])

    # ── Training ──────────────────────────────────────────────────────────────
    if rank == 0:
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        # DistributedSampler needs set_epoch for correct shuffling each epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{args.epochs}")
            logger.info(f"{'='*60}")

        train_metrics = train_one_epoch(
            model                  = model,
            loader                 = train_loader,
            optimizer              = optimizer,
            scheduler              = scheduler,
            device                 = device,
            mode                   = args.mode,
            lambda_error           = args.lambda_error,
            grad_accum             = args.grad_accum,
            grad_clip              = args.grad_clip,
            epoch                  = epoch + 1,
            logger                 = logger,
            log_every              = args.log_every,
            class_weights          = class_weights,
            focal_gamma            = args.focal_gamma,
            rank                   = rank,
            is_dist                = is_dist,
            difficulty_temperature = diff_temp,
        )

        # Validation and checkpointing: rank 0 only
        if rank == 0:
            raw_model = model.module if is_dist else model
            val_metrics = evaluate(
                model                  = raw_model,
                loader                 = val_loader,
                device                 = device,
                mode                   = args.mode,
                lambda_error           = args.lambda_error,
                class_weights          = class_weights,
                focal_gamma            = args.focal_gamma,
                difficulty_temperature = diff_temp,
            )

            logger.info(
                f"\nEpoch {epoch+1} summary:\n"
                f"  Train  loss={train_metrics['loss']:.4f}"
                f"  l_score={train_metrics['loss_score']:.4f}"
                f"  l_error={train_metrics['loss_error']:.4f}\n"
                f"  Val    loss={val_metrics['loss']:.4f}"
                f"  l_score={val_metrics['loss_score']:.4f}"
                f"  l_error={val_metrics['loss_error']:.4f}"
                f"  acc={val_metrics['accuracy']:.4f}"
                f"  ece={val_metrics['ece']:.4f}"
                f"  macro_f1={val_metrics['error_macro_f1']:.4f}"
            )

            # Per-class F1 summary
            f1_str = "  Error F1: " + "  ".join(
                f"c{c}={val_metrics[f'error_f1_c{c}']:.3f}(n={val_metrics[f'error_n_c{c}']})"
                for c in range(NUM_ERROR_TYPES)
            )
            logger.info(f1_str)

            torch.cuda.empty_cache()

            # Collect model weights once (14 GB bf16); defer optimizer.state_dict()
            # until the resume checkpoint so a killed write cannot corrupt the epoch file.
            model_state = raw_model.state_dict()

            # 1. Epoch checkpoint: model + val metrics only — small and fast.
            #    Written atomically (tmp → rename) so a kill mid-write leaves the
            #    previous epoch's file intact.
            epoch_ckpt = output_dir / f"epoch_{epoch+1:02d}.pt"
            _tmp_epoch = epoch_ckpt.with_suffix(".pt.tmp")
            torch.save({
                "epoch":    epoch,
                "model":    model_state,
                "val_loss": val_metrics["loss"],
                "args":     vars(args),
            }, _tmp_epoch)
            _tmp_epoch.rename(epoch_ckpt)
            logger.info(f"Checkpoint saved: {epoch_ckpt}")

            # 2. Best-model checkpoint (model only, atomic).
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                _tmp_best = best_ckpt.with_suffix(".pt.tmp")
                torch.save({
                    "epoch":    epoch,
                    "model":    model_state,
                    "val_loss": best_val_loss,
                    "args":     vars(args),
                }, _tmp_best)
                _tmp_best.rename(best_ckpt)
                logger.info(f"New best model (val_loss={best_val_loss:.4f}) → {best_ckpt}")

            # 3. Resume checkpoint: full optimizer + scheduler state (~42 GB total).
            #    Written last and atomically — if the OOM killer fires here, the epoch
            #    and best-model checkpoints above are already safely on disk.
            #    The previous epoch's resume.pt is preserved until this rename succeeds.
            resume_ckpt = output_dir / "resume.pt"
            _tmp_resume = output_dir / "resume.pt.tmp"
            torch.save({
                "epoch":         epoch,
                "model":         model_state,
                "optimizer":     optimizer.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "args":          vars(args),
            }, _tmp_resume)
            _tmp_resume.rename(resume_ckpt)
            logger.info(f"Resume checkpoint saved: {resume_ckpt}")

            all_metrics.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
            with open(metrics_path, "w") as f:
                json.dump(all_metrics, f, indent=2)

        # Synchronize all ranks before starting the next epoch
        if is_dist:
            dist.barrier()

    # ── Final model (rank 0 only) ─────────────────────────────────────────────
    if rank == 0:
        raw_model  = model.module if is_dist else model
        final_path = output_dir / "final_model.pt"
        torch.save(raw_model.state_dict(), final_path)
        logger.info(f"\nFinal model saved: {final_path}")
        logger.info("=" * 60)
        logger.info("Training complete.")
        logger.info(f"Best val loss: {best_val_loss:.4f}")
        logger.info(f"Output dir: {output_dir}")
        logger.info("=" * 60)

        last = all_metrics[-1]["val"]
        print(f"\nFinal val metrics (epoch {args.epochs}):")
        print(f"  loss     = {last['loss']:.4f}")
        print(f"  accuracy = {last['accuracy']:.4f}")
        print(f"  ece      = {last['ece']:.4f}")
        print(f"  macro_f1 = {last['error_macro_f1']:.4f}")

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
