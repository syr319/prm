"""
Step 7: Iterative distillation round 2.

Resumes from DistillPRM-7B-Instruct (best_model.pt) and trains for one epoch
on a combined dataset: original training data (sample_weight=1) + hard steps
mined by mine_hard_steps.py (sample_weight=3). Hard steps receive 3× the
gradient contribution via normalized per-sample loss weighting.

Usage (via run_iter2_train.sh):
    torchrun --nproc_per_node=8 distill/step7_iter2_train.py [args]

Single-GPU test:
    python3 distill/step7_iter2_train.py --epochs 1 --batch_size 2 --grad_accum 1
"""

import argparse
import datetime
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "distill"))

from step4_build_student_model import (
    DistillPRM,
    build_input_text,
    compute_difficulty,
    NUM_ERROR_TYPES,
)
from step5_train_distillpRM import (
    build_optimizer,
    collate_fn,
    compute_class_weights,
    evaluate,
    extract_error_type,
    _MODE_TEMPERATURE,
)


# ─── Weighted dataset ─────────────────────────────────────────────────────────

class WeightedDistillPRMDataset(Dataset):
    """DistillPRMDataset extended with per-sample training weight."""

    def __init__(
        self,
        records:        List[dict],
        tokenizer,
        max_length:     int   = 1024,
        default_weight: float = 1.0,
    ):
        self.records        = records
        self.tokenizer      = tokenizer
        self.max_length     = max_length
        self.default_weight = default_weight

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        r    = self.records[idx]
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
        hard_label = float(r["hard_label"])
        soft_score = float(r.get("soft_score", hard_label))
        if "error_type" in r:
            error_label = int(r["error_type"])
        else:
            error_label = extract_error_type(
                r.get("verification_cot", ""), int(r["hard_label"])
            )
        sample_weight = float(r.get("sample_weight", self.default_weight))
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "hard_label":     torch.tensor(hard_label,     dtype=torch.float32),
            "soft_score":     torch.tensor(soft_score,     dtype=torch.float32),
            "error_label":    torch.tensor(error_label,    dtype=torch.long),
            "sample_weight":  torch.tensor(sample_weight,  dtype=torch.float32),
        }


def weighted_collate_fn(batch, pad_token_id: int = 0):
    """Collate with sample_weight field included."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_list, attention_mask_list = [], []
    hard_labels, soft_scores, error_labels, sample_weights = [], [], [], []
    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids_list.append(
            F.pad(item["input_ids"],      (0, pad_len), value=pad_token_id)
        )
        attention_mask_list.append(
            F.pad(item["attention_mask"], (0, pad_len), value=0)
        )
        hard_labels.append(item["hard_label"])
        soft_scores.append(item["soft_score"])
        error_labels.append(item["error_label"])
        sample_weights.append(item["sample_weight"])
    return {
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "hard_label":     torch.stack(hard_labels),
        "soft_score":     torch.stack(soft_scores),
        "error_label":    torch.stack(error_labels),
        "sample_weight":  torch.stack(sample_weights),
    }


# ─── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch_iter2(
    model:                  torch.nn.Module,
    loader:                 DataLoader,
    optimizer:              torch.optim.Optimizer,
    scheduler,
    device:                 torch.device,
    lambda_error:           float,
    grad_accum:             int,
    grad_clip:              float,
    epoch:                  int,
    logger:                 logging.Logger,
    log_every:              int,
    class_weights:          Optional[torch.Tensor],
    focal_gamma:            float,
    rank:                   int,
    is_dist:                bool,
    difficulty_temperature: float,
) -> dict:
    model.train()
    total_loss = total_score_loss = total_err_loss = 0.0
    n_steps = 0
    t0      = time.time()
    optimizer.zero_grad()

    for step_idx, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        hard_label     = batch["hard_label"].to(device)
        soft_score     = batch["soft_score"].to(device)
        error_label    = batch["error_label"].to(device)
        sample_weight  = batch["sample_weight"].to(device)

        student_score, error_logits = model(input_ids, attention_mask)

        # ── Per-sample score loss (adaptive_t3) ───────────────────────────────
        difficulty  = compute_difficulty(soft_score, temperature=difficulty_temperature)
        alpha       = 0.9 * (1.0 - difficulty) + 0.1 * difficulty
        loss_ce     = F.binary_cross_entropy(student_score, hard_label.float(), reduction="none")
        student_d   = torch.stack([1.0 - student_score, student_score], dim=-1)
        teacher_d   = torch.stack([1.0 - soft_score,    soft_score],    dim=-1)
        loss_kl     = F.kl_div(
            torch.log(student_d + 1e-8), teacher_d, reduction="none"
        ).sum(dim=-1)
        l_score = alpha * loss_ce + (1.0 - alpha) * loss_kl    # [batch]

        # ── Per-sample focal error loss ────────────────────────────────────────
        ce_err  = F.cross_entropy(
            error_logits, error_label, weight=class_weights, reduction="none"
        )
        pt      = torch.exp(-ce_err)
        l_error = (1.0 - pt) ** focal_gamma * ce_err if focal_gamma > 0 else ce_err  # [batch]

        l_total = l_score + lambda_error * l_error    # [batch]

        # ── Normalize weights so gradient scale matches unweighted training ───
        w          = sample_weight / sample_weight.mean().clamp(min=1e-8)
        l_weighted = (l_total * w).mean()
        (l_weighted / grad_accum).backward()

        total_loss       += l_weighted.item()
        total_score_loss += l_score.mean().item()
        total_err_loss   += l_error.mean().item()
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
                f"  loss(w)={avg_loss:.4f}"
                f"  l_score={total_score_loss/n_steps:.4f}"
                f"  l_error={total_err_loss/n_steps:.4f}"
                f"  lr={lr_now:.2e}"
                f"  elapsed={elapsed:.0f}s"
            )

    # Flush remaining gradients in incomplete accumulation window
    if len(loader) % grad_accum != 0:
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    if is_dist and dist.is_initialized():
        t = torch.tensor(
            [total_loss, total_score_loss, total_err_loss, float(n_steps)],
            device=device,
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, total_score_loss, total_err_loss, n_steps = t.tolist()

    return {
        "loss":       total_loss       / n_steps,
        "loss_score": total_score_loss / n_steps,
        "loss_error": total_err_loss   / n_steps,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DistillPRM iterative distillation round 2.")
    parser.add_argument("--student_model",
                        default=str(ROOT / "models" / "Qwen2.5-Math-7B"))
    parser.add_argument("--resume_checkpoint",
                        default=str(ROOT / "outputs/distillprm-7b-instruct-adaptive-t3/adaptive_t3/best_model.pt"))
    parser.add_argument("--data_path",
                        default=str(ROOT / "data" / "genprm_math_steps_final.json"))
    parser.add_argument("--hard_data_path",
                        default=str(ROOT / "data" / "hard_steps_from_instruct.json"))
    parser.add_argument("--output_dir",
                        default=str(ROOT / "outputs" / "distillprm-7b-instruct-iter2"))
    parser.add_argument("--mode",           default="adaptive_t3",
                        choices=list(_MODE_TEMPERATURE.keys()),
                        help="Loss mode — determines difficulty temperature.")
    parser.add_argument("--epochs",         type=int,   default=1)
    parser.add_argument("--batch_size",     type=int,   default=4,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum",     type=int,   default=8)
    parser.add_argument("--backbone_lr",    type=float, default=2e-6)
    parser.add_argument("--head_lr",        type=float, default=2e-5)
    parser.add_argument("--weight_decay",   type=float, default=0.01)
    parser.add_argument("--warmup_ratio",   type=float, default=0.05)
    parser.add_argument("--max_length",     type=int,   default=1024)
    parser.add_argument("--lambda_error",   type=float, default=0.1)
    parser.add_argument("--focal_gamma",    type=float, default=2.0)
    parser.add_argument("--weight_scheme",  default="sqrt_inv_freq",
                        choices=["none", "inv_freq", "sqrt_inv_freq"])
    parser.add_argument("--val_frac",       type=float, default=0.05)
    parser.add_argument("--grad_clip",      type=float, default=1.0)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--log_every",      type=int,   default=100)
    args = parser.parse_args()

    diff_temp = _MODE_TEMPERATURE.get(args.mode, 3.0)

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    handlers: list = [logging.FileHandler(output_dir / "train.log")] if rank == 0 else []
    if rank == 0:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level    = logging.INFO if rank == 0 else logging.WARNING,
        format   = "%(asctime)s [%(levelname)s] %(message)s",
        handlers = handlers,
    )
    logger = logging.getLogger(__name__)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if rank == 0:
        logger.info(f"Device: {device}  world_size={world_size}")
        logger.info(f"Mode: {args.mode}  difficulty_temperature={diff_temp}")
        logger.info(
            f"Effective batch: {args.batch_size}"
            f" × {args.grad_accum}"
            f" × {world_size}"
            f" = {args.batch_size * args.grad_accum * world_size}"
        )
        logger.info(f"backbone_lr={args.backbone_lr}  head_lr={args.head_lr}")

    # ── Load data ─────────────────────────────────────────────────────────────
    orig_path = Path(args.data_path)
    hard_path = Path(args.hard_data_path)

    if rank == 0:
        logger.info(f"Loading original data: {orig_path}")
    with open(orig_path, encoding="utf-8") as f:
        orig_records = json.load(f)
    orig_records = [r for r in orig_records if r.get("hard_label") in (0, 1)]

    if rank == 0:
        logger.info(f"Loading hard steps: {hard_path}")
    with open(hard_path, encoding="utf-8") as f:
        hard_records = json.load(f)

    if rank == 0:
        logger.info(
            f"Original: {len(orig_records):,}  "
            f"Hard: {len(hard_records):,}  "
            f"Combined: {len(orig_records)+len(hard_records):,}"
        )

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # ── Dataset: val from original only; hard records all go to train ─────────
    n_val      = max(1, int(len(orig_records) * args.val_frac))
    n_train    = len(orig_records) - n_val
    g          = torch.Generator().manual_seed(args.seed)
    orig_ds    = WeightedDistillPRMDataset(orig_records, tokenizer, args.max_length, default_weight=1.0)
    orig_train_ds, val_ds = random_split(orig_ds, [n_train, n_val], generator=g)

    hard_ds  = WeightedDistillPRMDataset(hard_records, tokenizer, args.max_length, default_weight=3.0)
    train_ds = ConcatDataset([orig_train_ds, hard_ds])

    if rank == 0:
        logger.info(
            f"Train: {len(train_ds):,}"
            f"  (orig_train={n_train:,}  hard={len(hard_records):,})"
            f"  Val: {n_val:,}"
        )

    _collate_w = lambda b: weighted_collate_fn(b, pad_token_id=pad_id)
    _collate   = lambda b: collate_fn(b, pad_token_id=pad_id)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_dist else None
    train_loader  = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = (train_sampler is None),
        sampler     = train_sampler,
        num_workers = args.num_workers,
        collate_fn  = _collate_w,
        pin_memory  = True,
    )

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

    # ── Class weights (from original train split only — no double-counting) ───
    train_orig_records = [orig_records[i] for i in orig_train_ds.indices]  # type: ignore
    class_weights = compute_class_weights(
        train_orig_records, NUM_ERROR_TYPES, args.weight_scheme, device
    )
    if rank == 0 and class_weights is not None:
        logger.info(f"Class weights ({args.weight_scheme}): {[f'{w:.3f}' for w in class_weights.tolist()]}")

    # ── Model: load backbone then resume from checkpoint ──────────────────────
    if rank == 0:
        logger.info(f"Loading backbone from {args.student_model} ...")
    model = DistillPRM(model_name_or_path=args.student_model)

    resume_path = Path(args.resume_checkpoint)
    if rank == 0:
        logger.info(f"Resuming from {resume_path} ...")
    ckpt  = torch.load(str(resume_path), map_location="cpu", weights_only=True)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        val_loss_info = f"  (checkpoint val_loss={ckpt.get('val_loss', '?')})" if "val_loss" in ckpt else ""
        logger.info(f"Model loaded ({n_params/1e6:.1f}M params){val_loss_info}")

    if args.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable()
        if rank == 0:
            logger.info("Gradient checkpointing enabled.")

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = build_optimizer(
        model,
        backbone_lr     = args.backbone_lr,
        head_lr         = args.head_lr,
        weight_decay    = args.weight_decay,
        freeze_backbone = False,
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

    # ── DDP wrap ──────────────────────────────────────────────────────────────
    if is_dist:
        model = DDP(model, device_ids=[local_rank])

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_ckpt     = output_dir / "best_model.pt"
    metrics_path  = output_dir / "metrics.json"
    all_metrics: List[dict] = []

    if rank == 0:
        logger.info("=" * 60)
        logger.info("Iterative distillation round 2 — starting")
        logger.info("=" * 60)

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{args.epochs}")
            logger.info(f"{'='*60}")

        train_metrics = train_one_epoch_iter2(
            model                  = model,
            loader                 = train_loader,
            optimizer              = optimizer,
            scheduler              = scheduler,
            device                 = device,
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
                f"  Train  loss(w)={train_metrics['loss']:.4f}"
                f"  l_score={train_metrics['loss_score']:.4f}"
                f"  l_error={train_metrics['loss_error']:.4f}\n"
                f"  Val    loss={val_metrics['loss']:.4f}"
                f"  acc={val_metrics['accuracy']:.4f}"
                f"  ece={val_metrics['ece']:.4f}"
                f"  macro_f1={val_metrics['error_macro_f1']:.4f}"
            )

            torch.cuda.empty_cache()
            model_state = raw_model.state_dict()

            # Epoch checkpoint (atomic)
            epoch_ckpt = output_dir / f"epoch_{epoch+1:02d}.pt"
            _tmp = epoch_ckpt.with_suffix(".pt.tmp")
            torch.save({
                "epoch":    epoch,
                "model":    model_state,
                "val_loss": val_metrics["loss"],
                "args":     vars(args),
            }, _tmp)
            _tmp.rename(epoch_ckpt)
            logger.info(f"Checkpoint: {epoch_ckpt}")

            # Best-model checkpoint (atomic)
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                _tmp = best_ckpt.with_suffix(".pt.tmp")
                torch.save({
                    "epoch":    epoch,
                    "model":    model_state,
                    "val_loss": best_val_loss,
                    "args":     vars(args),
                }, _tmp)
                _tmp.rename(best_ckpt)
                logger.info(f"New best (val_loss={best_val_loss:.4f}) → {best_ckpt}")

            all_metrics.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
            with open(metrics_path, "w") as f:
                json.dump(all_metrics, f, indent=2)

        if is_dist:
            dist.barrier()

    # ── Final model ───────────────────────────────────────────────────────────
    if rank == 0:
        raw_model  = model.module if is_dist else model
        final_path = output_dir / "final_model.pt"
        torch.save(raw_model.state_dict(), final_path)
        logger.info(f"\nFinal model → {final_path}")
        logger.info("=" * 60)
        logger.info("Iterative distillation round 2 complete.")
        logger.info(f"Best val loss: {best_val_loss:.4f}")
        logger.info(f"Output dir: {output_dir}")
        logger.info("=" * 60)

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
