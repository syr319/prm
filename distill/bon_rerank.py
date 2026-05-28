"""
Best-of-N Reranking Evaluation — MATH500

Evaluates four reranking strategies on MATH500 across N ∈ {1,4,8,16,32}:
  random     — select one candidate randomly (fixed seed)
  majority   — most-frequent extracted answer
  PRM        — highest aggregated-step-score candidate (tagged by --tag)

Step aggregation (--agg):
  min   — minimum step score  (default for DistillPRM)
  avg   — mean of all step scores  (recommended for Skywork)
  last  — score of the final step only

Model types (--model_type):
  distillprm — DistillPRM checkpoint (.pt file), requires --student_model backbone
  skywork    — Skywork-o1-Open-PRM, loaded from a full model directory via trust_remote_code

Run once per PRM model; results are merged into bon_results.json.

Usage:
    # Random + Majority + DistillPRM-1.5B
    CUDA_VISIBLE_DEVICES=0 python3 distill/bon_rerank.py \\
        --prm_checkpoint models/DistillPRM-1.5B/adaptive_t3/best_model.pt \\
        --student_model  models/Qwen2.5-Math-1.5B \\
        --tag            DistillPRM-1.5B

    # DistillPRM-7B (adds to existing results)
    CUDA_VISIBLE_DEVICES=0 python3 distill/bon_rerank.py \\
        --prm_checkpoint models/DistillPRM-7B/adaptive_t3/best_model.pt \\
        --student_model  models/Qwen2.5-Math-7B \\
        --tag            DistillPRM-7B

    # Skywork-PRM-1.5B (avg aggregation recommended)
    CUDA_VISIBLE_DEVICES=0 python3 distill/bon_rerank.py \\
        --prm_checkpoint models/Skywork-PRM-1.5B \\
        --model_type     skywork \\
        --agg            avg \\
        --tag            Skywork-PRM-1.5B
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "distill"))

from step4_build_student_model import DistillPRM, build_input_text

N_VALUES = [1, 4, 8, 16, 32]


# ─── Answer extraction ────────────────────────────────────────────────────────

def extract_boxed(text: str) -> str:
    """Extract last \\boxed{...} content from text, handling nested braces."""
    pattern = r"\\boxed\{"
    positions = [m.start() for m in re.finditer(pattern, text)]
    if not positions:
        return ""
    start = positions[-1] + len(r"\boxed{") - 1   # position of opening brace
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    return ""


def normalize_answer(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\,", "").replace(r"\!", "")
    return s.lower()


def answers_equal(pred: str, gold: str) -> bool:
    """Check if two math answers are equivalent."""
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if not p:
        return False
    if p == g:
        return True
    # Numeric comparison
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        pass
    # SymPy equivalence (best-effort)
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        diff = simplify(parse_latex(p) - parse_latex(g))
        return diff == 0
    except Exception:
        return False


# ─── Step parsing ─────────────────────────────────────────────────────────────

def parse_steps(text: str) -> List[str]:
    """Split a solution into individual reasoning steps."""
    text = text.strip()
    if not text:
        return ["(empty)"]

    # Try double-newline paragraph split first
    paras = [p.strip() for p in re.split(r"\n\n+", text) if len(p.strip()) > 10]
    if len(paras) >= 2:
        return paras[:15]   # cap at 15 steps

    # Fall back to single-newline split
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 10]
    if len(lines) >= 2:
        return lines[:15]

    return [text]


# ─── Reranking strategies ─────────────────────────────────────────────────────

def select_random(candidates: List[str], n: int, rng: np.random.Generator) -> str:
    return rng.choice(candidates[:n])


# def select_majority(candidates: List[str], n: int) -> str:
#     """Select answer by majority vote on extracted \\boxed{} contents."""
#     answers = [extract_boxed(c) for c in candidates[:n]]
#     normalized = [normalize_answer(a) for a in answers]

#     # Group by normalized form, keep original for output
#     groups: Dict[str, Tuple[int, str]] = {}
#     for orig, norm in zip(answers, normalized):
#         if norm:
#             count, _ = groups.get(norm, (0, orig))
#             groups[norm] = (count + 1, orig)

#     if not groups:
#         # No boxed answers found — return last line of first candidate
#         first = candidates[0].strip()
#         return first.split("\n")[-1] if first else ""

#     best_norm = max(groups, key=lambda k: groups[k][0])
#     return groups[best_norm][1]
def select_majority(candidates: List[str], n: int) -> str:
    """Select answer by majority vote on extracted \\boxed{} contents."""
    answers = [extract_boxed(c) for c in candidates[:n]]
    normalized = [normalize_answer(a) for a in answers]

    # Group by normalized form, keep original candidate text for output
    groups: Dict[str, Tuple[int, str]] = {}
    for cand, norm in zip(candidates[:n], normalized):
        if norm:
            count, _ = groups.get(norm, (0, cand))
            groups[norm] = (count + 1, cand)

    if not groups:
        return candidates[0] if candidates else ""

    best_norm = max(groups, key=lambda k: groups[k][0])
    return groups[best_norm][1]

# ─── PRM scoring ──────────────────────────────────────────────────────────────

def load_prm(checkpoint: str, student_model: str, device: torch.device) -> DistillPRM:
    model = DistillPRM(model_name_or_path=student_model)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Loaded PRM ({n/1e6:.1f}M params) from {checkpoint}")
    return model


@torch.no_grad()
def score_all_candidates(
    model:      DistillPRM,
    tokenizer,
    problem:    str,
    candidates: List[str],
    device:     torch.device,
    max_length: int = 1024,
    batch_size: int = 32,
) -> List[List[float]]:
    """Return per-step scores for each candidate (raw, not aggregated)."""
    all_texts:   List[str] = []
    step_counts: List[int] = []

    for cand in candidates:
        steps = parse_steps(cand)
        step_counts.append(len(steps))
        for i, step in enumerate(steps):
            ctx = "\n\n".join(steps[:i])
            all_texts.append(build_input_text(problem, ctx, step))

    # Score in batches
    flat_scores: List[float] = []
    for i in range(0, len(all_texts), batch_size):
        chunk = all_texts[i : i + batch_size]
        enc = tokenizer(
            chunk,
            max_length     = max_length,
            truncation     = True,
            padding        = True,
            return_tensors = "pt",
        )
        scores, _ = model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        )
        flat_scores.extend(scores.cpu().float().tolist())

    # Group step scores per candidate
    per_cand: List[List[float]] = []
    offset = 0
    for n_steps in step_counts:
        per_cand.append(flat_scores[offset : offset + n_steps] or [0.0])
        offset += n_steps

    return per_cand


def aggregate_step_scores(step_scores: List[float], agg: str) -> float:
    """Aggregate per-step scores into a single candidate score."""
    if not step_scores:
        return 0.0
    if agg == "avg":
        return float(np.mean(step_scores))
    elif agg == "last":
        return step_scores[-1]
    else:  # min
        return float(min(step_scores))


def select_prm(per_step_scores: List[List[float]], n: int, agg: str) -> int:
    """Return index of best candidate among first n using the given aggregation."""
    agg_scores = [aggregate_step_scores(ss, agg) for ss in per_step_scores[:n]]
    return int(np.argmax(agg_scores))


# ─── Skywork-PRM loading and scoring ──────────────────────────────────────────

def load_skywork_prm(checkpoint_dir: str, device: torch.device):
    """
    Load Skywork-o1-Open-PRM from a local directory.

    The model uses AutoModel (auto_map → Qwen2ForRewardModel) with a ValueHead
    that produces per-token scalar scores. Returns (model, tokenizer).
    """
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        checkpoint_dir,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    n = sum(p.numel() for p in model.parameters())
    print(f"Loaded Skywork PRM ({n/1e6:.1f}M params) from {checkpoint_dir}")
    return model, tokenizer

def _skywork_build_input(
    problem: str,
    response: str,
    tokenizer,
    max_length: int = 2048,
) -> Tuple[torch.Tensor, int]:
    """
    Tokenize problem + response for Skywork-PRM scoring.

    Steps are split by double-newline or single-newline (same heuristic as
    parse_steps), then each step is suffixed with ' ки' so the model can
    score at those positions.

    Returns (input_ids_1d, num_steps).
    """
    steps = parse_steps(response)

    # Build tagged text: problem\n\nstep1\nки\nstep2\nки\n...
    # "ки" must NOT have a leading space: encode(" ки") splits into two tokens
    # [7665, 1802], while encode("ки") gives the single token [16748].
    tagged = "".join(step.strip() + "\nки\n" for step in steps).rstrip("\n")
    full_text = problem.strip() + "\n\n" + tagged

    ids = tokenizer.encode(
        full_text,
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
    )
    return torch.tensor(ids, dtype=torch.long), len(steps)


@torch.no_grad()
def score_all_candidates_skywork(
    model,
    tokenizer,
    problem:    str,
    candidates: List[str],
    device:     torch.device,
    max_length: int = 2048,
    batch_size: int = 8,
) -> List[List[float]]:
    """
    Return per-step scores for each candidate using Skywork-PRM.

    Qwen2ForRewardModel uses a ValueHead: v_head(hidden_states).squeeze(-1)
    which outputs a scalar per token.  We collect the values at every ' ки'
    step-marker token position and apply sigmoid to map to [0, 1].

    The model's forward() returns pooled_logits (last-token only), so we call
    model.model() directly to get full hidden states, then apply model.v_head().
    """
    # encode("ки") → [16748] reliably; convert_tokens_to_ids("ки") returns None
    # for this tokenizer because of how the vocab key is stored.
    step_tag_id: int = tokenizer.encode("ки", add_special_tokens=False)[-1]

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    per_cand: List[List[float]] = []

    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start : start + batch_size]

        all_ids:    List[torch.Tensor] = []
        all_nsteps: List[int]          = []
        for cand in chunk:
            ids, n_steps = _skywork_build_input(problem, cand, tokenizer, max_length)
            all_ids.append(ids)
            all_nsteps.append(n_steps)

        # Left-pad so the real sequence ends at the same position
        max_len = max(t.size(0) for t in all_ids)
        padded  = torch.full((len(all_ids), max_len), pad_id, dtype=torch.long)
        attn    = torch.zeros((len(all_ids), max_len), dtype=torch.long)
        for i, ids in enumerate(all_ids):
            offset = max_len - ids.size(0)
            padded[i, offset:] = ids
            attn[i,   offset:] = 1

        # Get full per-token hidden states via inner transformer
        # use_cache=False avoids DynamicCache.get_usable_length() which was
        # renamed to get_seq_length() in newer transformers versions.
        transformer_out = model.model(
            input_ids      = padded.to(device),
            attention_mask = attn.to(device),
            use_cache      = False,
        )
        hidden_states = transformer_out[0]                     # [batch, seq_len, hidden]
        token_scores  = model.v_head(hidden_states).squeeze(-1)  # [batch, seq_len]
        token_scores  = torch.sigmoid(token_scores).cpu().float()

        for i, ids_orig in enumerate(all_ids):
            padded_ids_i = padded[i].cpu()
            flags = (padded_ids_i == step_tag_id)
            step_positions = flags.nonzero(as_tuple=True)[0].tolist()

            step_scores = [token_scores[i, pos].item() for pos in step_positions]

            if not step_scores:
                step_scores = [0.5]  # no markers found — assign neutral score

            per_cand.append(step_scores)

    return per_cand


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_method(
    data:           List[dict],
    method_name:    str,
    n_values:       List[int],
    get_prediction, # (candidates, n) → str (predicted answer)
) -> Tuple[Dict[str, float], float]:
    """
    Runs the method for each N and returns (accuracy_dict, avg_time_ms_per_problem).
    avg_time_ms is averaged over problems and over N values.
    """
    accuracies: Dict[str, float] = {}
    total_time = 0.0
    n_timing   = 0

    for n in n_values:
        correct = 0
        t0 = time.perf_counter()
        for item in data:
            candidates = item["candidates"]
            gold       = item["answer"]
            pred       = get_prediction(candidates, n)
            pred_ans   = extract_boxed(pred) if pred else ""
            if answers_equal(pred_ans, gold):
                correct += 1
        elapsed = time.perf_counter() - t0
        accuracies[str(n)] = correct / len(data)
        total_time += elapsed * 1000 / len(data)   # ms per problem at this N
        n_timing   += 1

        print(f"  {method_name:20s}  N={n:>2}  acc={accuracies[str(n)]:.4f}")

    avg_time_ms = total_time / n_timing
    return accuracies, avg_time_ms


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BoN reranking evaluation on MATH500.")
    parser.add_argument("--candidates_path",
                        default=str(ROOT / "data" / "math500_candidates.json"))
    parser.add_argument("--output",
                        default=str(ROOT / "distill" / "eval_results" / "bon_results.json"))
    parser.add_argument("--prm_checkpoint",
                        default=None,
                        help="Path to PRM checkpoint (.pt file for distillprm, directory for skywork)")
    parser.add_argument("--student_model",
                        default=None,
                        help="Backbone model path (required for distillprm, unused for skywork)")
    parser.add_argument("--model_type",    default="distillprm",
                        choices=["distillprm", "skywork"],
                        help="PRM architecture: distillprm (default) or skywork")
    parser.add_argument("--tag",
                        default="DistillPRM",
                        help="Label for this PRM in results (e.g. DistillPRM-1.5B)")
    parser.add_argument("--agg",           default="min",
                        choices=["min", "avg", "last"],
                        help="Step score aggregation: min (default/distillprm) or avg (recommended for skywork)")
    parser.add_argument("--batch_size",   type=int, default=32,
                        help="PRM scoring batch size")
    parser.add_argument("--max_length",   type=int, default=1024)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load candidates
    print(f"Loading candidates from {args.candidates_path} ...")
    with open(args.candidates_path, encoding="utf-8") as f:
        data = json.load(f)
    n_candidates = len(data[0]["candidates"])
    print(f"  {len(data)} problems  ×  {n_candidates} candidates each")

    # Validate N values
    n_values = [n for n in N_VALUES if n <= n_candidates]

    # Load existing results for merging
    all_results: dict = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            prev = json.load(f)
        all_results = prev.get("results", {})
        print(f"Loaded existing results: {list(all_results.keys())}")

    rng = np.random.default_rng(args.seed)

    # ── Random baseline ────────────────────────────────────────────────────────
    if "random" not in all_results:
        print("\n[Random baseline]")
        _rng_per_n = {n: np.random.default_rng(args.seed) for n in n_values}
        acc, avg_t = evaluate_method(
            data, "random", n_values,
            lambda cands, n: select_random(cands, n, _rng_per_n[n])
        )
        all_results["random"] = {"accuracy": acc, "avg_time_ms": round(avg_t, 3)}
    else:
        print("[random] already in results — skipping")

    # ── Majority voting ────────────────────────────────────────────────────────
    if "majority" not in all_results:
        print("\n[Majority voting]")
        acc, avg_t = evaluate_method(
            data, "majority", n_values,
            lambda cands, n: select_majority(cands, n)
        )
        all_results["majority"] = {"accuracy": acc, "avg_time_ms": round(avg_t, 3)}
    else:
        print("[majority] already in results — skipping")

    # ── PRM reranking ──────────────────────────────────────────────────────────
    if args.prm_checkpoint and args.tag not in all_results:
        prm_ckpt = Path(args.prm_checkpoint)
        if not prm_ckpt.is_absolute():
            prm_ckpt = ROOT / args.prm_checkpoint
        # if not prm_ckpt.exists():
        #     print(f"\n[{args.tag}] Checkpoint not found: {prm_ckpt} — skipping PRM reranking.")
        # else:
            print(f"\n[{args.tag}] Loading PRM model (type={args.model_type}) ...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"  Device: {device}")

            if args.model_type == "skywork":
                model, tokenizer = load_skywork_prm(str(prm_ckpt), device)
                # Skywork scores per-candidate (whole response at once)
                _score_fn = lambda problem, candidates: score_all_candidates_skywork(
                    model, tokenizer, problem, candidates,
                    device, args.max_length, args.batch_size,
                )
            else:
                # distillprm — needs separate backbone tokenizer
                if not args.student_model:
                    print(f"\n[{args.tag}] --student_model is required for model_type=distillprm. Skipping.")
                    args.prm_checkpoint = None  # skip save below
                else:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(
                        args.student_model, trust_remote_code=True)
                    tokenizer.padding_side = "right"
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    model = load_prm(str(prm_ckpt), args.student_model, device)
                    _score_fn = lambda problem, candidates: score_all_candidates(
                        model, tokenizer, problem, candidates,
                        device, args.max_length, args.batch_size,
                    )

        if prm_ckpt.exists() and args.prm_checkpoint:
            # Pre-compute per-step scores for all 500 × N_max candidates
            print(f"  Scoring all {len(data)} × {n_candidates} candidates  (agg={args.agg}) ...")
            all_per_step: List[List[List[float]]] = []
            t_score_start = time.perf_counter()

            for i, item in enumerate(data):
                per_step = _score_fn(item["problem"], item["candidates"])
                all_per_step.append(per_step)
                if (i + 1) % 50 == 0:
                    elapsed = time.perf_counter() - t_score_start
                    eta = elapsed / (i + 1) * (len(data) - i - 1)
                    print(f"  [{i+1}/{len(data)}]  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

            total_scoring_time_ms = (time.perf_counter() - t_score_start) * 1000
            avg_scoring_ms = total_scoring_time_ms / len(data)
            print(f"  Scoring done: {total_scoring_time_ms/1000:.1f}s  "
                  f"avg={avg_scoring_ms:.1f} ms/problem (N={n_candidates})")

            # Evaluate for each N
            print(f"\n[{args.tag}] Evaluating accuracy (agg={args.agg}) ...")
            acc: Dict[str, float] = {}
            for n in n_values:
                correct = 0
                for item, per_step in zip(data, all_per_step):
                    best_idx = select_prm(per_step, n, args.agg)
                    pred_ans = extract_boxed(item["candidates"][best_idx])
                    if answers_equal(pred_ans, item["answer"]):
                        correct += 1
                acc[str(n)] = correct / len(data)
                print(f"  {args.tag:20s}  N={n:>2}  acc={acc[str(n)]:.4f}")

            # Timing: avg ms per problem across N_values (scales with N)
            timing_per_n = {
                str(n): round(avg_scoring_ms * n / n_candidates, 1)
                for n in n_values
            }

            all_results[args.tag] = {
                "accuracy":        acc,
                "model_type":      args.model_type,
                "agg":             args.agg,
                "avg_time_ms":     round(avg_scoring_ms, 1),
                "timing_per_n_ms": timing_per_n,
                "checkpoint":      str(prm_ckpt),
            }

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

    elif args.tag in all_results:
        print(f"\n[{args.tag}] already in results — skipping PRM reranking.")

    # ── Save results ───────────────────────────────────────────────────────────
    output = {
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "n_problems":   len(data),
        "n_candidates": n_candidates,
        "N_values":     n_values,
        "results":      all_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved → {output_path}")

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Method':<22}", end="")
    for n in n_values:
        print(f"{'N='+str(n):>9}", end="")
    print(f"{'ms/prob':>10}")
    print("-" * 72)
    for method, res in all_results.items():
        acc = res["accuracy"]
        t   = res.get("avg_time_ms", 0.0)
        print(f"{method:<22}", end="")
        for n in n_values:
            v = acc.get(str(n), float("nan"))
            print(f"{v*100:>8.1f}%", end="")
        print(f"{t:>10.1f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
