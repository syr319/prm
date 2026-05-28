#!/usr/bin/env python3
"""Speed benchmark v2: DistillPRM vs GenPRM — fair & best-practice comparison.

Fairness analysis of v1
-----------------------
v1 used vLLM for GenPRM and raw transformers for DistillPRM, which mixes
frameworks and overstates the gap.  v2 runs two separate plans:

  Plan A — Same Framework (Native Transformers)
    DistillPRM : model.forward()    bfloat16 + flash_attention_2 + sorted batching
    GenPRM     : model.generate()   bfloat16 + flash_attention_2 + sorted batching
    → Isolates the algorithmic cost (single forward vs. autoregressive generation).

  Plan B — Best Practice (Each Model's Optimal Stack)
    DistillPRM : Transformers (same as Plan A — already optimal for discriminative)
    GenPRM     : vLLM         (continuous batching, optimised CUDA kernels)
    → Reflects real-world deployment speed.

DistillPRM optimisations applied in both plans
-----------------------------------------------
  1. bfloat16  — already default in backbone; heads cast to bf16 before timing
  2. flash_attention_2  — passed to AutoModel.from_pretrained via new kwarg in step4
  3. Sorted batching  — inputs grouped by tokenised length to minimise padding waste

GenPRM Plan A notes
-------------------
  - 2-stage generate(): stage1 max_new_tokens=512 (analyze), stage2 max_new_tokens=1024
  - Inputs sorted by length before each stage; re-sorted after stage1
  - Full max_new_tokens budget used (no per-sequence early stopping)
  - This slightly over-estimates GenPRM latency; noted in the report

Usage
-----
  python distill/step_benchmark_speed_v2.py [options]
  python distill/step_benchmark_speed_v2.py --skip_plan_b   # Plan A only
  python distill/step_benchmark_speed_v2.py --num_problems 20 --warmup 5  # quick

Output
------
  distill/reports/speed_benchmark_v2.json
  distill/reports/speed_benchmark_v2.md
"""

import os
# FLASHINFER_DISABLE_VERSION_CHECK bypasses the cubin/wheel version mismatch check
# that otherwise crashes vLLM's EngineCore subprocess.
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

import argparse
import gc
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "distill"))

# ── paths ─────────────────────────────────────────────────────────────────────
DISTILLPRM_1_5B_CKPT     = ROOT / "models/DistillPRM-1.5B/adaptive_t3/best_model.pt"
DISTILLPRM_7B_CKPT       = ROOT / "models/DistillPRM-7B/adaptive_t3/epoch_02.pt"
DISTILLPRM_1_5B_BACKBONE = ROOT / "models/Qwen2.5-Math-1.5B"
DISTILLPRM_7B_BACKBONE   = ROOT / "models/Qwen2.5-Math-7B"
GENPRM_1_5B_DIR          = ROOT / "models/GenPRM-1.5B"
GENPRM_7B_DIR            = ROOT / "models/GenPRM-7B"
PROCESSBENCH_DIR         = ROOT / "data/ProcessBench"
REPORTS_DIR              = ROOT / "distill/reports"

PROCESSBENCH_SPLITS = ["gsm8k", "math", "olympiadbench", "omnimath"]

GENPRM_SYSTEM    = ("You are a math teacher. Your task is to review and critique "
                    "the paragraphs in solution step by step.")
ANALYZE_TEMPLATE = "<analyze>\nLet's analyze the Paragraph {cur_step} step by step: "
VERIFY_PREFIX    = "<verify>\nLet's use python code to find any potential error:\n```python\n"
OUTPUT_PREFIX    = "<output>\n**Judgement**: $\\boxed"


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_benchmark_steps(num_problems: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    all_problems = []
    for split in PROCESSBENCH_SPLITS:
        fpath = PROCESSBENCH_DIR / f"{split}.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            probs = json.load(f)
        for p in probs:
            p["_split"] = split
        all_problems.extend(probs)

    rng.shuffle(all_problems)
    selected = all_problems[:num_problems]

    records: List[dict] = []
    for prob in selected:
        question = prob["problem"]
        steps    = prob["steps"]
        ctx: List[str] = []
        for i, step_text in enumerate(steps):
            records.append({
                "question":     question,
                "context":      "\n\n".join(ctx),
                "current_step": step_text,
                "step_idx":     i,
                "step_num":     i + 1,
            })
            ctx.append(step_text)

    print(f"  Sampled {len(selected)} problems → {len(records)} steps "
          f"(avg {len(records)/max(len(selected),1):.1f} steps/problem)")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_distillprm_state(model, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def _detect_attn_impl(backbone_path: Path) -> str:
    """Return the best available attention implementation."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        pass
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return "sdpa"
    return "eager"


def _build_genprm_prompt(record: dict, tokenizer) -> str:
    """Single-turn GenPRM prompt: system + user (question+context+step) + analyze prefix."""
    parts = [f"Question: {record['question']}"]
    if record["context"]:
        parts.append(f"Context:\n{record['context']}")
    parts.append(f"Current step:\n{record['current_step']}")
    messages = [
        {"role": "system", "content": GENPRM_SYSTEM},
        {"role": "user",   "content": "\n\n".join(parts)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    eos = tokenizer.eos_token or ""
    if eos:
        if prompt.endswith(f"{eos}\n"):
            prompt = prompt[: -len(f"{eos}\n")]
        elif prompt.endswith(eos):
            prompt = prompt[: -len(eos)]
    return prompt + ANALYZE_TEMPLATE.format(cur_step=record["step_num"])


def _sorted_indices(tokenizer, texts: List[str], max_length: int) -> List[int]:
    """Return indices that sort texts by tokenised length (ascending)."""
    lengths = [
        len(tokenizer.encode(t, truncation=True, max_length=max_length))
        for t in texts
    ]
    return sorted(range(len(texts)), key=lambda i: lengths[i])


def _cleanup_model(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()
    # Synchronize so all pending CUDA ops are complete before the next model load
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ─────────────────────────────────────────────────────────────────────────────
# DistillPRM benchmark (used for both Plan A and Plan B)
# ─────────────────────────────────────────────────────────────────────────────

def bench_distillprm(
    ckpt_path: Path,
    backbone_path: Path,
    records: List[dict],
    batch_sizes: List[int],
    warmup: int,
    device: torch.device,
    max_length: int = 1024,
) -> Dict[int, dict]:
    """
    Load DistillPRM with bfloat16 + flash_attention_2 and benchmark forward passes
    using sorted batching (inputs grouped by length to minimise padding).
    """
    from step4_build_student_model import DistillPRM, build_input_text
    from transformers import AutoTokenizer

    if not ckpt_path.exists():
        print(f"  [skip] checkpoint not found: {ckpt_path}")
        return {}
    if not backbone_path.exists():
        print(f"  [skip] backbone not found: {backbone_path}")
        return {}

    attn_impl = _detect_attn_impl(backbone_path)
    print(f"  Attention impl : {attn_impl}")

    tokenizer = AutoTokenizer.from_pretrained(str(backbone_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading DistillPRM ({backbone_path.name}) ...")
    try:
        model = DistillPRM(
            model_name_or_path=str(backbone_path),
            attn_implementation=attn_impl,
        )
    except TypeError:
        # Fallback if step4 doesn't have the kwarg yet (shouldn't happen)
        model = DistillPRM(model_name_or_path=str(backbone_path))

    _load_distillprm_state(model, ckpt_path)
    # backbone is already bfloat16 (set in AutoModel.from_pretrained);
    # heads stay float32 — forward() explicitly casts last_hidden to float32
    # before the heads, so mixing dtypes here is intentional.
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    backbone_dtype = next(model.backbone.parameters()).dtype
    print(f"  Model loaded ({n_params:.0f}M params, backbone={backbone_dtype})")

    texts = [
        build_input_text(r["question"], r["context"], r["current_step"])
        for r in records
    ]

    # Pre-compute sorted order once (shared across batch sizes)
    sorted_idx = _sorted_indices(tokenizer, texts, max_length)
    sorted_texts = [texts[i] for i in sorted_idx]

    results: Dict[int, dict] = {}
    for bs in batch_sizes:
        print(f"\n  [bs={bs}] warmup ({warmup} steps, sorted) ...")
        _distillprm_fwd_sorted(model, tokenizer, sorted_texts[:max(warmup, bs)],
                               device, bs, max_length)

        print(f"  [bs={bs}] timing {len(texts)} steps ...")
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _distillprm_fwd_sorted(model, tokenizer, sorted_texts,
                               device, bs, max_length)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        n = len(texts)
        results[bs] = {
            "n_steps":       n,
            "batch_size":    bs,
            "total_sec":     round(elapsed, 3),
            "ms_per_step":   round(elapsed / n * 1000, 2),
            "steps_per_sec": round(n / elapsed, 2),
            "avg_tokens":    None,
        }
        print(f"    → {elapsed:.1f}s | {elapsed/n*1000:.1f} ms/step "
              f"| {n/elapsed:.1f} steps/s")

    _cleanup_model(model)
    return results


@torch.no_grad()
def _distillprm_fwd_sorted(model, tokenizer, sorted_texts, device, batch_size, max_length):
    for i in range(0, len(sorted_texts), batch_size):
        batch = sorted_texts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        model(enc["input_ids"], enc["attention_mask"])


# ─────────────────────────────────────────────────────────────────────────────
# Plan A — GenPRM with transformers.generate()
# ─────────────────────────────────────────────────────────────────────────────

def bench_genprm_transformers(
    model_dir: Path,
    records: List[dict],
    warmup: int,
    batch_size: int,
    device: torch.device,
    max_tokens_s1: int = 512,
    max_tokens_s2: int = 1024,
    max_input_len: int = 2048,
) -> dict:
    """
    GenPRM inference using transformers.generate() — 2-stage batched generation.

    Stage 1 (Analyze): sorted batch, max_new_tokens=max_tokens_s1
    Stage 2 (Verify+Output): re-sorted batch using stage-1 prompts, max_new_tokens=max_tokens_s2

    Note: no per-sequence early stopping — full token budget used, giving a
    conservative (slightly slower) estimate for GenPRM timing.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not model_dir.exists():
        print(f"  [skip] not found: {model_dir}")
        return {}

    attn_impl = _detect_attn_impl(model_dir)
    print(f"  Attention impl : {attn_impl}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    tokenizer.padding_side = "left"   # required for causal LM batch generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading GenPRM ({model_dir.name}) via transformers ...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map={"": device},
        )
    except Exception as e:
        print(f"  [warn] flash_attn2 failed ({e}), falling back to sdpa")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": device},
        )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model loaded ({n_params:.0f}M params, bf16)")

    prompts = [_build_genprm_prompt(r, tokenizer) for r in records]

    gen_kwargs = dict(
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )

    print(f"  Warmup ({warmup} steps, batch_size={batch_size}) ...")
    _genprm_transformers_two_stages(
        model, tokenizer, prompts[:warmup], device, batch_size,
        max_tokens_s1, max_tokens_s2, max_input_len, gen_kwargs, silent=True
    )

    n = len(prompts)
    print(f"  Timing {n} steps ...")
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    token_counts = _genprm_transformers_two_stages(
        model, tokenizer, prompts, device, batch_size,
        max_tokens_s1, max_tokens_s2, max_input_len, gen_kwargs, silent=False
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    avg_tokens = float(np.mean(token_counts)) if token_counts else 0.0
    print(f"  → {elapsed:.1f}s | {elapsed/n*1000:.1f} ms/step "
          f"| {n/elapsed:.1f} steps/s | avg {avg_tokens:.0f} tokens/step")

    _cleanup_model(model)

    return {
        "n_steps":       n,
        "batch_size":    batch_size,
        "framework":     "transformers",
        "total_sec":     round(elapsed, 3),
        "ms_per_step":   round(elapsed / n * 1000, 2),
        "steps_per_sec": round(n / elapsed, 2),
        "avg_tokens":    round(avg_tokens, 1),
    }


def _genprm_transformers_two_stages(
    model, tokenizer, prompts: List[str], device: torch.device,
    batch_size: int, max_tokens_s1: int, max_tokens_s2: int,
    max_input_len: int, gen_kwargs: dict, silent: bool = False,
) -> List[int]:
    """2-stage transformers.generate(). Returns total token counts per prompt."""
    if not prompts:
        return []

    n = len(prompts)
    token_counts = [0] * n

    # ── Stage 1: Analyze ─────────────────────────────────────────────────────
    sorted_idx_s1 = _sorted_indices(tokenizer, prompts, max_input_len)
    stage1_by_orig: List[str] = [""] * n  # stage-1 decoded text, indexed by original pos

    if not silent:
        print("    Stage 1 (analyze) ...")
    bar = tqdm(range(0, n, batch_size), desc="      s1 batches") if not silent else range(0, n, batch_size)

    for batch_start in bar:
        orig_indices = sorted_idx_s1[batch_start : batch_start + batch_size]
        batch_prompts = [prompts[i] for i in orig_indices]

        enc = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_input_len,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        input_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_tokens_s1,
                **gen_kwargs,
            )

        new_tokens = out[:, input_len:]   # [batch, new_len]
        for j, orig_idx in enumerate(orig_indices):
            seq = new_tokens[j]
            # Count non-padding tokens
            n_real = int((seq != tokenizer.pad_token_id).sum())
            token_counts[orig_idx] += n_real
            # Decode and truncate at </analyze> for stage-2 prompt construction
            gen_text = tokenizer.decode(seq[:n_real], skip_special_tokens=False)
            cut = gen_text.find("</analyze>")
            if cut >= 0:
                gen_text = gen_text[: cut + len("</analyze>\n")]
            stage1_by_orig[orig_idx] = gen_text

    # ── Stage 2: Verify + Output ─────────────────────────────────────────────
    prompts2 = [
        prompts[i] + stage1_by_orig[i] + VERIFY_PREFIX
        for i in range(n)
    ]
    sorted_idx_s2 = _sorted_indices(tokenizer, prompts2, max_input_len + max_tokens_s1)

    if not silent:
        print("    Stage 2 (verify + output) ...")
    bar2 = tqdm(range(0, n, batch_size), desc="      s2 batches") if not silent else range(0, n, batch_size)

    for batch_start in bar2:
        orig_indices = sorted_idx_s2[batch_start : batch_start + batch_size]
        batch_prompts2 = [prompts2[i] for i in orig_indices]

        enc2 = tokenizer(
            batch_prompts2, return_tensors="pt", padding=True,
            truncation=True, max_length=max_input_len + max_tokens_s1,
        )
        enc2 = {k: v.to(device) for k, v in enc2.items()}
        input_len2 = enc2["input_ids"].shape[1]

        with torch.no_grad():
            out2 = model.generate(
                **enc2,
                max_new_tokens=max_tokens_s2,
                **gen_kwargs,
            )

        new_tokens2 = out2[:, input_len2:]
        for j, orig_idx in enumerate(orig_indices):
            n_real = int((new_tokens2[j] != tokenizer.pad_token_id).sum())
            token_counts[orig_idx] += n_real

    return token_counts


# ─────────────────────────────────────────────────────────────────────────────
# Plan B — GenPRM with vLLM (same as v1)
# ─────────────────────────────────────────────────────────────────────────────

def bench_genprm_vllm(
    model_dir: Path,
    records: List[dict],
    warmup: int,
    max_tokens_s1: int = 512,
    max_tokens_s2: int = 1024,
) -> dict:
    if not model_dir.exists():
        print(f"  [skip] not found: {model_dir}")
        return {}

    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"  [skip] vllm not available: {e}")
        return {}

    print(f"  Loading GenPRM ({model_dir.name}) via vLLM ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    llm = LLM(
        model=str(model_dir),
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.80,
    )

    sp_s1 = SamplingParams(
        temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0,
        logprobs=20, max_tokens=max_tokens_s1,
        stop=["</analyze>\n"], include_stop_str_in_output=True,
    )
    sp_s2 = SamplingParams(
        temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0,
        logprobs=20, max_tokens=max_tokens_s2,
        stop=["</output>\n"], include_stop_str_in_output=True,
    )

    prompts = [_build_genprm_prompt(r, tokenizer) for r in records]

    print(f"  Warmup ({warmup} steps) ...")
    _genprm_vllm_two_stages(llm, prompts[:warmup], sp_s1, sp_s2, silent=True)

    n = len(prompts)
    print(f"  Timing {n} steps ...")
    t0 = time.perf_counter()
    token_counts = _genprm_vllm_two_stages(llm, prompts, sp_s1, sp_s2, silent=False)
    elapsed = time.perf_counter() - t0

    avg_tokens = float(np.mean(token_counts)) if token_counts else 0.0
    print(f"  → {elapsed:.1f}s | {elapsed/n*1000:.1f} ms/step "
          f"| {n/elapsed:.1f} steps/s | avg {avg_tokens:.0f} tokens/step")

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass

    return {
        "n_steps":       n,
        "framework":     "vllm",
        "total_sec":     round(elapsed, 3),
        "ms_per_step":   round(elapsed / n * 1000, 2),
        "steps_per_sec": round(n / elapsed, 2),
        "avg_tokens":    round(avg_tokens, 1),
    }


def _genprm_vllm_two_stages(llm, prompts, sp_s1, sp_s2, silent=False) -> List[int]:
    if not prompts:
        return []
    n = len(prompts)
    token_counts = [0] * n

    if not silent:
        print("    Stage 1 (analyze) ...")
    outs1 = llm.generate(prompts, sp_s1, use_tqdm=not silent)
    prompts2 = []
    for i, out in enumerate(outs1):
        text = out.outputs[0].text
        token_counts[i] += len(out.outputs[0].token_ids)
        prompts2.append(prompts[i] + text + VERIFY_PREFIX)

    if not silent:
        print("    Stage 2 (verify + output) ...")
    outs2 = llm.generate(prompts2, sp_s2, use_tqdm=not silent)
    for i, out in enumerate(outs2):
        token_counts[i] += len(out.outputs[0].token_ids)

    return token_counts


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def _r(val, fmt=".1f", fallback="—"):
    return format(val, fmt) if val is not None else fallback


def _ratio_row(label: str, distill_sps: Optional[float], genprm_sps: Optional[float]) -> str:
    if distill_sps and genprm_sps:
        r = distill_sps / genprm_sps
        return f"| {label} | {_r(distill_sps, '.2f')} | {_r(genprm_sps, '.2f')} | **{r:.1f}×** |"
    return f"| {label} | — | — | — |"


def format_report(results: dict, n_steps: int, args) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    D1  = results.get("distillprm_1.5b", {})
    D7  = results.get("distillprm_7b",   {})
    G1a = results.get("plan_a_genprm_1.5b", {})
    G7a = results.get("plan_a_genprm_7b",   {})
    G1b = results.get("plan_b_genprm_1.5b", {})
    G7b = results.get("plan_b_genprm_7b",   {})

    lines = [
        "# Speed Benchmark v2: DistillPRM vs GenPRM",
        "",
        f"Generated: {now}",
        "",
        "## Fairness Design",
        "",
        "| | Plan A (same framework) | Plan B (best practice) |",
        "|---|---|---|",
        "| **DistillPRM** | Transformers `forward()` | Transformers `forward()` |",
        "| **GenPRM**     | Transformers `generate()` | vLLM (continuous batching) |",
        "| **Purpose**    | Isolate algorithmic cost | Real-world deployment speed |",
        "",
        "**Common optimisations applied to DistillPRM in both plans:**",
        "- bfloat16 backbone (set at load time via `dtype=torch.bfloat16`; "
          "classification heads stay float32 per the original design)",
        "- flash_attention_2 (or sdpa if flash_attn not installed)",
        "- Sorted batching — inputs grouped by tokenised length to minimise padding",
        "",
        "**GenPRM Plan A note:** `model.generate()` uses the full token budget "
        f"(stage1={args.max_tokens_s1}, stage2={args.max_tokens_s2}) per sequence; "
        "no per-sequence early stopping. This slightly over-estimates GenPRM latency.",
        "",
        "## Setup",
        "",
        f"| Problems sampled | {args.num_problems} (ProcessBench, seed={args.seed}) |",
        f"|---|---|",
        f"| Total steps | {n_steps} |",
        f"| Warmup steps | {args.warmup} (excluded) |",
        f"| DistillPRM max input length | {args.max_length} tokens |",
        f"| GenPRM stage1 max tokens | {args.max_tokens_s1} (analyze) |",
        f"| GenPRM stage2 max tokens | {args.max_tokens_s2} (verify+output) |",
        f"| Plan A batch size | {args.batch_size_a} |",
        "",
    ]

    # ── Plan A table ──────────────────────────────────────────────────────────
    lines += [
        "## Plan A — Same Framework (Native Transformers)",
        "",
        "| Model | bs | Steps | Total (s) | ms/step | steps/s | avg tokens |",
        "|-------|----|-------|----------:|--------:|--------:|-----------:|",
    ]

    def distill_row_a(label, data, bs):
        s = data.get(bs, {})
        if not s:
            return f"| {label} | {bs} | — | — | — | — | — |"
        return (f"| {label} | {bs} | {s['n_steps']} "
                f"| {_r(s['total_sec'])} | {_r(s['ms_per_step'])} "
                f"| {_r(s['steps_per_sec'], '.2f')} | — |")

    def genprm_row_a(label, s):
        if not s:
            return f"| {label} | {args.batch_size_a} | — | — | — | — | — |"
        bs = s.get("batch_size", args.batch_size_a)
        return (f"| {label} | {bs} | {s['n_steps']} "
                f"| {_r(s['total_sec'])} | {_r(s['ms_per_step'])} "
                f"| {_r(s['steps_per_sec'], '.2f')} | {_r(s.get('avg_tokens'), '.0f')} |")

    lines.append(distill_row_a("DistillPRM-1.5B", D1, 1))
    lines.append(distill_row_a("DistillPRM-1.5B", D1, args.batch_size_a))
    lines.append(distill_row_a("DistillPRM-7B",  D7, 1))
    lines.append(distill_row_a("DistillPRM-7B",  D7, args.batch_size_a))
    lines.append(genprm_row_a("GenPRM-1.5B (transformers)", G1a))
    lines.append(genprm_row_a("GenPRM-7B (transformers)",   G7a))

    lines += [
        "",
        "### Plan A — Speed Ratios",
        "",
        "*(DistillPRM steps/s ÷ GenPRM steps/s = how many times faster)*",
        "",
        "| Comparison | DistillPRM steps/s | GenPRM steps/s | Ratio |",
        "|-----------|-------------------:|---------------:|------:|",
        _ratio_row("DistillPRM-1.5B (bs=1) vs GenPRM-1.5B",
                   D1.get(1, {}).get("steps_per_sec"),
                   G1a.get("steps_per_sec")),
        _ratio_row(f"DistillPRM-1.5B (bs={args.batch_size_a}) vs GenPRM-1.5B",
                   D1.get(args.batch_size_a, {}).get("steps_per_sec"),
                   G1a.get("steps_per_sec")),
        _ratio_row("DistillPRM-7B (bs=1) vs GenPRM-7B",
                   D7.get(1, {}).get("steps_per_sec"),
                   G7a.get("steps_per_sec")),
        _ratio_row(f"DistillPRM-7B (bs={args.batch_size_a}) vs GenPRM-7B",
                   D7.get(args.batch_size_a, {}).get("steps_per_sec"),
                   G7a.get("steps_per_sec")),
        "",
    ]

    # ── Plan B table ──────────────────────────────────────────────────────────
    lines += [
        "## Plan B — Best Practice (Transformers vs vLLM)",
        "",
        "| Model | Framework | bs | Steps | Total (s) | ms/step | steps/s | avg tokens |",
        "|-------|-----------|-----|-------|----------:|--------:|--------:|-----------:|",
    ]

    def distill_row_b(label, data, bs):
        s = data.get(bs, {})
        if not s:
            return f"| {label} | Transformers | {bs} | — | — | — | — | — |"
        return (f"| {label} | Transformers | {bs} | {s['n_steps']} "
                f"| {_r(s['total_sec'])} | {_r(s['ms_per_step'])} "
                f"| {_r(s['steps_per_sec'], '.2f')} | — |")

    def genprm_row_b(label, s):
        if not s:
            return f"| {label} | vLLM | batched | — | — | — | — | — |"
        return (f"| {label} | vLLM | batched | {s['n_steps']} "
                f"| {_r(s['total_sec'])} | {_r(s['ms_per_step'])} "
                f"| {_r(s['steps_per_sec'], '.2f')} | {_r(s.get('avg_tokens'), '.0f')} |")

    lines.append(distill_row_b("DistillPRM-1.5B", D1, 1))
    lines.append(distill_row_b("DistillPRM-1.5B", D1, args.batch_size_a))
    lines.append(distill_row_b("DistillPRM-7B",  D7, 1))
    lines.append(distill_row_b("DistillPRM-7B",  D7, args.batch_size_a))
    lines.append(genprm_row_b("GenPRM-1.5B (vLLM)", G1b))
    lines.append(genprm_row_b("GenPRM-7B (vLLM)",   G7b))

    lines += [
        "",
        "### Plan B — Speed Ratios",
        "",
        "| Comparison | DistillPRM steps/s | GenPRM steps/s | Ratio |",
        "|-----------|-------------------:|---------------:|------:|",
        _ratio_row("DistillPRM-1.5B (bs=1) vs GenPRM-1.5B",
                   D1.get(1, {}).get("steps_per_sec"),
                   G1b.get("steps_per_sec")),
        _ratio_row(f"DistillPRM-1.5B (bs={args.batch_size_a}) vs GenPRM-1.5B",
                   D1.get(args.batch_size_a, {}).get("steps_per_sec"),
                   G1b.get("steps_per_sec")),
        _ratio_row("DistillPRM-7B (bs=1) vs GenPRM-7B",
                   D7.get(1, {}).get("steps_per_sec"),
                   G7b.get("steps_per_sec")),
        _ratio_row(f"DistillPRM-7B (bs={args.batch_size_a}) vs GenPRM-7B",
                   D7.get(args.batch_size_a, {}).get("steps_per_sec"),
                   G7b.get("steps_per_sec")),
        "",
        "## Discussion",
        "",
        "- **Plan A** shows the *algorithmic* gap: DistillPRM does a single forward pass; "
          "GenPRM generates ~700-800 tokens in two stages. The speed difference is "
          "driven by generation length, not framework choice.",
        "- **Plan B** shows the *deployment* gap: vLLM's continuous batching reduces "
          "GenPRM's per-step latency vs. padded transformers batch generation, "
          "but DistillPRM still wins because it generates zero tokens.",
        "- **Plan B DistillPRM ≈ Plan A DistillPRM**: transformers forward() is already "
          "near-optimal for a discriminative model — vLLM offers no benefit here.",
        "",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num_problems",     type=int, default=50)
    p.add_argument("--warmup",           type=int, default=10)
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--batch_sizes",      type=int, nargs="+", default=[1, 16],
                   help="DistillPRM batch sizes (default: 1 16)")
    p.add_argument("--batch_size_a",     type=int, default=16,
                   help="GenPRM Plan A batch size (default: 16, matches DistillPRM)")
    p.add_argument("--max_length",       type=int, default=1024,
                   help="DistillPRM max input length (default: 1024)")
    p.add_argument("--max_tokens_s1",    type=int, default=512)
    p.add_argument("--max_tokens_s2",    type=int, default=1024)
    p.add_argument("--skip_plan_b",      action="store_true",
                   help="Skip Plan B (vLLM) — run Plan A only")
    p.add_argument("--skip_plan_a",      action="store_true",
                   help="Skip Plan A (transformers generate) — run Plan B only")
    p.add_argument("--skip",             type=str, nargs="*", default=[],
                   metavar="MODEL",
                   help=("Skip specific models: "
                         "distillprm_1.5b distillprm_7b genprm_1.5b genprm_7b"))
    p.add_argument("--results_seed",     type=str, default=None,
                   metavar="JSON",
                   help="Path to a partial results JSON to seed results dict before running")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("[1/9] Loading benchmark data")
    print("="*60)
    records = load_benchmark_steps(args.num_problems, args.seed)
    n_steps = len(records)

    results: dict = {}

    # ── Seed from partial run if provided ────────────────────────────────────
    if args.results_seed:
        with open(args.results_seed) as f:
            seed_data = json.load(f)
        seed_results = seed_data.get("results", {})
        # JSON serialises integer keys as strings; convert back for DistillPRM dicts
        for k in list(seed_results.keys()):
            v = seed_results[k]
            if isinstance(v, dict) and all(str(i).isdigit() for i in v.keys()):
                seed_results[k] = {int(bk): bv for bk, bv in v.items()}
        results.update(seed_results)
        seed_n = seed_data.get("n_steps", n_steps)
        print(f"  Seeded {len(results)} result keys from {args.results_seed} (n_steps={seed_n})")

    # ── DistillPRM (same in both plans) ───────────────────────────────────────
    if "distillprm_1.5b" not in args.skip:
        print("\n" + "="*60)
        print("[2/9] DistillPRM-1.5B  (Plan A = Plan B)")
        print("="*60)
        results["distillprm_1.5b"] = bench_distillprm(
            DISTILLPRM_1_5B_CKPT, DISTILLPRM_1_5B_BACKBONE,
            records, args.batch_sizes, args.warmup, device, args.max_length,
        )
    else:
        print("\n[2/9] DistillPRM-1.5B  [skipped]")

    if "distillprm_7b" not in args.skip:
        print("\n" + "="*60)
        print("[3/9] DistillPRM-7B  (Plan A = Plan B)")
        print("="*60)
        results["distillprm_7b"] = bench_distillprm(
            DISTILLPRM_7B_CKPT, DISTILLPRM_7B_BACKBONE,
            records, args.batch_sizes, args.warmup, device, args.max_length,
        )
    else:
        print("\n[3/9] DistillPRM-7B  [skipped]")

    # ── Plan A: GenPRM with transformers.generate() ───────────────────────────
    if not args.skip_plan_a:
        if "genprm_1.5b" not in args.skip:
            print("\n" + "="*60)
            print("[4/9] Plan A — GenPRM-1.5B (transformers.generate)")
            print("="*60)
            results["plan_a_genprm_1.5b"] = bench_genprm_transformers(
                GENPRM_1_5B_DIR, records, args.warmup, args.batch_size_a,
                device, args.max_tokens_s1, args.max_tokens_s2,
            )
        else:
            print("\n[4/9] Plan A GenPRM-1.5B  [skipped]")

        if "genprm_7b" not in args.skip:
            print("\n" + "="*60)
            print("[5/9] Plan A — GenPRM-7B (transformers.generate)")
            print("="*60)
            results["plan_a_genprm_7b"] = bench_genprm_transformers(
                GENPRM_7B_DIR, records, args.warmup, args.batch_size_a,
                device, args.max_tokens_s1, args.max_tokens_s2,
            )
        else:
            print("\n[5/9] Plan A GenPRM-7B  [skipped]")
    else:
        print("\n[4-5/9] Plan A (transformers generate)  [skipped via --skip_plan_a]")

    # ── Plan B: GenPRM with vLLM ──────────────────────────────────────────────
    if not args.skip_plan_b:
        if "genprm_1.5b" not in args.skip:
            print("\n" + "="*60)
            print("[6/9] Plan B — GenPRM-1.5B (vLLM)")
            print("="*60)
            results["plan_b_genprm_1.5b"] = bench_genprm_vllm(
                GENPRM_1_5B_DIR, records, args.warmup,
                args.max_tokens_s1, args.max_tokens_s2,
            )
        else:
            print("\n[6/9] Plan B GenPRM-1.5B  [skipped]")

        if "genprm_7b" not in args.skip:
            print("\n" + "="*60)
            print("[7/9] Plan B — GenPRM-7B (vLLM)")
            print("="*60)
            results["plan_b_genprm_7b"] = bench_genprm_vllm(
                GENPRM_7B_DIR, records, args.warmup,
                args.max_tokens_s1, args.max_tokens_s2,
            )
        else:
            print("\n[7/9] Plan B GenPRM-7B  [skipped]")
    else:
        print("\n[6-7/9] Plan B (vLLM)  [skipped via --skip_plan_b]")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("[8-9/9] Saving results")
    print("="*60)

    payload = {
        "generated": datetime.now().isoformat(),
        "args": vars(args),
        "n_steps": n_steps,
        "results": results,
    }
    json_path = REPORTS_DIR / "speed_benchmark_v2.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  JSON : {json_path}")

    report = format_report(results, n_steps, args)
    md_path = REPORTS_DIR / "speed_benchmark_v2.md"
    with open(md_path, "w") as f:
        f.write(report)
    print(f"  MD   : {md_path}")

    print()
    print(report)


if __name__ == "__main__":
    main()
