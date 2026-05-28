#!/usr/bin/env python3
"""Speed benchmark: DistillPRM (discriminative) vs GenPRM (generative).

Randomly samples N problems from ProcessBench, expands to all their steps
(~200-400 steps total), then times each model under identical single-GPU
conditions.

Models benchmarked
------------------
  A. DistillPRM-1.5B  -- single forward pass, batch_size=1 and batch_size=16
  B. DistillPRM-7B    -- single forward pass, batch_size=1 and batch_size=16
  C. GenPRM-1.5B      -- 2-stage vLLM generation (analyze → verify+output)
  D. GenPRM-7B        -- 2-stage vLLM generation (analyze → verify+output)

GenPRM stages
-------------
  Stage 1: generate <analyze>...</analyze> (stop='</analyze>\\n', max 512 tok)
  Stage 2: append <verify> prefix, generate through verify code + <output>
           judgment (stop='</output>\\n', max 1024 tok, no code execution)

Usage
-----
  python distill/step_benchmark_speed.py [options]
  python distill/step_benchmark_speed.py --skip genprm_7b        # skip one
  python distill/step_benchmark_speed.py --num_problems 20       # quick run

Output
------
  distill/reports/speed_benchmark.json
  distill/reports/speed_benchmark.md
"""

import os
# Must be set before vLLM spawns any subprocess
os.environ["VLLM_USE_V1"] = "0"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

import argparse
import gc
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "distill"))

# ── model / data paths ────────────────────────────────────────────────────────
DISTILLPRM_1_5B_CKPT     = ROOT / "models/DistillPRM-1.5B/adaptive_t3/best_model.pt"
DISTILLPRM_7B_CKPT       = ROOT / "models/DistillPRM-7B/adaptive_t3/epoch_02.pt"
DISTILLPRM_1_5B_BACKBONE = ROOT / "models/Qwen2.5-Math-1.5B"
DISTILLPRM_7B_BACKBONE   = ROOT / "models/Qwen2.5-Math-7B"
GENPRM_1_5B_DIR          = ROOT / "models/GenPRM-1.5B"
GENPRM_7B_DIR            = ROOT / "models/GenPRM-7B"
PROCESSBENCH_DIR         = ROOT / "data/ProcessBench"
REPORTS_DIR              = ROOT / "distill/reports"

PROCESSBENCH_SPLITS = ["gsm8k", "math", "olympiadbench", "omnimath"]

# ── GenPRM prompt constants (from GenPRM/src/prm_evaluation/genprm_inference.py)
GENPRM_SYSTEM = (
    "You are a math teacher. Your task is to review and critique "
    "the paragraphs in solution step by step."
)
ANALYZE_TEMPLATE = "<analyze>\nLet's analyze the Paragraph {cur_step} step by step: "
VERIFY_PREFIX    = "<verify>\nLet's use python code to find any potential error:\n```python\n"
OUTPUT_PREFIX    = "<output>\n**Judgement**: $\\boxed"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_benchmark_steps(num_problems: int, seed: int) -> List[dict]:
    """Sample `num_problems` from ProcessBench and return per-step records."""
    rng = random.Random(seed)
    all_problems = []
    for split in PROCESSBENCH_SPLITS:
        fpath = PROCESSBENCH_DIR / f"{split}.json"
        if not fpath.exists():
            print(f"  [warn] not found: {fpath}")
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
        context_parts: List[str] = []
        for i, step_text in enumerate(steps):
            records.append({
                "question":     question,
                "context":      "\n\n".join(context_parts),
                "current_step": step_text,
                "step_idx":     i,
                "step_num":     i + 1,   # 1-indexed (for GenPRM analyze template)
                "_split":       prob["_split"],
            })
            context_parts.append(step_text)

    print(f"  Sampled {len(selected)} problems → {len(records)} steps "
          f"(avg {len(records)/max(len(selected),1):.1f} steps/problem)")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# DistillPRM benchmark
# ─────────────────────────────────────────────────────────────────────────────

def _load_distillprm_checkpoint(model: "DistillPRM", ckpt_path: Path) -> None:
    """Load checkpoint saved by step5_train_distillpRM.py into `model`."""
    print(f"    Loading checkpoint: {ckpt_path} ...")
    # weights_only=True matches how step6_evaluate.py loads the model
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    # Strip DDP "module." prefix if present
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def bench_distillprm(
    ckpt_path: Path,
    backbone_path: Path,
    records: List[dict],
    batch_sizes: List[int],
    warmup: int,
    device: torch.device,
    max_length: int = 1024,
) -> Dict[int, dict]:
    """Load DistillPRM and benchmark at each batch size. Returns {bs: stats}."""
    from step4_build_student_model import DistillPRM, build_input_text
    from transformers import AutoTokenizer

    if not ckpt_path.exists():
        print(f"  [skip] checkpoint not found: {ckpt_path}")
        return {}
    if not backbone_path.exists():
        print(f"  [skip] backbone not found: {backbone_path}")
        return {}

    print(f"  Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(str(backbone_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Building model ({backbone_path.name}) ...")
    model = DistillPRM(model_name_or_path=str(backbone_path))
    _load_distillprm_checkpoint(model, ckpt_path)
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model loaded ({n_params:.0f}M params)")

    # Pre-build all input strings (done once, shared across batch sizes)
    texts = [
        build_input_text(r["question"], r["context"], r["current_step"])
        for r in records
    ]

    results: Dict[int, dict] = {}
    for bs in batch_sizes:
        print(f"\n  [batch_size={bs}] warmup ({warmup} steps) ...")
        _distillprm_forward(model, tokenizer, texts[:max(warmup, bs)],
                            device, bs, max_length)

        print(f"  [batch_size={bs}] timing {len(texts)} steps ...")
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _distillprm_forward(model, tokenizer, texts, device, bs, max_length)
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
        print(f"    → {elapsed:.1f}s total | {elapsed/n*1000:.1f} ms/step "
              f"| {n/elapsed:.1f} steps/s")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


@torch.no_grad()
def _distillprm_forward(
    model, tokenizer, texts: List[str],
    device: torch.device, batch_size: int, max_length: int,
) -> None:
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        model(enc["input_ids"], enc["attention_mask"])


# ─────────────────────────────────────────────────────────────────────────────
# GenPRM benchmark
# ─────────────────────────────────────────────────────────────────────────────

def _build_genprm_prompt(record: dict, tokenizer) -> str:
    """
    Build a single-turn GenPRM prompt for one step.

    Mirrors the original GenPRM input format: system message + one user message
    containing the full context, then the model is primed with the analyze
    template so generation begins from there.
    """
    parts = [f"Question: {record['question']}"]
    if record["context"]:
        parts.append(f"Context:\n{record['context']}")
    parts.append(f"Current step:\n{record['current_step']}")
    user_content = "\n\n".join(parts)

    messages = [
        {"role": "system", "content": GENPRM_SYSTEM},
        {"role": "user",   "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Strip trailing eos_token (GenPRM convention from build_prompt())
    eos = tokenizer.eos_token or ""
    if eos:
        if prompt.endswith(f"{eos}\n"):
            prompt = prompt[: -len(f"{eos}\n")]
        elif prompt.endswith(eos):
            prompt = prompt[: -len(eos)]

    # Prime with analyze template — model continues from here
    prompt += ANALYZE_TEMPLATE.format(cur_step=record["step_num"])
    return prompt


def _genprm_two_stages(
    llm,
    prompts: List[str],
    sp_analyze,
    sp_verify_output,
    silent: bool = False,
) -> List[int]:
    """
    Run GenPRM 2-stage batched inference. Returns per-step total token counts.

    Stage 1 — Analyze
        Generate until </analyze>\\n (max 512 tokens).
    Stage 2 — Verify + Output (no code execution)
        Append <verify> prefix to stage-1 output, generate until </output>\\n
        (max 1024 tokens). The model naturally writes the python code block,
        skips the execution placeholder, and ends with the judgment.
    """
    if not prompts:
        return []

    token_counts = [0] * len(prompts)

    # ── Stage 1: Analyze ──────────────────────────────────────────────────────
    if not silent:
        print("    Stage 1 (analyze) ...")
    outs1 = llm.generate(prompts, sp_analyze, use_tqdm=not silent)

    prompts2: List[str] = []
    for i, out in enumerate(outs1):
        gen = out.outputs[0].text
        token_counts[i] += len(out.outputs[0].token_ids)
        # Append verify prefix; model will generate code then the output block
        prompts2.append(prompts[i] + gen + VERIFY_PREFIX)

    # ── Stage 2: Verify + Output ───────────────────────────────────────────────
    if not silent:
        print("    Stage 2 (verify + output) ...")
    outs2 = llm.generate(prompts2, sp_verify_output, use_tqdm=not silent)
    for i, out in enumerate(outs2):
        token_counts[i] += len(out.outputs[0].token_ids)

    return token_counts


def bench_genprm(
    model_dir: Path,
    records: List[dict],
    warmup: int,
    max_tokens_analyze: int = 512,
    max_tokens_verify_output: int = 1024,
) -> dict:
    """Load GenPRM via vLLM and benchmark 2-stage generation. Returns stats."""
    if not model_dir.exists():
        print(f"  [skip] model directory not found: {model_dir}")
        return {}

    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"  [skip] vllm not available: {e}")
        return {}

    print(f"  Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    print(f"  Loading model via vLLM: {model_dir.name} ...")
    llm = LLM(
        model=str(model_dir),
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )

    # Sampling parameters matching GenPRM defaults
    sp_analyze = SamplingParams(
        temperature=0.6, top_p=0.95, top_k=20,
        repetition_penalty=1.0,
        logprobs=20,
        max_tokens=max_tokens_analyze,
        stop=["</analyze>\n"],
        include_stop_str_in_output=True,
    )
    # Stage 2: generate verify code + model naturally transitions to output.
    # We stop at </output>\n which includes the judgment $\boxed{Yes/No}.
    # No code execution — the model generates through the verify section on its
    # own (it may emit \n```\n, then see no code output, then write <output>).
    sp_verify_output = SamplingParams(
        temperature=0.6, top_p=0.95, top_k=20,
        repetition_penalty=1.0,
        logprobs=20,
        max_tokens=max_tokens_verify_output,
        stop=["</output>\n"],
        include_stop_str_in_output=True,
    )

    prompts = [_build_genprm_prompt(r, tokenizer) for r in records]

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f"  Warmup ({warmup} steps) ...")
    warm_prompts = prompts[:warmup] if len(prompts) >= warmup else prompts
    _genprm_two_stages(llm, warm_prompts, sp_analyze, sp_verify_output, silent=True)

    # ── Timed run ─────────────────────────────────────────────────────────────
    n = len(prompts)
    print(f"  Timing {n} steps ...")
    t0 = time.perf_counter()
    token_counts = _genprm_two_stages(
        llm, prompts, sp_analyze, sp_verify_output, silent=False
    )
    elapsed = time.perf_counter() - t0

    avg_tokens = float(np.mean(token_counts)) if token_counts else 0.0
    print(f"  → {elapsed:.1f}s total | {elapsed/n*1000:.1f} ms/step "
          f"| {n/elapsed:.1f} steps/s | avg {avg_tokens:.0f} tokens/step")

    # ── vLLM cleanup ──────────────────────────────────────────────────────────
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
        "total_sec":     round(elapsed, 3),
        "ms_per_step":   round(elapsed / n * 1000, 2),
        "steps_per_sec": round(n / elapsed, 2),
        "avg_tokens":    round(avg_tokens, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def _r(val: Optional[float], fmt: str = ".1f", fallback: str = "—") -> str:
    return format(val, fmt) if val is not None else fallback


def format_report(results: dict, n_steps: int, args) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Speed Benchmark: DistillPRM vs GenPRM",
        "",
        f"Generated: {now}",
        "",
        "## Setup",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Problems sampled | {args.num_problems} (ProcessBench, seed={args.seed}) |",
        f"| Total steps | {n_steps} |",
        f"| Warmup steps | {args.warmup} (excluded from timing) |",
        f"| Max input length (DistillPRM) | {args.max_length} tokens |",
        f"| GenPRM Stage 1 max tokens | {args.max_tokens_analyze} (analyze) |",
        f"| GenPRM Stage 2 max tokens | {args.max_tokens_verify_output} (verify+output) |",
        "| Code execution | disabled (generation time only) |",
        "",
        "## Inference Speed",
        "",
        "| Model | Type | Batch | Steps | Total (s) | ms/step | steps/s | avg tokens |",
        "|-------|------|-------|-------|----------:|--------:|--------:|-----------:|",
    ]

    def distill_row(label: str, key: str, bs: int) -> str:
        s = results.get(key, {}).get(bs)
        if not s:
            return f"| {label} | Discriminative | {bs} | — | — | — | — | — |"
        return (f"| {label} | Discriminative | {bs} "
                f"| {s['n_steps']} "
                f"| {_r(s['total_sec'])} "
                f"| {_r(s['ms_per_step'])} "
                f"| {_r(s['steps_per_sec'], '.2f')} "
                f"| — |")

    def genprm_row(label: str, key: str) -> str:
        s = results.get(key)
        if not s:
            return f"| {label} | Generative (vLLM) | batched | — | — | — | — | — |"
        return (f"| {label} | Generative (vLLM) | batched "
                f"| {s['n_steps']} "
                f"| {_r(s['total_sec'])} "
                f"| {_r(s['ms_per_step'])} "
                f"| {_r(s['steps_per_sec'], '.2f')} "
                f"| {_r(s.get('avg_tokens'), '.0f')} |")

    for bs in args.batch_sizes:
        lines.append(distill_row("DistillPRM-1.5B", "distillprm_1.5b", bs))
    for bs in args.batch_sizes:
        lines.append(distill_row("DistillPRM-7B",  "distillprm_7b",  bs))
    lines.append(genprm_row("GenPRM-1.5B", "genprm_1.5b"))
    lines.append(genprm_row("GenPRM-7B",   "genprm_7b"))

    # Speed ratios
    lines += [
        "",
        "## Speed Ratios",
        "",
        "Steps/s(DistillPRM) / Steps/s(GenPRM) — how many times faster DistillPRM is.",
        "",
        "| Comparison | DistillPRM steps/s | GenPRM steps/s | Ratio |",
        "|-----------|-------------------:|---------------:|------:|",
    ]

    def ratio_row(label: str, genprm_key: str, distill_key: str, bs: int) -> str:
        g = results.get(genprm_key, {})
        d = results.get(distill_key, {}).get(bs, {})
        gs = g.get("steps_per_sec") if g else None
        ds = d.get("steps_per_sec") if d else None
        if gs and ds:
            ratio = ds / gs
            return (f"| {label} "
                    f"| {_r(ds, '.2f')} "
                    f"| {_r(gs, '.2f')} "
                    f"| **{ratio:.1f}× faster** |")
        return f"| {label} | — | — | — |"

    for bs in args.batch_sizes:
        lines.append(ratio_row(
            f"DistillPRM-1.5B (bs={bs}) vs GenPRM-1.5B",
            "genprm_1.5b", "distillprm_1.5b", bs))
    for bs in args.batch_sizes:
        lines.append(ratio_row(
            f"DistillPRM-7B (bs={bs}) vs GenPRM-7B",
            "genprm_7b", "distillprm_7b", bs))

    lines += [
        "",
        "## Notes",
        "",
        "- **DistillPRM**: single `model.forward()` call per batch; "
          "timing includes tokenization and GPU transfer.",
        "- **GenPRM**: two batched vLLM calls per step set — "
          "(1) analyze until `</analyze>`, (2) verify+output until `</output>`. "
          "All steps are processed in parallel within each stage.",
        "- Code execution (Stage 2 `exec()`) is **excluded**; only generation "
          "latency is measured.",
        "- Both model families evaluated on the same single GPU.",
        "- Input context grows with step index (early steps are shorter); "
          "reported times reflect the natural distribution from ProcessBench.",
        "",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num_problems",           type=int,   default=50,
                   help="Number of ProcessBench problems to sample (default: 50)")
    p.add_argument("--warmup",                 type=int,   default=10,
                   help="Warmup steps excluded from timing (default: 10)")
    p.add_argument("--seed",                   type=int,   default=42)
    p.add_argument("--batch_sizes",            type=int,   nargs="+", default=[1, 16],
                   help="DistillPRM batch sizes to test (default: 1 16)")
    p.add_argument("--max_length",             type=int,   default=1024,
                   help="Max input token length for DistillPRM (default: 1024)")
    p.add_argument("--max_tokens_analyze",     type=int,   default=512,
                   help="GenPRM Stage 1 max tokens (default: 512)")
    p.add_argument("--max_tokens_verify_output", type=int, default=1024,
                   help="GenPRM Stage 2 max tokens (default: 1024)")
    p.add_argument("--skip", type=str, nargs="*", default=[],
                   metavar="MODEL",
                   help=("Models to skip. Choices: "
                         "distillprm_1.5b distillprm_7b genprm_1.5b genprm_7b"))
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM:   {total_mem:.0f} GB")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load benchmark data ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("[1/5] Loading benchmark data")
    print("="*60)
    records = load_benchmark_steps(args.num_problems, args.seed)
    n_steps = len(records)

    results: dict = {}

    # ── 2. DistillPRM-1.5B ───────────────────────────────────────────────────
    if "distillprm_1.5b" not in args.skip:
        print("\n" + "="*60)
        print("[2/5] DistillPRM-1.5B")
        print("="*60)
        results["distillprm_1.5b"] = bench_distillprm(
            ckpt_path=DISTILLPRM_1_5B_CKPT,
            backbone_path=DISTILLPRM_1_5B_BACKBONE,
            records=records,
            batch_sizes=args.batch_sizes,
            warmup=args.warmup,
            device=device,
            max_length=args.max_length,
        )
    else:
        print("\n[2/5] DistillPRM-1.5B  [skipped]")

    # ── 3. DistillPRM-7B ─────────────────────────────────────────────────────
    if "distillprm_7b" not in args.skip:
        print("\n" + "="*60)
        print("[3/5] DistillPRM-7B")
        print("="*60)
        results["distillprm_7b"] = bench_distillprm(
            ckpt_path=DISTILLPRM_7B_CKPT,
            backbone_path=DISTILLPRM_7B_BACKBONE,
            records=records,
            batch_sizes=args.batch_sizes,
            warmup=args.warmup,
            device=device,
            max_length=args.max_length,
        )
    else:
        print("\n[3/5] DistillPRM-7B  [skipped]")

    # ── 4. GenPRM-1.5B ───────────────────────────────────────────────────────
    if "genprm_1.5b" not in args.skip:
        print("\n" + "="*60)
        print("[4/5] GenPRM-1.5B")
        print("="*60)
        results["genprm_1.5b"] = bench_genprm(
            model_dir=GENPRM_1_5B_DIR,
            records=records,
            warmup=args.warmup,
            max_tokens_analyze=args.max_tokens_analyze,
            max_tokens_verify_output=args.max_tokens_verify_output,
        )
    else:
        print("\n[4/5] GenPRM-1.5B  [skipped]")

    # ── 5. GenPRM-7B ─────────────────────────────────────────────────────────
    if "genprm_7b" not in args.skip:
        print("\n" + "="*60)
        print("[5/5] GenPRM-7B")
        print("="*60)
        results["genprm_7b"] = bench_genprm(
            model_dir=GENPRM_7B_DIR,
            records=records,
            warmup=args.warmup,
            max_tokens_analyze=args.max_tokens_analyze,
            max_tokens_verify_output=args.max_tokens_verify_output,
        )
    else:
        print("\n[5/5] GenPRM-7B  [skipped]")

    # ── Save results ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Saving results")
    print("="*60)

    payload = {
        "generated": datetime.now().isoformat(),
        "args": vars(args),
        "n_steps": n_steps,
        "results": results,
    }
    json_path = REPORTS_DIR / "speed_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  JSON : {json_path}")

    report = format_report(results, n_steps, args)
    md_path = REPORTS_DIR / "speed_benchmark.md"
    with open(md_path, "w") as f:
        f.write(report)
    print(f"  MD   : {md_path}")

    print()
    print(report)


if __name__ == "__main__":
    main()
