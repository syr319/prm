"""
Best-of-N Candidate Generation — MATH500

Generates N candidate solutions per problem using Qwen2.5-Math-7B-Instruct via vLLM.
Results saved to data/math500_candidates.json for downstream bon_rerank.py evaluation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 distill/bon_generate.py

Tip: generation takes ~25-35 min on a single H20 GPU for 500 × 32 candidates.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "distill"))


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_math500(cache_path: Path) -> list:
    """Load MATH-500 test set, caching locally to avoid repeated downloads."""
    if cache_path.exists():
        print(f"Loading MATH-500 from local cache: {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    print("Downloading MATH-500 from HuggingFace (hf-mirror.com) ...")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = [
        {
            "problem":   r["problem"],
            "answer":    r["answer"],
            "subject":   r.get("subject", ""),
            "level":     r.get("level", 0),
            "unique_id": r.get("unique_id", str(i)),
        }
        for i, r in enumerate(ds)
    ]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(problems, f, ensure_ascii=False, indent=2)
    print(f"Cached {len(problems)} problems → {cache_path}")
    return problems


# ─── Prompt formatting ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def build_prompt(problem: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ─── Generation ───────────────────────────────────────────────────────────────

def generate_candidates(
    problems:       list,
    model_path:     str,
    num_candidates: int,
    temperature:    float,
    max_new_tokens: int,
    max_model_len:  int,
    existing:       dict,   # problem_text → candidates (for resuming)
) -> list:
    """Run vLLM generation. Skips problems already in `existing`."""
    from vllm import LLM, SamplingParams            # type: ignore
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Filter out already-generated problems
    todo = [p for p in problems if p["problem"] not in existing]
    if not todo:
        print("All problems already generated. Nothing to do.")
        return []

    print(f"Generating {num_candidates} candidates × {len(todo)} problems "
          f"({len(problems)-len(todo)} already done).")

    prompts = [build_prompt(p["problem"], tokenizer) for p in todo]

    print(f"Loading vLLM model from {model_path} ...")
    llm = LLM(
        model                  = model_path,
        tensor_parallel_size   = 1,
        dtype                  = "bfloat16",
        max_model_len          = max_model_len,
        gpu_memory_utilization = 0.90,
        trust_remote_code      = True,
    )

    sampling_params = SamplingParams(
        temperature    = temperature,
        max_tokens     = max_new_tokens,
        n              = num_candidates,
        stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
    )

    print("Running generation ...")
    outputs = llm.generate(prompts, sampling_params)

    new_results = []
    for prob, output in zip(todo, outputs):
        candidates = [o.text.strip() for o in output.outputs]
        new_results.append({
            "problem":    prob["problem"],
            "answer":     prob["answer"],
            "subject":    prob.get("subject", ""),
            "level":      prob.get("level", 0),
            "unique_id":  prob.get("unique_id", ""),
            "candidates": candidates,
        })

    return new_results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate MATH500 candidates via vLLM.")
    parser.add_argument("--model_path",     default=str(ROOT / "models" / "Qwen2.5-Math-7B-Instruct"))
    parser.add_argument("--output",         default=str(ROOT / "data" / "math500_candidates.json"))
    parser.add_argument("--math500_cache",  default=str(ROOT / "data" / "math500_raw.json"))
    parser.add_argument("--num_candidates", type=int,   default=32)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int,   default=2048)
    parser.add_argument("--max_model_len",  type=int,   default=4096)
    args = parser.parse_args()

    output_path = Path(args.output)
    cache_path  = Path(args.math500_cache)

    # Load MATH-500
    problems = load_math500(cache_path)
    print(f"MATH-500: {len(problems)} problems loaded.")

    # Load existing results (resume support)
    existing = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            prev = json.load(f)
        existing = {r["problem"]: r["candidates"] for r in prev}
        print(f"Resuming: {len(existing)} problems already generated.")

    # Generate
    new_results = generate_candidates(
        problems       = problems,
        model_path     = args.model_path,
        num_candidates = args.num_candidates,
        temperature    = args.temperature,
        max_new_tokens = args.max_new_tokens,
        max_model_len  = args.max_model_len,
        existing       = existing,
    )

    # Merge with existing and sort by original order
    if new_results:
        existing_map = {r["problem"]: r for r in (
            json.load(open(output_path)) if output_path.exists() else []
        )}
        for r in new_results:
            existing_map[r["problem"]] = r

        # Preserve original problem order
        all_results = []
        seen = set()
        for p in problems:
            key = p["problem"]
            if key in existing_map and key not in seen:
                all_results.append(existing_map[key])
                seen.add(key)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(all_results)} problems → {output_path}")
        print(f"Each problem has {len(all_results[0]['candidates'])} candidates.")
    else:
        print(f"No new results to save. Output: {output_path}")


if __name__ == "__main__":
    main()
