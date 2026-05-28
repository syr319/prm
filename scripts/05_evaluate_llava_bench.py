"""
Step 5: Evaluate LLaVA-Bench results using Qwen-VL-Max API (same judge as MM-Vet).

Scores four response sets: Pass@1, LLaVA-Critic BoN@8, VisualPRM BoN@8, OpenPRM BoN@8.
API key read from DASHSCOPE_API_KEY environment variable.
Saves per-question scores and prints final summary table.

Usage:
  export DASHSCOPE_API_KEY=sk-xxx
  python3 scripts/05_evaluate_llava_bench.py
"""

import os
import re
import json
import time
import threading
import argparse
from pathlib import Path
from openai import OpenAI

PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "llava_bench"
BON_DIR  = RESULTS_DIR / "bon"
EVAL_DIR = RESULTS_DIR / "eval"

# Input files for each method
METHOD_FILES = {
    "pass_at_1":             (BON_DIR / "candidates.json",                    "pass_at_1"),
    "llava_critic_bon":      (BON_DIR / "llava_critic_bon.json",               "best_response"),
    "visualprm_bon":         (BON_DIR / "visualprm_bon.json",                  "best_response"),
    "openprm_bon_ckpt2100":  (BON_DIR / "llava_bench_openprm_bon_2100.json",   "best_response"),
    "openprm_bon_ckpt2658":  (BON_DIR / "llava_bench_openprm_bon_2658.json",   "best_response"),
}

EVAL_PROMPT = """\
You are evaluating the quality of an AI assistant's response to a visual question.

Question: {question}
Reference Answer: {reference}
Model Response: {prediction}

The reference answer represents a high-quality response. \
Rate the model's response on a scale of 0 to 1:
- 1.0: The response is as good as or better than the reference answer
- 0.5: The response partially captures the reference answer
- 0.0: The response is incorrect, irrelevant, or significantly worse than the reference

Do not provide any explanation. Output only a single number between 0 and 1.\
"""

MAX_WORKERS = 8
RETRY_ATTEMPTS = 3
RETRY_DELAY = 3

_model_lock = threading.Lock()
_exhausted: set = set()

MODELS = [
    "qwen-vl-max",
]


def _is_quota_error(e: Exception) -> bool:
    msg = str(e)
    if "403" in msg or "AllocationQuota" in msg or "FreeTierOnly" in msg:
        return True
    return any(k in msg.lower() for k in ("quota", "insufficient", "rate limit", "429"))


def get_model() -> str:
    with _model_lock:
        for m in MODELS:
            if m not in _exhausted:
                return m
        _exhausted.clear()
        return MODELS[0]


def mark_exhausted(model: str):
    with _model_lock:
        _exhausted.add(model)
        for m in MODELS:
            if m not in _exhausted:
                print(f"  [rotation] {model!r} → {m!r}")
                return


def get_client() -> OpenAI:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY not set")
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def parse_score(text: str) -> float:
    text = text.strip()
    try:
        return max(0.0, min(1.0, float(text)))
    except ValueError:
        pass
    m = re.search(r"([01](?:\.\d+)?|\d*\.\d+)", text)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    return 0.0


def score_one(client: OpenAI, question: str, reference: str, prediction: str) -> float:
    prompt = EVAL_PROMPT.format(
        question=question, reference=reference, prediction=prediction
    )
    for attempt in range(RETRY_ATTEMPTS):
        model = get_model()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16,
                temperature=0.0,
            )
            return parse_score(resp.choices[0].message.content.strip())
        except Exception as e:
            if _is_quota_error(e):
                mark_exhausted(model)
            elif attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"  ERROR: {e}")
    return 0.0


def evaluate_method(client: OpenAI, name: str, src_file: Path, resp_key: str) -> dict | None:
    out_file = EVAL_DIR / f"eval_{name}.json"

    if not src_file.exists():
        print(f"  SKIP {name}: {src_file.name} not found")
        return None

    with open(src_file) as f:
        data = json.load(f)

    # Checkpoint
    existing = {}
    if out_file.exists():
        with open(out_file) as f:
            existing = json.load(f)
        print(f"  Resuming {name}: {len(existing)} already scored")

    scores = dict(existing)
    todo = {qid: item for qid, item in data.items() if qid not in scores}
    total = len(data)

    for i, (qid, item) in enumerate(todo.items()):
        prediction = item.get(resp_key, "")
        reference  = item.get("answer", item.get("reference", ""))
        question   = item.get("question", "")
        score = score_one(client, question, reference, prediction)
        scores[qid] = {
            "question_id": qid,
            "question":    question,
            "reference":   reference,
            "prediction":  prediction,
            "category":    item.get("category", item.get("capability", "")),
            "score":       score,
        }
        if (i + 1) % 10 == 0 or (i + 1) == len(todo):
            with open(out_file, "w") as f:
                json.dump(scores, f, indent=2, ensure_ascii=False)
            done = len(scores)
            avg = sum(v["score"] for v in scores.values()) / done
            print(f"  [{done}/{total}] {name}: avg={avg:.4f}")

    with open(out_file, "w") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    return scores


def category_breakdown(scores: dict) -> dict[str, float]:
    cat_scores: dict[str, list] = {}
    for item in scores.values():
        cat = item.get("category", "")
        cats = [cat] if isinstance(cat, str) else cat
        for c in cats:
            if c:
                cat_scores.setdefault(c, []).append(item["score"])
    return {c: sum(v) / len(v) for c, v in sorted(cat_scores.items())}


def print_summary(results: dict, breakdowns: dict):
    labels = {
        "pass_at_1":             "Qwen2.5-VL-7B Pass@1",
        "llava_critic_bon":      "+ LLaVA-Critic BoN@8",
        "visualprm_bon":         "+ VisualPRM BoN@8",
        "openprm_bon_ckpt2100":  "+ OpenPRM BoN@8 (ckpt2100)",
        "openprm_bon_ckpt2658":  "+ OpenPRM BoN@8 (ckpt2658)",
    }
    print("\n" + "=" * 57)
    print("  LLaVA-Bench Evaluation Results  (judge: Qwen-VL-Max)")
    print("=" * 57)
    print(f"  {'Setting':<28} {'Score':>10}")
    print("-" * 57)
    for key, label in labels.items():
        score = results.get(key)
        s = f"{score * 100:.1f}" if score is not None else "N/A"
        print(f"  {label:<28} {s:>10}")
    print("=" * 57)

    for key, caps in breakdowns.items():
        if not caps:
            continue
        print(f"\n  Category breakdown — {labels.get(key, key)}")
        for cat, avg in caps.items():
            print(f"    {cat:<20} {avg * 100:.1f}")


def main(args):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    client = get_client()

    final_scores, breakdowns = {}, {}

    for name, (src, resp_key) in METHOD_FILES.items():
        print(f"\n--- Evaluating: {name} ---")
        scores = evaluate_method(client, name, src, resp_key)
        if scores:
            avg = sum(v["score"] for v in scores.values()) / len(scores)
            final_scores[name] = avg
            breakdowns[name] = category_breakdown(scores)
        else:
            final_scores[name] = None
            breakdowns[name] = {}

    summary = {"scores": final_scores, "breakdown": breakdowns}
    out = EVAL_DIR / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(final_scores, breakdowns)
    print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
