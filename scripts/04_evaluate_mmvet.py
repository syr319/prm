"""
Step 4: Evaluate MM-Vet results using Qwen-VL-Max API (OpenAI-compatible).
Scores three response sets: Pass@1, LLaVA-Critic BoN@8, VisualPRM BoN@8.
API key read from DASHSCOPE_API_KEY environment variable.
Saves per-question scores and prints final summary table.
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from openai import OpenAI

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "mm-vet"
RESULTS_DIR = PROJECT_DIR / "results" / "mmvet"
BON_DIR  = RESULTS_DIR / "bon"
EVAL_DIR = RESULTS_DIR / "eval"

# Output files for each evaluation run
EVAL_FILES = {
    "pass_at_1":        EVAL_DIR / "eval_pass_at_1.json",
    "llava_critic_bon": EVAL_DIR / "eval_llava_critic_bon.json",
    "visualprm_bon":    EVAL_DIR / "eval_visualprm_bon.json",
}

# Official MM-Vet evaluation prompt (text-only judge, consistent with original paper)
MMVET_EVAL_PROMPT = """\
Compare the ground truth and prediction from AI models, to give a correctness score \
for the prediction. The question is: {question}. \
The ground truth is: {ground_truth}. \
The prediction is: {prediction}. \
There are several types of questions including Recognition, OCR, Math, Knowledge, \
Language Generation and Spatial Awareness. \
Note: the ground truth may contain multiple acceptable answers separated by <AND>; \
the prediction is correct if it matches any one of them. \
Give a score from 0 (totally wrong) to 1 (fully correct). \
Do not provide any other output text or explanation. \
Just provide the correctness score between 0 and 1 only.\
"""

RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # seconds between retries


def get_client() -> OpenAI:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY environment variable not set.")
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def parse_score(text: str) -> float:
    """Extract a float in [0, 1] from the API response."""
    text = text.strip()
    try:
        val = float(text)
        return max(0.0, min(1.0, val))
    except ValueError:
        pass
    m = re.search(r"([01](?:\.\d+)?|\d+\.\d+)", text)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    return 0.0


def score_one(client: OpenAI, question: str, ground_truth: str, prediction: str) -> float:
    """Call Qwen-VL-Max to score a single prediction. Text-only (no image needed)."""
    prompt = MMVET_EVAL_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction,
    )
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model="qwen-vl-max",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            return parse_score(raw)
        except Exception as e:
            print(f"  API error (attempt {attempt+1}/{RETRY_ATTEMPTS}): {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
    return 0.0


def evaluate_set(client: OpenAI, name: str, source_file: Path, response_key: str, output_file: Path):
    """Evaluate one response set (pass@1 or BoN) and save scores."""
    if not source_file.exists():
        print(f"  SKIP {name}: {source_file} not found.")
        return None

    with open(source_file) as f:
        data = json.load(f)

    # Load checkpoint
    existing = {}
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
        print(f"  Resuming {name}: {len(existing)} already scored.")

    scores = dict(existing)
    todo = {qid: item for qid, item in data.items() if qid not in scores}
    total = len(data)

    for i, (qid, item) in enumerate(todo.items()):
        prediction = item.get(response_key, "")
        ground_truth = item.get("answer", "")
        question = item.get("question", "")
        score = score_one(client, question, ground_truth, prediction)
        scores[qid] = {
            "question_id": qid,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "capability": item.get("capability", []),
            "score": score,
        }
        if (i + 1) % 20 == 0 or (i + 1) == len(todo):
            with open(output_file, "w") as f:
                json.dump(scores, f, indent=2, ensure_ascii=False)
            done = len(scores)
            print(f"  [{done}/{total}] {name}: running avg={sum(v['score'] for v in scores.values())/done:.4f}")

    # Final save
    with open(output_file, "w") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    avg = sum(v["score"] for v in scores.values()) / len(scores) if scores else 0.0
    return avg


def capability_breakdown(scores: dict) -> dict[str, float]:
    """Compute per-capability average score."""
    cap_scores: dict[str, list] = {}
    for item in scores.values():
        caps = item.get("capability", [])
        if isinstance(caps, str):
            caps = [caps]
        for cap in caps:
            cap_scores.setdefault(cap, []).append(item["score"])
    return {cap: sum(v) / len(v) for cap, v in sorted(cap_scores.items())}


def print_summary(results: dict[str, float | None], breakdown: dict[str, dict]):
    print("\n" + "=" * 55)
    print("  MM-Vet Evaluation Results")
    print("=" * 55)
    header = f"{'Setting':<28} {'MM-Vet Score':>12}"
    print(header)
    print("-" * 55)
    labels = {
        "pass_at_1":             "Qwen2.5-VL-7B Pass@1",
        "llava_critic_bon":      "+ LLaVA-Critic BoN@8",
        "visualprm_bon":         "+ VisualPRM BoN@8",
        "openprm_bon":           "+ OpenPRM BoN@8 (ckpt2100, holistic)",
        "openprm_bon_ckpt2658":  "+ OpenPRM BoN@8 (ckpt2658, holistic)",
        "openprm_bon_average":   "+ OpenPRM BoN@8 (ckpt2100, step-avg)",
    }
    for key, label in labels.items():
        score = results.get(key)
        score_str = f"{score * 100:.1f}" if score is not None else "N/A"
        print(f"  {label:<26} {score_str:>10}")
    print("=" * 55)

    # Per-capability breakdown if available
    for key, caps in breakdown.items():
        if not caps:
            continue
        label = labels.get(key, key)
        print(f"\n  Capability breakdown — {label}")
        for cap, avg in caps.items():
            print(f"    {cap:<20} {avg * 100:.1f}")


def main(args):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    client = get_client()

    sfx = f"_{args.suffix}" if args.suffix else ""

    # Map: (source_file, response_key, output_file)
    eval_configs = {
        "pass_at_1":             (BON_DIR / f"candidates{sfx}.json",             "pass_at_1",    EVAL_DIR / f"eval_pass_at_1{sfx}.json"),
        "llava_critic_bon":      (BON_DIR / f"llava_critic_bon{sfx}.json",        "best_response", EVAL_DIR / f"eval_llava_critic_bon{sfx}.json"),
        "visualprm_bon":         (BON_DIR / f"visualprm_bon{sfx}.json",           "best_response", EVAL_DIR / f"eval_visualprm_bon{sfx}.json"),
        "openprm_bon":           (BON_DIR / f"openprm_bon{sfx}.json",             "best_response", EVAL_DIR / f"eval_openprm_bon{sfx}.json"),
        "openprm_bon_ckpt2658":  (BON_DIR / f"openprm_bon_ckpt2658{sfx}.json",   "best_response", EVAL_DIR / f"eval_openprm_bon_ckpt2658{sfx}.json"),
        "openprm_bon_average":   (BON_DIR / f"openprm_bon_average{sfx}.json",    "best_response", EVAL_DIR / f"eval_openprm_bon_average{sfx}.json"),
    }

    final_scores = {}
    breakdowns = {}

    for name, (src, resp_key, out) in eval_configs.items():
        print(f"\n--- Evaluating: {name} ---")
        avg = evaluate_set(client, name, src, resp_key, out)
        final_scores[name] = avg
        if out.exists():
            with open(out) as f:
                data = json.load(f)
            breakdowns[name] = capability_breakdown(data)
        else:
            breakdowns[name] = {}

    # Save summary
    summary_file = EVAL_DIR / f"summary{sfx}.json"
    with open(summary_file, "w") as f:
        json.dump({"scores": final_scores, "breakdown": breakdowns}, f, indent=2)

    print_summary(final_scores, breakdowns)
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix appended to all input/output filenames (e.g. stepbystep)")
    args = parser.parse_args()
    main(args)
