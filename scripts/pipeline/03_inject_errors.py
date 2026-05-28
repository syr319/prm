"""
Pipeline Step 3: Reverse error injection to create negative samples.

For each positive reasoning trace, generates 2 negative variants by injecting
one of four error types at a randomly chosen intermediate step (steps 2-4):

  1. hallucination      — fabricate non-existent visual content
  2. reasoning_gap      — skip critical reasoning, jump to conclusion
  3. inconsistency      — contradict something stated in a prior step
  4. visual_grounding   — misattribute visual attribute (color, size, position)

Uses Qwen-VL-Max API (needs image for hallucination/visual_grounding types).
Saves to data/pipeline/negative_samples.json.
Supports checkpointing.
"""

import os
import json
import time
import base64
import random
import argparse
import threading
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data" / "pipeline"
INPUT_FILE = DATA_DIR / "reasoning_traces.json"
OUTPUT_FILE = DATA_DIR / "negative_samples.json"

MAX_RETRIES = 3
RETRY_DELAY = 3
MAX_WORKERS = 8
NEGATIVES_PER_POSITIVE = 2

MODELS = [
    "qwen-vl-max",
    "qwen-vl-max-latest",
    "qwen-vl-max-2025-08-13",
    "qwen-vl-max-2025-04-08",
    "qwen-vl-max-2025-04-02",
    "qwen-vl-max-1230",
    "qwen-vl-max-1119",
    "qwen-vl-max-2025-01-25",
]

ERROR_TYPES = ["hallucination", "reasoning_gap", "inconsistency", "visual_grounding"]

# Prompt for each error type
ERROR_PROMPTS = {
    "hallucination": """\
You are given an image, a question, and a step-by-step reasoning trace. \
Your task is to create a corrupted version of the trace by modifying Step {step_num} to \
introduce a HALLUCINATION error: make the model describe a visual element (object, person, \
text, or action) that does NOT actually appear in the image. The fabricated content should \
sound plausible but be factually wrong given the image.

Keep all other steps identical. Only change Step {step_num}.

Question: {question}

Original trace:
{trace}

Output the full modified trace in the same format (Step 1 through Step 5), with Step {step_num} \
containing the hallucination. Do not add any explanation.""",

    "reasoning_gap": """\
You are given an image, a question, and a step-by-step reasoning trace. \
Your task is to create a corrupted version of the trace by modifying Step {step_num} to \
introduce a REASONING GAP error: replace the step with a hasty, unjustified conclusion that \
skips the logical reasoning that should happen at this point. The step should jump directly to \
a conclusion without showing the intermediate reasoning.

Keep all other steps identical. Only change Step {step_num}.

Question: {question}

Original trace:
{trace}

Output the full modified trace in the same format (Step 1 through Step 5). Do not add explanation.""",

    "inconsistency": """\
You are given an image, a question, and a step-by-step reasoning trace. \
Your task is to create a corrupted version of the trace by modifying Step {step_num} to \
introduce an INCONSISTENCY error: make Step {step_num} directly contradict something that was \
stated in an earlier step (e.g., claim a different number, opposite attribute, or conflicting fact).

Keep all other steps identical. Only change Step {step_num}.

Question: {question}

Original trace:
{trace}

Output the full modified trace in the same format (Step 1 through Step 5). Do not add explanation.""",

    "visual_grounding": """\
You are given an image, a question, and a step-by-step reasoning trace. \
Your task is to create a corrupted version of the trace by modifying Step {step_num} to \
introduce a VISUAL GROUNDING ERROR: incorrectly describe a visual attribute of something \
that IS in the image — change a color, size, quantity, spatial relationship, or other \
attribute to a plausible but wrong value.

Keep all other steps identical. Only change Step {step_num}.

Question: {question}

Original trace:
{trace}

Output the full modified trace in the same format (Step 1 through Step 5). Do not add explanation.""",
}


_model_lock = threading.Lock()
_exhausted: set = set()


def _is_quota_error(e: Exception) -> bool:
    msg = str(e)
    msg_lower = msg.lower()
    if any(k in msg_lower for k in ("quota", "insufficient", "rate limit", "ratelimit", "429")):
        return True
    if "403" in msg or "AllocationQuota" in msg or "FreeTierOnly" in msg:
        return True
    return False


def get_current_model() -> str:
    with _model_lock:
        for m in MODELS:
            if m not in _exhausted:
                return m
        print("  [model rotation] all models exhausted, resetting list")
        _exhausted.clear()
        return MODELS[0]


def mark_exhausted(model: str) -> str:
    with _model_lock:
        _exhausted.add(model)
        for m in MODELS:
            if m not in _exhausted:
                print(f"  [model rotation] {model!r} exhausted → switching to {m!r}")
                return m
        print("  [model rotation] all models exhausted, resetting list")
        _exhausted.clear()
        return MODELS[0]


def get_client() -> OpenAI:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY not set")
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def format_trace(steps: list[str]) -> str:
    return "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))


def parse_modified_trace(text: str, original_steps: list[str]) -> list[str] | None:
    """Extract 5 steps from modified trace. Falls back to original if parsing fails."""
    import re
    parts = re.split(r"Step\s+\d+\s*:", text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 5:
        return parts[:5]
    # Try to find individual steps
    steps = []
    for i in range(1, 6):
        m = re.search(rf"Step\s+{i}\s*:\s*(.+?)(?=Step\s+{i+1}|$)", text, re.DOTALL | re.IGNORECASE)
        if m:
            steps.append(m.group(1).strip())
        else:
            steps.append(original_steps[i-1])  # fallback to original
    return steps if len(steps) == 5 else None


def image_to_base64(image_path: str) -> str | None:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def inject_one(client: OpenAI, positive: dict, error_type: str, step_num: int) -> dict | None:
    """Inject an error into one step of a positive trace."""
    trace_str = format_trace(positive["steps"])
    prompt = ERROR_PROMPTS[error_type].format(
        step_num=step_num,
        question=positive["question"],
        trace=trace_str,
    )

    # For visual errors, include the image
    needs_image = error_type in ("hallucination", "visual_grounding")
    b64 = image_to_base64(positive["image_path"]) if needs_image else None

    for attempt in range(MAX_RETRIES):
        try:
            if b64:
                content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ]
            else:
                content = prompt

            model = get_current_model()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=512,
                temperature=0.5,
            )
            raw = resp.choices[0].message.content.strip()
            modified_steps = parse_modified_trace(raw, positive["steps"])
            if modified_steps is None:
                if attempt < MAX_RETRIES - 1:
                    continue
                return None

            # Verify that step_num actually changed
            if modified_steps[step_num - 1] == positive["steps"][step_num - 1]:
                if attempt < MAX_RETRIES - 1:
                    continue

            # Build step-level labels: the injected step and all after are wrong
            step_labels = []
            for i in range(5):
                if i + 1 == step_num:
                    step_labels.append({"step": modified_steps[i], "label": "incorrect",
                                        "error_type": error_type})
                elif i + 1 > step_num:
                    # Steps after the error are also unreliable
                    step_labels.append({"step": modified_steps[i], "label": "incorrect",
                                        "error_type": "downstream"})
                else:
                    step_labels.append({"step": modified_steps[i], "label": "correct",
                                        "error_type": None})

            return {
                "id": f"{positive['id']}_neg_{error_type}_{step_num}",
                "source_id": positive["id"],
                "image": positive["image"],
                "image_path": positive["image_path"],
                "question": positive["question"],
                "reference_answer": positive["reference_answer"],
                "steps": modified_steps,
                "step_labels": step_labels,
                "error_type": error_type,
                "error_step": step_num,
                "label": "negative",
            }
        except Exception as e:
            if _is_quota_error(e):
                mark_exhausted(model)
            elif attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"  ERROR [{positive['id']} {error_type} step{step_num}]: {e}")
    return None


def process_positive(args_tuple):
    """Worker function: generate NEGATIVES_PER_POSITIVE negatives for one positive."""
    client, positive, rng = args_tuple
    negatives = []

    # Choose NEGATIVES_PER_POSITIVE random (error_type, step_num) combos
    combos = [
        (et, sn)
        for et in ERROR_TYPES
        for sn in [2, 3, 4]  # inject only in middle steps
    ]
    chosen = rng.sample(combos, min(NEGATIVES_PER_POSITIVE, len(combos)))

    for error_type, step_num in chosen:
        result = inject_one(client, positive, error_type, step_num)
        if result:
            negatives.append(result)

    return negatives


def main(args):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found. Run 02_generate_reasoning.py first.")
        return
    with open(INPUT_FILE) as f:
        positives = json.load(f)

    if args.test_run:
        positives = positives[:args.test_run]
        print(f"TEST RUN: {len(positives)} positives → ~{len(positives)*NEGATIVES_PER_POSITIVE} negatives")
    else:
        print(f"Full run: {len(positives)} positives → ~{len(positives)*NEGATIVES_PER_POSITIVE} negatives")

    # Checkpoint
    existing_ids = set()
    existing = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            existing = json.load(f)
        existing_ids = {r["source_id"] for r in existing}
        print(f"Resuming: {len(existing_ids)} sources already done")

    todo = [p for p in positives if p["id"] not in existing_ids]
    if not todo:
        print("All done.")
        return

    client = get_client()
    results = list(existing)
    rng = random.Random(42)

    def save():
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_positive, (client, pos, rng)): pos
            for pos in todo
        }
        for i, future in enumerate(as_completed(futures), 1):
            negs = future.result()
            results.extend(negs)
            if i % 20 == 0 or i == len(todo):
                save()
                print(f"  [{i}/{len(todo)}] negatives={len(results)}")

    save()
    print(f"\nDone. {len(results)} negative samples saved to {OUTPUT_FILE}")

    # Print a sample for test run
    if args.test_run and results:
        sample = results[0]
        print("\n=== Negative Sample ===")
        print(f"Error type: {sample['error_type']} at step {sample['error_step']}")
        for j, sl in enumerate(sample["step_labels"], 1):
            label_flag = "✗" if sl["label"] == "incorrect" else "✓"
            print(f"  {label_flag} Step {j}: {sl['step'][:70]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", type=int, default=0)
    args = parser.parse_args()
    main(args)
