"""
Pipeline Step 2: Generate step-by-step reasoning chains using Qwen-VL-Max API.

For each (image, question) in seed_data.json, calls qwen-vl-max to produce a
5-step reasoning trace. Saves positive samples to data/pipeline/reasoning_traces.json.
Supports --test-run 100 to validate format before full run.
Supports checkpointing (safe to resume).
"""

import os
import re
import json
import time
import base64
import argparse
import threading
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data" / "pipeline"
SEED_FILE = DATA_DIR / "seed_data.json"
OUTPUT_FILE = DATA_DIR / "reasoning_traces.json"

MAX_RETRIES = 3
RETRY_DELAY = 3
MAX_WORKERS = 8   # parallel API calls

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

REASONING_PROMPT = """\
You are given an image and a question about it. Generate a detailed step-by-step reasoning \
trace with exactly 5 steps that leads to answering the question.

Question: {question}

Format your response EXACTLY as follows (no other text):
Step 1: [Carefully observe and describe the relevant visual content in the image]
Step 2: [Identify and analyze the key information relevant to the question]
Step 3: [Apply reasoning or domain knowledge to interpret what you observed]
Step 4: [Draw intermediate conclusions or handle any complexity]
Step 5: [State the final answer clearly and concisely]

Each step should be 1-3 sentences. Steps must be logically connected and grounded in \
what is actually visible in the image. Do not fabricate content that is not in the image.\
"""

STEP_PATTERN = re.compile(
    r"Step\s+1\s*:\s*(.+?)\s*Step\s+2\s*:\s*(.+?)\s*Step\s+3\s*:\s*(.+?)\s*"
    r"Step\s+4\s*:\s*(.+?)\s*Step\s+5\s*:\s*(.+?)(?:\s*$)",
    re.DOTALL | re.IGNORECASE,
)


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
        # All exhausted — reset and try again
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


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_steps(text: str) -> list[str] | None:
    """Extract 5 steps from model output. Returns None if format is wrong."""
    m = STEP_PATTERN.search(text)
    if m:
        return [m.group(i).strip() for i in range(1, 6)]

    # Fallback: split on "Step N:" pattern
    parts = re.split(r"Step\s+\d+\s*:", text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 5:
        return parts[:5]
    return None


def generate_one(client: OpenAI, item: dict) -> dict | None:
    """Generate reasoning trace for one item. Returns result dict or None on failure."""
    image_path = item["image_path"]
    if not Path(image_path).exists():
        return None

    b64 = image_to_base64(image_path)
    prompt = REASONING_PROMPT.format(question=item["question"])

    for attempt in range(MAX_RETRIES):
        model = get_current_model()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=512,
                temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()
            steps = parse_steps(raw)
            if steps is None:
                if attempt < MAX_RETRIES - 1:
                    continue
                # Accept with warning even if format is imperfect
                steps = [s.strip() for s in raw.split("\n") if s.strip()][:5]
                if not steps:
                    return None

            return {
                "id": item["id"],
                "image": item["image"],
                "image_path": image_path,
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "steps": steps,
                "raw_response": raw,
                "label": "positive",
            }
        except Exception as e:
            if _is_quota_error(e):
                mark_exhausted(model)
            elif attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"  ERROR [{item['id']}]: {e}")
    return None


def main(args):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load seed data
    if not SEED_FILE.exists():
        print(f"ERROR: {SEED_FILE} not found. Run 01_prepare_seed.py first.")
        return
    with open(SEED_FILE) as f:
        seed_data = json.load(f)

    # Limit to test run if requested
    if args.test_run:
        seed_data = seed_data[:args.test_run]
        print(f"TEST RUN: processing {len(seed_data)} samples")
    else:
        print(f"Full run: {len(seed_data)} samples")

    # Load checkpoint
    existing = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            existing = {r["id"]: r for r in json.load(f)}
        print(f"Resuming: {len(existing)} already done")

    todo = [item for item in seed_data if item["id"] not in existing]
    if not todo:
        print("All done.")
        return

    print(f"Processing {len(todo)} items with {MAX_WORKERS} parallel workers...")

    client = get_client()
    results = list(existing.values())
    failed = 0

    def save():
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(generate_one, client, item): item for item in todo}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                results.append(result)
            else:
                failed += 1

            if i % 20 == 0 or i == len(todo):
                save()
                print(f"  [{i}/{len(todo)}] done={len(results)}, failed={failed}")

    save()
    print(f"\nDone. {len(results)} traces saved to {OUTPUT_FILE}")

    # Print format validation for test run
    if args.test_run and results:
        print("\n=== Format Sample ===")
        sample = results[0]
        print(f"ID: {sample['id']}")
        print(f"Question: {sample['question'][:60]}")
        for j, step in enumerate(sample["steps"], 1):
            print(f"  Step {j}: {step[:80]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", type=int, default=0,
                        help="Only process N samples (0 = full run)")
    args = parser.parse_args()
    main(args)
