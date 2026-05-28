"""
Pipeline Step 4: Convert reasoning traces + negative samples to ms-swift sharegpt format.

For each sample (positive or negative), generates one training instance per step k:
  - Input : <image> + question + steps 1…k-1 (context) + step k (current step)
  - Output: "correct" or "incorrect"

Positive samples  → all 5 steps labeled "correct"
Negative samples  → follow step_labels[i]["label"] from 03_inject_errors.py
                    ("correct" for pre-error steps, "incorrect" for error + downstream)

Output format: ms-swift sharegpt JSONL  →  data/train_data.jsonl
               Optionally split into train/val with --val-ratio (default 0.05).

Usage:
  python3 scripts/pipeline/04_convert_to_swift.py
  python3 scripts/pipeline/04_convert_to_swift.py --val-ratio 0.1
  python3 scripts/pipeline/04_convert_to_swift.py --test-run 10   # first 10 sources only
  python3 scripts/pipeline/04_convert_to_swift.py --skip-downstream  # exclude downstream-incorrect steps
"""

import json
import random
import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR    = PROJECT_DIR / "data" / "pipeline"
POS_FILE    = DATA_DIR / "reasoning_traces.json"
NEG_FILE    = DATA_DIR / "negative_samples.json"
OUT_DIR     = PROJECT_DIR / "data"
OUT_FILE    = OUT_DIR / "train_data.jsonl"
VAL_FILE    = OUT_DIR / "val_data.jsonl"

RANDOM_SEED = 42

# ── Prompt template ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a process reward model. Your task is to evaluate whether a single "
    "reasoning step in a multi-step chain is correct or incorrect.\n"
    "A step is INCORRECT if it contains hallucinations, reasoning errors, "
    "unsupported claims, or contradicts earlier steps.\n"
    "Answer with exactly one word: correct or incorrect."
)

def build_user_content(question: str, steps: list[str], step_idx: int) -> str:
    """
    Build the user-turn text for step (step_idx, 0-based).
    The <image> token at the front is handled separately in the messages list.
    """
    lines = [f"Question: {question}", ""]

    if step_idx > 0:
        lines.append("Previous reasoning steps:")
        for i in range(step_idx):
            lines.append(f"Step {i+1}: {steps[i]}")
        lines.append("")

    lines.append(f"Step to evaluate (Step {step_idx+1}):")
    lines.append(steps[step_idx])
    lines.append("")
    lines.append(
        f"Is Step {step_idx+1} correct or incorrect? "
        "Answer with exactly one word: correct or incorrect."
    )
    return "\n".join(lines)


def make_record(image_path: str, question: str, steps: list[str],
                step_idx: int, label: str) -> dict:
    """Return one sharegpt-format dict for ms-swift."""
    user_text = build_user_content(question, steps, step_idx)
    return {
        "messages": [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": f"<image>\n{user_text}"},
            {"role": "assistant","content": label},
        ],
        "images": [image_path],
    }


def convert_positive(item: dict) -> list[dict]:
    """All 5 steps of a positive trace → label 'correct'."""
    records = []
    for i, step in enumerate(item["steps"]):
        records.append(make_record(
            image_path=item["image_path"],
            question=item["question"],
            steps=item["steps"],
            step_idx=i,
            label="correct",
        ))
    return records


def convert_negative(item: dict, skip_downstream: bool = False) -> list[dict]:
    """Steps of a negative sample → label from step_labels[i]['label']."""
    records = []
    for i, sl in enumerate(item["step_labels"]):
        if skip_downstream and sl.get("error_type") == "downstream":
            continue
        records.append(make_record(
            image_path=item["image_path"],
            question=item["question"],
            steps=item["steps"],
            step_idx=i,
            label=sl["label"],   # "correct" or "incorrect"
        ))
    return records


def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading {POS_FILE} ...")
    positives = json.load(open(POS_FILE))
    print(f"Loading {NEG_FILE} ...")
    negatives = json.load(open(NEG_FILE))

    if args.test_run:
        positives = positives[:args.test_run]
        # keep only negatives that derive from selected positives
        pos_ids = {p["id"] for p in positives}
        negatives = [n for n in negatives if n["source_id"] in pos_ids]
        print(f"TEST RUN: {len(positives)} positives, {len(negatives)} negatives")
    else:
        print(f"Positives: {len(positives)}  Negatives: {len(negatives)}")

    # ── Convert ───────────────────────────────────────────────────────────────
    all_records: list[dict] = []

    for item in positives:
        all_records.extend(convert_positive(item))

    for item in negatives:
        all_records.extend(convert_negative(item, skip_downstream=args.skip_downstream))

    # ── Stats ─────────────────────────────────────────────────────────────────
    n_correct   = sum(1 for r in all_records if r["messages"][-1]["content"] == "correct")
    n_incorrect = sum(1 for r in all_records if r["messages"][-1]["content"] == "incorrect")
    print(f"\nTotal records : {len(all_records)}")
    print(f"  correct     : {n_correct}  ({100*n_correct/len(all_records):.1f}%)")
    print(f"  incorrect   : {n_incorrect}  ({100*n_incorrect/len(all_records):.1f}%)")

    # ── Shuffle + split ───────────────────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_records)

    if args.val_ratio > 0:
        split = int(len(all_records) * (1 - args.val_ratio))
        train_records = all_records[:split]
        val_records   = all_records[split:]
    else:
        train_records = all_records
        val_records   = []

    # ── Write ─────────────────────────────────────────────────────────────────
    def write_jsonl(path: Path, records: list[dict]):
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_jsonl(OUT_FILE, train_records)
    print(f"\nTrain: {len(train_records)} records → {OUT_FILE}")

    if val_records:
        write_jsonl(VAL_FILE, val_records)
        print(f"Val  : {len(val_records)} records → {VAL_FILE}")

    # ── Sample preview ────────────────────────────────────────────────────────
    print("\n=== Format Preview (first 2 records) ===")
    for rec in train_records[:2]:
        print(f"  image : {rec['images'][0]}")
        print(f"  system: {rec['messages'][0]['content'][:60]}...")
        # show first 120 chars of user turn (skip <image> token)
        user_body = rec['messages'][1]['content'].replace('<image>\n', '', 1)
        print(f"  user  : {user_body[:120].strip()}...")
        print(f"  answer: {rec['messages'][2]['content']}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Fraction of data for validation set (0 = no split)")
    parser.add_argument("--skip-downstream", action="store_true",
                        help="Exclude downstream-incorrect steps (steps after injected error)")
    parser.add_argument("--test-run", type=int, default=0,
                        help="Only process first N positive sources (0 = full run)")
    args = parser.parse_args()
    main(args)
