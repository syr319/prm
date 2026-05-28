"""
Step 1: Parse GenPRM-MATH-Data and analyze the dataset.

Data format:
  - Parquet file with one column: 'conversations'
  - Each conversation: [system, user, assistant, user, assistant, ...]
  - First user message: "Question: <question_text>\n\n<step1_text>"
  - Subsequent user messages: just the step text
  - Each assistant message contains:
      <analyze>...</analyze>  -- verification reasoning (CoT)
      <verify>...</verify>    -- Python code + code output
      <output>...\boxed{Yes/No}...</output>  -- hard label
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "GenPRM-MATH-Data" / "data" / "train-00000-of-00001.parquet"
OUTPUT_PATH = ROOT / "data" / "genprm_math_steps.json"

# ─── Regex patterns ───────────────────────────────────────────────────────────
RE_ANALYZE = re.compile(r"<analyze>(.*?)</analyze>", re.DOTALL)
RE_VERIFY  = re.compile(r"<verify>(.*?)</verify>",  re.DOTALL)
RE_OUTPUT  = re.compile(r"<output>(.*?)</output>",  re.DOTALL)
RE_LABEL   = re.compile(r"\\boxed\{(Yes|No)\}", re.IGNORECASE)
RE_CODE    = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)


def extract_question_and_step(first_user_content: str):
    """Split first user message into (question, step1_text).

    Format: "Question: <question>\n\n<step_text>"
    """
    if first_user_content.startswith("Question:"):
        # Split on first double-newline after the question prefix
        parts = first_user_content.split("\n\n", 1)
        question = parts[0].removeprefix("Question:").strip()
        step_text = parts[1].strip() if len(parts) > 1 else ""
    else:
        question = ""
        step_text = first_user_content.strip()
    return question, step_text


def parse_assistant(content: str):
    """Extract verification_cot, verification_code, and hard_label."""
    analyze_m = RE_ANALYZE.search(content)
    verify_m  = RE_VERIFY.search(content)
    output_m  = RE_OUTPUT.search(content)

    verification_cot  = analyze_m.group(1).strip() if analyze_m else ""
    verification_code = ""
    if verify_m:
        verify_text = verify_m.group(1)
        code_blocks = RE_CODE.findall(verify_text)
        verification_code = "\n\n".join(b.strip() for b in code_blocks)

    hard_label = -1
    if output_m:
        label_m = RE_LABEL.search(output_m.group(1))
        if label_m:
            hard_label = 1 if label_m.group(1).lower() == "yes" else 0

    return verification_cot, verification_code, hard_label


def parse_conversation(conv):
    """Parse a single conversation into a list of step dicts."""
    # conv[0] = system, conv[1:] = alternating user/assistant
    messages = conv[1:]  # skip system

    steps = []
    question = ""

    # Pair up (user, assistant) turns
    for i in range(0, len(messages) - 1, 2):
        user_msg      = messages[i]
        assistant_msg = messages[i + 1]

        if user_msg["role"] != "user" or assistant_msg["role"] != "assistant":
            continue  # malformed turn

        step_index = i // 2  # 0-based

        if step_index == 0:
            question, current_step = extract_question_and_step(user_msg["content"])
        else:
            current_step = user_msg["content"].strip()

        # Build context: all prior steps (user messages before this one)
        context_parts = []
        for j in range(0, i, 2):
            prev_user = messages[j]["content"].strip()
            if j == 0:
                # Remove "Question: ...\n\n" prefix to get just the step text
                _, step_only = extract_question_and_step(prev_user)
                context_parts.append(step_only)
            else:
                context_parts.append(prev_user)
        context = "\n\n".join(context_parts)

        verification_cot, verification_code, hard_label = parse_assistant(
            assistant_msg["content"]
        )

        steps.append({
            "question":          question,
            "context":           context,
            "current_step":      current_step,
            "verification_cot":  verification_cot,
            "verification_code": verification_code,
            "hard_label":        hard_label,
            "step_index":        step_index,   # 0-based
            "total_steps":       -1,           # filled after all steps are collected
        })

    return steps


def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} conversations.\n")

    all_steps = []
    for i, row in df.iterrows():
        conv = row["conversations"]
        steps = parse_conversation(conv)
        total = len(steps)
        for s in steps:
            s["total_steps"] = total
        all_steps.extend(steps)

    # ─── Statistics ──────────────────────────────────────────────────────────
    n_convs  = len(df)
    n_steps  = len(all_steps)
    n_valid  = sum(1 for s in all_steps if s["hard_label"] != -1)
    n_correct = sum(1 for s in all_steps if s["hard_label"] == 1)
    n_wrong   = sum(1 for s in all_steps if s["hard_label"] == 0)
    n_unknown = sum(1 for s in all_steps if s["hard_label"] == -1)

    steps_per_q = [s["total_steps"] for s in all_steps if s["step_index"] == 0]

    # Error position distribution (among wrong steps)
    wrong_positions = defaultdict(int)
    for s in all_steps:
        if s["hard_label"] == 0:
            wrong_positions[s["step_index"]] += 1

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total conversations :  {n_convs:,}")
    print(f"Total step records  :  {n_steps:,}")
    print(f"  Labeled (Yes/No)  :  {n_valid:,}")
    print(f"  Unknown label     :  {n_unknown:,}")
    print()
    print(f"Correct steps (Yes) :  {n_correct:,}  ({n_correct/n_valid*100:.1f}% of labeled)")
    print(f"Wrong steps   (No)  :  {n_wrong:,}   ({n_wrong/n_valid*100:.1f}% of labeled)")
    print()
    print(f"Steps per question  :")
    steps_arr = np.array(steps_per_q)
    print(f"  min={steps_arr.min()}  max={steps_arr.max()}  "
          f"mean={steps_arr.mean():.2f}  median={np.median(steps_arr):.1f}")

    print()
    print("Error step position distribution (step_index, 0-based):")
    sorted_pos = sorted(wrong_positions.items())
    # Show first 20 positions
    for pos, cnt in sorted_pos[:20]:
        bar = "#" * (cnt * 40 // max(wrong_positions.values()))
        print(f"  step {pos:2d}: {cnt:5d}  {bar}")
    if len(sorted_pos) > 20:
        print(f"  ... ({len(sorted_pos) - 20} more positions)")

    # CoT length stats
    cot_lengths_correct = [len(s["verification_cot"]) for s in all_steps if s["hard_label"] == 1]
    cot_lengths_wrong   = [len(s["verification_cot"]) for s in all_steps if s["hard_label"] == 0]
    print()
    print("Verification CoT length (chars):")
    print(f"  Correct steps: mean={np.mean(cot_lengths_correct):.0f}  "
          f"median={np.median(cot_lengths_correct):.0f}  "
          f"p90={np.percentile(cot_lengths_correct, 90):.0f}")
    print(f"  Wrong steps:   mean={np.mean(cot_lengths_wrong):.0f}  "
          f"median={np.median(cot_lengths_wrong):.0f}  "
          f"p90={np.percentile(cot_lengths_wrong, 90):.0f}")

    # ─── Save ────────────────────────────────────────────────────────────────
    print(f"\nSaving {n_steps:,} step records to {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_steps, f, ensure_ascii=False, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
