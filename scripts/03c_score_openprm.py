"""
Step 3c: Score candidates with OpenPRM (our trained model) for Best-of-N reranking.

Holistic scoring mode: score each candidate response as a whole.
Supports MM-Vet and LLaVA-Bench datasets.

Usage:
  python3 scripts/03c_score_openprm.py                        # MM-Vet
  python3 scripts/03c_score_openprm.py --dataset llava_bench  # LLaVA-Bench
"""

import re
import json
import argparse
from pathlib import Path
from PIL import Image

PROJECT_DIR = Path(__file__).parent.parent
MODEL_DIR   = PROJECT_DIR / "output" / "openprm-7b-merged"

DATASET_CONFIGS = {
    "mmvet": {
        "input":  PROJECT_DIR / "results" / "mmvet"       / "bon" / "candidates.json",
        "output": PROJECT_DIR / "results" / "mmvet"       / "bon" / "openprm_bon.json",
    },
    "llava_bench": {
        "input":  PROJECT_DIR / "results" / "llava_bench" / "bon" / "candidates.json",
        "output": PROJECT_DIR / "results" / "llava_bench" / "bon" / "openprm_bon.json",
    },
}

# === Training-aligned prompts ===
# System prompt matches training data exactly
SYSTEM_PROMPT = (
    "You are a process reward model. Your task is to evaluate whether a single "
    "reasoning step in a multi-step chain is correct or incorrect.\n"
    "A step is INCORRECT if it contains hallucinations, reasoning errors, "
    "unsupported claims, or contradicts earlier steps.\n"
    "Answer with exactly one word: correct or incorrect."
)

# Vision token comes from template; user content starts with \n to match
# training format where <image>\n precedes the question text.
QWEN_CHAT_TMPL = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def split_steps(text: str) -> list[str]:
    """Split a response into reasoning steps."""
    # Try explicit "Step N:" pattern
    parts = re.split(r"Step\s+\d+\s*:", text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        return parts[:5]
    # Fallback: paragraph split
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(parts) >= 2:
        return parts[:5]
    # Final fallback: treat whole response as one step
    return [text.strip()] if text.strip() else [""]


def build_step_prompt(question: str, steps: list[str], step_idx: int) -> str:
    """Build user content matching training format exactly.

    Training format (after <image>\n):
        Question: {q}\n\nPrevious reasoning steps:\nStep 1: {s1}\n...\n\n
        Step to evaluate (Step N):\n{sN}\n\n
        Is Step N correct or incorrect? Answer with exactly one word: correct or incorrect.
    """
    lines = [f"\nQuestion: {question}", ""]  # leading \n matches <image>\n in training
    if step_idx > 0:
        lines.append("Previous reasoning steps:")
        for i in range(step_idx):
            lines.append(f"Step {i + 1}: {steps[i]}")
        lines.append("")
    lines.append(f"Step to evaluate (Step {step_idx + 1}):")
    lines.append(steps[step_idx])
    lines.append("")
    lines.append(
        f"Is Step {step_idx + 1} correct or incorrect? "
        "Answer with exactly one word: correct or incorrect."
    )
    return "\n".join(lines)


def score_candidates(llm, sampling_params, qid: str, item: dict) -> dict:
    """Score each candidate by averaging per-step scores (PRM-style)."""
    image_path = item["image_path"]
    question   = item["question"]
    candidates = item["candidates"]

    try:
        pil_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  WARNING [{qid}]: cannot open image: {e}")
        pil_image = None

    # Build one prompt per (candidate, step) pair
    prompts = []
    meta = []  # (cand_idx, step_count) for reconstructing per-candidate scores
    for cand_idx, cand in enumerate(candidates):
        steps = split_steps(cand)
        for step_idx in range(len(steps)):
            user_text = build_step_prompt(question, steps, step_idx)
            prompt_text = QWEN_CHAT_TMPL.format(system=SYSTEM_PROMPT, user=user_text)
            entry = {"prompt": prompt_text}
            if pil_image is not None:
                entry["multi_modal_data"] = {"image": pil_image}
            prompts.append(entry)
        meta.append(len(steps))  # how many prompts this candidate contributed

    outputs = llm.generate(prompts, sampling_params)

    # Reconstruct per-candidate average scores
    candidate_scores = []
    out_idx = 0
    for n_steps in meta:
        step_scores = []
        for j in range(n_steps):
            text = outputs[out_idx].outputs[0].text.strip().lower()
            step_scores.append(1.0 if text.startswith("correct") else 0.0)
            out_idx += 1
        avg = sum(step_scores) / len(step_scores) if step_scores else 0.0
        candidate_scores.append(avg)

    best_idx = int(max(range(len(candidate_scores)),
                       key=lambda i: candidate_scores[i]))
    return {
        "question_id":   qid,
        "image_path":    image_path,
        "question":      question,
        "answer":        item.get("answer", item.get("reference", "")),
        "capability":    item.get("capability", item.get("category", [])),
        "scores":        candidate_scores,
        "best_idx":      best_idx,
        "best_response": candidates[best_idx],
        "pass_at_1":     item["pass_at_1"],
    }


def main(args):
    cfg = DATASET_CONFIGS[args.dataset]
    bon_dir = cfg["input"].parent
    input_file  = bon_dir / args.input  if getattr(args, "input",  None) else cfg["input"]
    output_file = bon_dir / args.output if getattr(args, "output", None) else cfg["output"]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"ERROR: {input_file} not found.")
        return

    with open(input_file) as f:
        candidates_data = json.load(f)
    print(f"Loaded {len(candidates_data)} questions from {input_file}")

    # Checkpoint
    existing = {}
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
        print(f"Resuming: {len(existing)} already scored.")

    todo = {qid: item for qid, item in candidates_data.items() if qid not in existing}
    if not todo:
        print("All done.")
        return

    if getattr(args, "model_dir", None):
        model_path = str(PROJECT_DIR / args.model_dir)
    else:
        model_path = str(MODEL_DIR)
        if not MODEL_DIR.exists():
            lora_dir = PROJECT_DIR / "output" / "openprm-7b-lora"
            checkpoints = sorted(
                lora_dir.glob("v*/checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1])
            )
            if checkpoints:
                model_path = str(checkpoints[-1])
                print(f"Merged model not found; using checkpoint: {model_path}")
            else:
                print("ERROR: OpenPRM model not found.")
                return
    print(f"Loading OpenPRM from: {model_path}")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": 28 * 28 * 4,
            "max_pixels": 28 * 28 * 512,
        },
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=5)

    results = dict(existing)
    items = list(todo.items())
    total = len(candidates_data)

    for i, (qid, item) in enumerate(items, 1):
        result = score_candidates(llm, sampling_params, qid, item)
        results[qid] = result

        if i % 10 == 0 or i == len(items):
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [{len(results)}/{total}] saved")

    print(f"\nDone. {len(results)} questions → {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mmvet", "llava_bench"], default="mmvet")
    parser.add_argument("--input",  type=str, default=None,
                        help="Input filename inside bon/ dir (overrides default)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename inside bon/ dir (overrides default)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Override model directory (relative to project root)")
    args = parser.parse_args()
    main(args)
