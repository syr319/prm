"""
Step 2 (LLaVA-Bench): Generate 8 candidate responses per question using
Qwen2.5-VL-7B-Instruct with vLLM.

Uses a structured 5-step reasoning prompt so that OpenPRM can score each step.
Saves results to results/llava_bench_candidates.json. Supports checkpointing.
"""

import json
import argparse
from pathlib import Path
from PIL import Image

PROJECT_DIR = Path(__file__).parent.parent
DATA_FILE   = PROJECT_DIR / "data" / "llava-bench" / "questions.json"
MODEL_DIR   = PROJECT_DIR / "models" / "Qwen2.5-VL-7B-Instruct"
RESULTS_DIR = PROJECT_DIR / "results" / "llava_bench"
OUTPUT_FILE = RESULTS_DIR / "bon" / "candidates.json"

NUM_CANDIDATES = 8
TEMPERATURE    = 0.8
TOP_P          = 0.9
MAX_TOKENS     = 1024

# Structured prompt so OpenPRM can score each step
STEP_PROMPT = (
    "Answer the question about the image using a step-by-step reasoning process.\n\n"
    "Format your response as:\n"
    "Step 1: [Observe and describe the relevant visual content]\n"
    "Step 2: [Identify key information relevant to the question]\n"
    "Step 3: [Apply reasoning or domain knowledge]\n"
    "Step 4: [Draw intermediate conclusions]\n"
    "Step 5: [State the final answer clearly]\n\n"
    "Question: {question}"
)

QWEN_CHAT_TMPL = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def main(args):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found. Run dataset download first.")
        return

    with open(DATA_FILE) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} LLaVA-Bench questions.")

    # Checkpoint
    existing = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            existing = json.load(f)
        print(f"Resuming: {len(existing)} already done.")

    todo = [q for q in questions if q["id"] not in existing]
    if not todo:
        print("All done.")
        return

    print(f"Processing {len(todo)} questions ({NUM_CANDIDATES} candidates each)...")

    from vllm import LLM, SamplingParams

    model_path = str(MODEL_DIR) if MODEL_DIR.exists() else "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"Loading model: {model_path}")
    llm = LLM(
        model=model_path,
        max_model_len=32768,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": 28 * 28 * 4,
            "max_pixels": 28 * 28 * 1280,
        },
    )
    sampling_params = SamplingParams(
        temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS, n=NUM_CANDIDATES,
    )

    results = dict(existing)
    batch_size = args.batch_size

    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start: batch_start + batch_size]
        prompts, meta = [], []

        for item in batch:
            img_path = item["image_path"]
            if not Path(img_path).exists():
                print(f"  WARNING: image not found: {img_path}")
                continue
            pil_image = Image.open(img_path).convert("RGB")
            prompt_text = STEP_PROMPT.format(question=item["question"])
            prompts.append({
                "prompt": QWEN_CHAT_TMPL.format(prompt=prompt_text),
                "multi_modal_data": {"image": pil_image},
            })
            meta.append(item)

        if not prompts:
            continue

        outputs = llm.generate(prompts, sampling_params)

        for item, output in zip(meta, outputs):
            candidates = [o.text.strip() for o in output.outputs]
            results[item["id"]] = {
                "question_id": item["id"],
                "image_path":  item["image_path"],
                "question":    item["question"],
                "reference":   item["reference"],
                "category":    item["category"],
                "candidates":  candidates,
                "pass_at_1":   candidates[0],
            }

        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        done = len(results)
        print(f"  [{done}/{len(questions)}] saved to {OUTPUT_FILE}")

    print(f"\nDone. {len(results)} questions → {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    main(args)
