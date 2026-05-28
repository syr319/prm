"""
Step 2: Generate 8 candidate responses per MM-Vet question using Qwen2.5-VL-7B-Instruct.
Uses vLLM for efficient batch inference. Saves results to results/candidates.json.
Supports checkpointing — safe to resume if interrupted.
"""

import os
import sys
import json
import base64
import argparse
from pathlib import Path
from PIL import Image

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "mm-vet"
MODEL_DIR = PROJECT_DIR / "models" / "Qwen2.5-VL-7B-Instruct"
RESULTS_DIR = PROJECT_DIR / "results" / "mmvet"
OUTPUT_FILE = RESULTS_DIR / "bon" / "candidates.json"

NUM_CANDIDATES = 8
TEMPERATURE = 0.8
TOP_P = 0.9
MAX_TOKENS = 1024

STEP_PROMPT_SUFFIX = (
    "\n\nPlease think step by step and structure your response as:\n"
    "Step 1: [observe the image]\n"
    "Step 2: [analyze key information]\n"
    "Step 3: [reasoning]\n"
    "Step 4: [conclusion]\n"
    "Step 5: [final answer]"
)


def load_mmvet_data(data_dir: Path):
    """Load MM-Vet questions from JSON file."""
    # MM-Vet stores data in mm-vet.json (v1) or similar
    for fname in ["mm-vet.json", "MMVet.json", "data.json"]:
        json_path = data_dir / fname
        if json_path.exists():
            with open(json_path) as f:
                return json.load(f)

    # Try looking inside subdirectories
    for json_path in data_dir.rglob("*.json"):
        with open(json_path) as f:
            try:
                data = json.load(f)
                if isinstance(data, dict) and any("question" in str(v) for v in data.values()):
                    print(f"Using {json_path}")
                    return data
            except Exception:
                continue

    raise FileNotFoundError(f"MM-Vet JSON not found in {data_dir}")


def find_image(data_dir: Path, imagename: str) -> Path:
    """Find image file, trying common subdirectory patterns."""
    candidates = [
        data_dir / "images" / imagename,
        data_dir / imagename,
        data_dir / "data" / "images" / imagename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Image not found: {imagename}")


def encode_image_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main(args):
    output_file = Path(args.output) if args.output else OUTPUT_FILE
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results for checkpointing
    existing = {}
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
        print(f"Resuming: {len(existing)} questions already done.")

    # Load MM-Vet data
    print(f"Loading MM-Vet from {DATA_DIR} ...")
    mmvet_data = load_mmvet_data(DATA_DIR)
    print(f"Loaded {len(mmvet_data)} questions.")

    # Filter questions that still need processing
    todo = {qid: item for qid, item in mmvet_data.items() if qid not in existing}
    if not todo:
        print("All questions already processed.")
        return

    print(f"Processing {len(todo)} questions with {NUM_CANDIDATES} candidates each...")

    # Initialize vLLM
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    model_path = str(MODEL_DIR) if MODEL_DIR.exists() else "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"Loading model from: {model_path}")

    llm = LLM(
        model=model_path,
        max_model_len=32768,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": 28 * 28 * 4,      # minimum 4 tiles
            "max_pixels": 28 * 28 * 1280,   # max ~1M pixels (~1280 tiles)
        },
    )

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        n=NUM_CANDIDATES,
    )

    results = dict(existing)
    batch_size = args.batch_size
    items = list(todo.items())

    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start: batch_start + batch_size]
        prompts = []
        batch_meta = []

        for qid, item in batch:
            question = item["question"]
            imagename = item.get("imagename", item.get("image", ""))
            try:
                image_path = find_image(DATA_DIR, imagename)
            except FileNotFoundError as e:
                print(f"WARNING: {e}, skipping {qid}")
                continue

            # Load image as PIL object (required by vLLM for Qwen2.5-VL)
            pil_image = Image.open(image_path).convert("RGB")

            # Qwen2.5-VL chat format
            q_text = question + (STEP_PROMPT_SUFFIX if args.stepbystep else "")
            prompt = {
                "prompt": (
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                    f"{q_text}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                ),
                "multi_modal_data": {"image": pil_image},
            }
            prompts.append(prompt)
            batch_meta.append((qid, item, imagename))

        if not prompts:
            continue

        outputs = llm.generate(prompts, sampling_params)

        for (qid, item, imagename), output in zip(batch_meta, outputs):
            candidates = [o.text.strip() for o in output.outputs]
            results[qid] = {
                "question_id": qid,
                "image_path": str(find_image(DATA_DIR, imagename)),
                "question": item["question"],
                "answer": item.get("answer", item.get("groundtruth", "")),
                "capability": item.get("capability", []),
                "candidates": candidates,
                "pass_at_1": candidates[0],  # greedy-equivalent (first sample)
            }

        # Save checkpoint after each batch
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(results)}/{len(mmvet_data)} questions to {output_file}")

    print(f"\nDone. {len(results)} questions saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Number of questions per vLLM batch")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output file path")
    parser.add_argument("--stepbystep", action="store_true",
                        help="Append step-by-step reasoning prompt to each question")
    args = parser.parse_args()
    main(args)
