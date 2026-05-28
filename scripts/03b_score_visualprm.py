"""
Step 3b: Score 8 candidates per MM-Vet question with VisualPRM-8B.
Uses the model's built-in select_best_response() which calls generate_steps_with_soft_score().
Steps are split by '\n\n' (default InternVL separator).
Aggregation: mean of per-step soft scores (P('+') token).
Saves to results/visualprm_bon.json. Supports checkpointing.
"""

import sys
import json
import math
import torch
import argparse
from pathlib import Path
from PIL import Image

PROJECT_DIR = Path(__file__).parent.parent
MODEL_DIR = PROJECT_DIR / "models" / "VisualPRM-8B"

DATASET_DEFAULTS = {
    "mmvet": {
        "input":  PROJECT_DIR / "results" / "mmvet"       / "bon" / "candidates.json",
        "output": PROJECT_DIR / "results" / "mmvet"       / "bon" / "visualprm_bon.json",
    },
    "llava_bench": {
        "input":  PROJECT_DIR / "results" / "llava_bench" / "bon" / "candidates.json",
        "output": PROJECT_DIR / "results" / "llava_bench" / "bon" / "visualprm_bon.json",
    },
}
# Legacy names kept for compatibility (overridden in main)
RESULTS_DIR = PROJECT_DIR / "results" / "mmvet"
INPUT_FILE  = DATASET_DEFAULTS["mmvet"]["input"]
OUTPUT_FILE = DATASET_DEFAULTS["mmvet"]["output"]

# Add model directory to path for trust_remote_code imports
sys.path.insert(0, str(MODEL_DIR))


def build_transform(input_size: int):
    """Standard InternVL image normalization."""
    from torchvision import transforms
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    return transforms.Compose([
        transforms.Resize((input_size, input_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 12,
                       image_size: int = 448, use_thumbnail: bool = False):
    """Tile image into patches for InternVL dynamic resolution."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % target_aspect_ratio[0]) * image_size,
            (i // target_aspect_ratio[0]) * image_size,
            ((i % target_aspect_ratio[0]) + 1) * image_size,
            ((i // target_aspect_ratio[0]) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image_to_tensor(image: Image.Image, input_size: int = 448, max_num: int = 6) -> torch.Tensor:
    """Convert PIL Image to pixel_values tensor for VisualPRM."""
    image = image.convert("RGB")
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values


def load_model(model_path: str):
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading VisualPRM from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def score_candidates(model, tokenizer, image: Image.Image,
                     question: str, candidates: list[str]) -> list[float]:
    """
    Use VisualPRM's select_best_response() to get per-candidate aggregate scores.
    Returns list of mean step scores in original candidate order.
    """
    pixel_values = load_image_to_tensor(image).to(model.device).to(torch.bfloat16)

    # select_best_response with return_scores=True returns sorted list of (response, score)
    sorted_results = model.select_best_response(
        tokenizer=tokenizer,
        question=question,
        response_list=candidates,
        pixel_values=pixel_values,
        max_steps=12,
        gather_func=lambda x: sum(x) / len(x),  # mean aggregation
        return_scores=True,
    )

    # Map back to original order
    score_map = {resp: score for resp, score in sorted_results}
    return [score_map.get(c, 0.0) for c in candidates]


def main(args):
    dset = args.dataset if hasattr(args, "dataset") and args.dataset else "mmvet"
    defaults = DATASET_DEFAULTS[dset]
    bon_dir = defaults["input"].parent
    input_file  = bon_dir / args.input  if getattr(args, "input",  None) else defaults["input"]
    output_file = bon_dir / args.output if getattr(args, "output", None) else defaults["output"]
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"ERROR: {input_file} not found.")
        return
    with open(input_file) as f:
        candidates_data = json.load(f)
    print(f"Loaded {len(candidates_data)} questions from {input_file}")

    existing = {}
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
        print(f"Resuming: {len(existing)} already scored.")

    todo = {qid: item for qid, item in candidates_data.items() if qid not in existing}
    if not todo:
        print("All done.")
        return

    model_path = str(MODEL_DIR) if MODEL_DIR.exists() else "OpenGVLab/VisualPRM-8B"
    model, tokenizer = load_model(model_path)

    results = dict(existing)
    total = len(todo)

    for i, (qid, item) in enumerate(todo.items()):
        image = Image.open(item["image_path"]).convert("RGB")
        question = item["question"]
        candidates = item["candidates"]

        try:
            agg_scores = score_candidates(model, tokenizer, image, question, candidates)
        except Exception as e:
            print(f"  WARNING [{qid}]: scoring failed ({e}), using zeros.")
            agg_scores = [0.0] * len(candidates)

        best_idx = int(max(range(len(agg_scores)), key=lambda j: agg_scores[j]))
        results[qid] = {
            "question_id": qid,
            "image_path": item["image_path"],
            "question": question,
            "answer": item.get("answer", item.get("reference", "")),
            "capability": item.get("capability", item.get("category", [])),
            "agg_scores": agg_scores,
            "best_idx": best_idx,
            "best_response": candidates[best_idx],
            "pass_at_1": item["pass_at_1"],
        }

        if (i + 1) % 10 == 0 or (i + 1) == total:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(
                f"  [{i+1}/{total}] {qid}: "
                f"scores={[f'{s:.3f}' for s in agg_scores]}, best={best_idx}"
            )

    print(f"\nDone. {len(results)} questions saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mmvet", "llava_bench"], default="mmvet")
    parser.add_argument("--input",  type=str, default=None,
                        help="Input filename inside bon/ dir (overrides default)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename inside bon/ dir (overrides default)")
    args = parser.parse_args()
    main(args)
