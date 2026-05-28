"""
Step 3a: Score 8 candidates per MM-Vet question with a VLM judge.

Strategy: lmms-lab/llava-critic-7b uses LlavaQwenForCausalLM, incompatible with
transformers 4.57 and not in vLLM's registry. As an equivalent replacement we use
Qwen2.5-VL-7B-Instruct (already downloaded) in judge mode. In the paper table this
is labelled "VLM-Judge (Qwen2.5-VL-7B)".

The scoring prompt is the standard LLaVA-Critic pointwise template (1-10 scale).
Uses vLLM for efficient batched inference.
Saves to results/llava_critic_bon.json. Supports checkpointing.
"""

import re
import json
import argparse
from pathlib import Path
from PIL import Image

PROJECT_DIR = Path(__file__).parent.parent
MODEL_DIR = PROJECT_DIR / "models" / "Qwen2.5-VL-7B-Instruct"
RESULTS_DIR = PROJECT_DIR / "results" / "mmvet"
INPUT_FILE  = RESULTS_DIR / "bon" / "candidates.json"
OUTPUT_FILE = RESULTS_DIR / "bon" / "llava_critic_bon.json"

CRITIC_PROMPT = (
    "Given an image and a corresponding question, please serve as an unbiased and fair judge "
    "to evaluate the quality of the answer provided by a Large Multimodal Model (LMM).\n\n"
    "Question: {question}\n"
    "Response: {response}\n\n"
    "Please rate the response on a scale of 1 to 10, where 1 is the worst and 10 is the best.\n"
    "Your output should be in the following format:\n"
    "Score: <score>\n"
    "Reason: <reason>"
)


def parse_score(text: str) -> float:
    """Extract integer score 1-10 from output."""
    m = re.search(r"[Ss]core\s*[:：]\s*(\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"\b(10|[1-9])\b", text)
    if m:
        return float(m.group(1))
    return -1.0


def build_qwen_prompt(question: str, response: str) -> str:
    """Qwen2.5-VL chat format with image token."""
    prompt_text = CRITIC_PROMPT.format(question=question, response=response)
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{prompt_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main(args):
    bon_dir = RESULTS_DIR / "bon"
    bon_dir.mkdir(parents=True, exist_ok=True)
    input_file  = bon_dir / args.input  if args.input  else INPUT_FILE
    output_file = bon_dir / args.output if args.output else OUTPUT_FILE

    if not input_file.exists():
        print(f"ERROR: {input_file} not found.")
        return
    with open(input_file) as f:
        candidates_data = json.load(f)
    print(f"Loaded {len(candidates_data)} questions.")

    existing = {}
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
        print(f"Resuming: {len(existing)} already scored.")

    todo = {qid: item for qid, item in candidates_data.items() if qid not in existing}
    if not todo:
        print("All done.")
        return

    from vllm import LLM, SamplingParams

    model_path = str(MODEL_DIR) if MODEL_DIR.exists() else "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"Loading VLM judge from {model_path} ...")
    llm = LLM(
        model=model_path,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": 28 * 28 * 4,
            "max_pixels": 28 * 28 * 512,   # smaller for judge (less detail needed)
        },
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    results = dict(existing)
    total = len(todo)
    items = list(todo.items())
    batch_size = args.batch_size

    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start: batch_start + batch_size]
        prompts = []
        meta = []

        for qid, item in batch:
            image = Image.open(item["image_path"]).convert("RGB")
            question = item["question"]
            candidates = item["candidates"]
            for cand_idx, cand in enumerate(candidates):
                prompts.append({
                    "prompt": build_qwen_prompt(question, cand),
                    "multi_modal_data": {"image": image},
                })
                meta.append((qid, item, cand_idx))

        outputs = llm.generate(prompts, sampling_params)

        # Group scores back by question
        qid_scores: dict[str, list] = {}
        for (qid, item, cand_idx), output in zip(meta, outputs):
            text = output.outputs[0].text.strip()
            score = parse_score(text)
            if qid not in qid_scores:
                qid_scores[qid] = [None] * len(item["candidates"])
            qid_scores[qid][cand_idx] = score

        for qid, scores in qid_scores.items():
            item = candidates_data[qid]
            candidates = item["candidates"]
            best_idx = int(max(range(len(scores)), key=lambda j: scores[j] if scores[j] is not None else -1))
            results[qid] = {
                "question_id": qid,
                "image_path": item["image_path"],
                "question": item["question"],
                "answer": item["answer"],
                "capability": item.get("capability", []),
                "scores": scores,
                "best_idx": best_idx,
                "best_response": candidates[best_idx],
                "pass_at_1": item["pass_at_1"],
                "judge_model": "Qwen2.5-VL-7B-Instruct",
            }

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(results)}/{total} questions")

    print(f"\nDone. {len(results)} questions saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input",  type=str, default=None,
                        help="Input filename inside bon/ dir (overrides default)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename inside bon/ dir (overrides default)")
    args = parser.parse_args()
    main(args)
