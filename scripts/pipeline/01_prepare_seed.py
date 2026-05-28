"""
Pipeline Step 1: Prepare seed data from LLaVA-Instruct-150K.

Filters:
  - Must have image (all entries do)
  - Answer length >= 50 chars
  - Question must contain image reference (not pure text)
  - Prefers rich open-domain questions (description, analysis, reasoning)

Then downloads only the needed COCO images from images.cocodataset.org.
Saves: data/pipeline/seed_data.json + data/pipeline/coco_images/
"""

import os
import json
import random
import hashlib
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_DIR = Path(__file__).parent.parent.parent
LLAVA_JSON = PROJECT_DIR / "data" / "llava-instruct" / "llava_instruct_150k.json"
OUT_DIR = PROJECT_DIR / "data" / "pipeline"
SEED_FILE = OUT_DIR / "seed_data.json"
IMG_DIR = OUT_DIR / "coco_images"

COCO_BASE_URL = "http://images.cocodataset.org/train2017"
TARGET_COUNT = 3000
MIN_ANSWER_LEN = 50
RANDOM_SEED = 42

# Keywords that signal richer open-domain questions
PREFERRED_KEYWORDS = [
    "describe", "what", "how", "why", "explain", "analyze", "tell me",
    "what is happening", "what do you see", "identify", "what kind",
]


def score_entry(entry: dict) -> int:
    """Higher = better seed sample. Returns 0 if should be filtered out."""
    convs = entry.get("conversations", [])
    if len(convs) < 2:
        return 0

    question = convs[0]["value"].replace("<image>", "").strip()
    answer = convs[1]["value"].strip()

    if len(answer) < MIN_ANSWER_LEN:
        return 0
    if not entry.get("image"):
        return 0
    if "<image>" not in convs[0]["value"]:  # must reference image
        return 0

    score = len(answer)  # longer answers tend to be richer
    q_lower = question.lower()
    for kw in PREFERRED_KEYWORDS:
        if kw in q_lower:
            score += 100
    return score


def download_image(image_name: str, img_dir: Path, retries: int = 3) -> bool:
    """Download a single COCO image. Returns True on success."""
    dest = img_dir / image_name
    if dest.exists():
        return True

    url = f"{COCO_BASE_URL}/{image_name}"
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"  FAILED: {image_name} — {e}")
    return False


def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Filter and sample ─────────────────────────────────────────────
    print(f"Loading LLaVA-Instruct from {LLAVA_JSON} ...")
    with open(LLAVA_JSON) as f:
        raw_data = json.load(f)
    print(f"Total entries: {len(raw_data)}")

    # Score all entries
    scored = []
    for entry in raw_data:
        s = score_entry(entry)
        if s > 0:
            scored.append((s, entry))

    print(f"Entries passing filter: {len(scored)}")
    random.seed(RANDOM_SEED)
    # Prefer higher-scored entries but add randomness
    scored.sort(key=lambda x: x[0], reverse=True)
    # Take top 15000 by score, then randomly sample from them
    pool = scored[:15000]
    random.shuffle(pool)
    selected = [entry for _, entry in pool[:TARGET_COUNT]]
    print(f"Selected {len(selected)} seed samples")

    # Build seed_data format
    seed_data = []
    for entry in selected:
        convs = entry["conversations"]
        question = convs[0]["value"].replace("<image>", "").strip()
        answer = convs[1]["value"].strip()
        seed_data.append({
            "id": entry["id"],
            "image": entry["image"],
            "image_path": str(IMG_DIR / entry["image"]),
            "question": question,
            "reference_answer": answer,
            "source": "llava_instruct_150k",
        })

    with open(SEED_FILE, "w") as f:
        json.dump(seed_data, f, indent=2, ensure_ascii=False)
    print(f"Saved seed data: {SEED_FILE}")

    # ── Step 2: Download images ───────────────────────────────────────────────
    if args.skip_download:
        print("Skipping image download (--skip-download)")
        return

    images_needed = list({s["image"] for s in seed_data})
    already = [img for img in images_needed if (IMG_DIR / img).exists()]
    to_download = [img for img in images_needed if not (IMG_DIR / img).exists()]
    print(f"Images: {len(already)} cached, {len(to_download)} to download")

    if not to_download:
        print("All images already cached.")
        return

    success = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_image, img, IMG_DIR): img for img in to_download}
        for i, future in enumerate(as_completed(futures), 1):
            ok = future.result()
            if ok:
                success += 1
            else:
                failed += 1
            if i % 200 == 0 or i == len(to_download):
                print(f"  [{i}/{len(to_download)}] downloaded={success}, failed={failed}")

    print(f"\nDone. Images: {success} ok, {failed} failed")
    print(f"Seed file: {SEED_FILE}")
    print(f"Image dir: {IMG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true", help="Skip image download")
    parser.add_argument("--workers", type=int, default=32, help="Parallel download threads")
    args = parser.parse_args()
    main(args)
