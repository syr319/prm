#!/bin/bash
# Step 1: Download MM-Vet dataset and all required models
# Uses hf-mirror.com for reliable access in China

set -e

export HF_ENDPOINT=https://hf-mirror.com
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"
MODEL_DIR="$PROJECT_DIR/models"

mkdir -p "$DATA_DIR" "$MODEL_DIR"

echo "=== [1/4] Downloading MM-Vet dataset (whyu/mm-vet via datasets library) ==="
python3 - <<EOF
import os, json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from pathlib import Path

out_dir = Path("$DATA_DIR/mm-vet")
img_dir = out_dir / "images"
img_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset("whyu/mm-vet", split="test")

records = {}
for item in ds:
    qid = item["id"]
    img_path = img_dir / f"{qid}.png"
    if not img_path.exists():
        item["image"].convert("RGB").save(img_path)
    records[qid] = {
        "question": item["question"],
        "answer": item["answer"],
        "capability": item["capability"],
        "imagename": f"{qid}.png",
    }

with open(out_dir / "mm-vet.json", "w") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"MM-Vet saved: {len(records)} questions, images in {img_dir}")
EOF

echo "=== [2/4] Downloading Qwen2.5-VL-7B-Instruct ==="
python3 - <<EOF
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
    local_dir="$MODEL_DIR/Qwen2.5-VL-7B-Instruct",
    ignore_patterns=["*.git*", "*.h5", "flax_model*", "tf_model*"],
)
print("Qwen2.5-VL-7B-Instruct downloaded")
EOF

echo "=== [3/4] Downloading LLaVA-Critic-7B ==="
python3 - <<EOF
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="lmms-lab/llava-critic-7b",
    local_dir="$MODEL_DIR/llava-critic-7b",
    ignore_patterns=["*.git*", "*.h5", "flax_model*", "tf_model*"],
)
print("LLaVA-Critic-7B downloaded")
EOF

echo "=== [4/4] Downloading VisualPRM-8B ==="
python3 - <<EOF
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="OpenGVLab/VisualPRM-8B",
    local_dir="$MODEL_DIR/VisualPRM-8B",
    ignore_patterns=["*.git*", "*.h5", "flax_model*", "tf_model*"],
)
print("VisualPRM-8B downloaded")
EOF

echo ""
echo "=== All downloads complete ==="
echo "Data: $DATA_DIR"
echo "Models: $MODEL_DIR"
du -sh "$DATA_DIR"/mm-vet "$MODEL_DIR"/Qwen2.5-VL-7B-Instruct "$MODEL_DIR"/llava-critic-7b "$MODEL_DIR"/VisualPRM-8B 2>/dev/null
