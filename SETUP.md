# Environment Setup

## Hardware

- 8× NVIDIA H20 (or A100 80G) for 7B training
- 1× GPU (≥24 GB VRAM) for 1.5B training and evaluation
- CUDA 12.6+, Driver ≥ 560

## 1. Python Environment

```bash
conda create -n prm python=3.12 -y
conda activate prm

# PyTorch (CUDA 12.6)
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Flash-Attention (required for training)
pip install flash-attn --no-build-isolation

# Everything else
pip install -r requirements.txt
```

## 2. Download Backbone Models

```bash
# Set mirror if HuggingFace is slow
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p models

# Qwen2.5-Math-1.5B (backbone for 1.5B PRM)
huggingface-cli download Qwen/Qwen2.5-Math-1.5B \
    --local-dir models/Qwen2.5-Math-1.5B

# Qwen2.5-Math-7B (backbone for 7B PRM)
huggingface-cli download Qwen/Qwen2.5-Math-7B \
    --local-dir models/Qwen2.5-Math-7B

# Qwen2.5-Math-7B-Instruct (BoN candidate generation)
huggingface-cli download Qwen/Qwen2.5-Math-7B-Instruct \
    --local-dir models/Qwen2.5-Math-7B-Instruct

# Skywork PRM (baseline comparison)
huggingface-cli download Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B \
    --local-dir models/Skywork-PRM-1.5B
```

## 3. Download Training Data

Training data is from the [GenPRM](https://github.com/mathllm/GenPRM) dataset
(171K step-level annotations on MATH problems).

```bash
# Option A: HuggingFace datasets
python3 -c "
from datasets import load_dataset
import json, os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
ds = load_dataset('xzymustbexzy/GenPRM-MATH-Data', split='train')
# convert to the expected format — see distill/step1_analyze_data.py
"

# Option B: direct download of the preprocessed file
# data/genprm_math_steps_final.json  (~300 MB)
# Contact the repo owner or run distill/step3_generate_soft_scores.py to rebuild it.
```

MATH-500 candidates for BoN evaluation:
```bash
# Auto-downloaded on first run of bon_generate.py
# or manually cache from HuggingFace: HuggingFaceH4/MATH-500
```

## 4. Training Pipeline

### Stage 1 — Train DistillPRM-1.5B

```bash
bash distill/run_adaptive_temp.sh        # adaptive_t3 loss, 1.5B backbone
```

### Stage 2 — Train DistillPRM-7B

```bash
bash distill/run_7B_train.sh             # 8 GPUs, ~4 hours on H20
```

### Stage 3 — Iterative Distillation (iter2)

```bash
# Mine hard steps from current 7B model
python3 distill/mine_hard_steps.py \
    --checkpoint   models/DistillPRM-7B/adaptive_t3/best_model.pt \
    --student_model models/Qwen2.5-Math-7B \
    --output        data/hard_steps_from_7b.json

# Re-train with upweighted hard steps
bash distill/run_iter2_train.sh
```

## 5. Evaluation

### ProcessBench (error detection F1)

```bash
bash distill/run_eval.sh                 # evaluates checkpoints in outputs/
```

### Best-of-N Reranking on MATH-500

```bash
# Step 1 — generate 32 candidates per problem (needs vLLM, ~30 min on H20)
CUDA_VISIBLE_DEVICES=0 python3 distill/bon_generate.py

# Step 2 — rerank with each PRM
CUDA_VISIBLE_DEVICES=0 python3 distill/bon_rerank.py \
    --prm_checkpoint models/DistillPRM-7B/adaptive_t3/best_model.pt \
    --student_model  models/Qwen2.5-Math-7B \
    --agg avg --tag DistillPRM-7B

CUDA_VISIBLE_DEVICES=0 python3 distill/bon_rerank.py \
    --prm_checkpoint models/Skywork-PRM-1.5B \
    --model_type skywork --agg avg \
    --tag Skywork-PRM-1.5B
```

Results are saved to `distill/eval_results/bon_results.json`.

## Key Results (as of 2026-05)

| Model               | ProcessBench F1 | BoN-32 Acc |
|---------------------|-----------------|------------|
| DistillPRM-1.5B     | —               | 63.4% (min) |
| DistillPRM-7B       | 61.89%          | 78.2% (min) |
| Skywork-PRM-1.5B    | —               | TBD (avg)  |

## Directory Structure

```
PRM/
├── distill/           # Main training & evaluation code
│   ├── step4_build_student_model.py   # DistillPRM model definition
│   ├── step5_train_distillpRM.py      # Training loop (1.5B / 7B)
│   ├── step6_evaluate.py             # ProcessBench evaluation
│   ├── step7_iter2_train.py          # Iterative distillation (iter2)
│   ├── mine_hard_steps.py            # Hard-step mining for iter2
│   ├── bon_generate.py               # MATH-500 candidate generation
│   ├── bon_rerank.py                 # BoN reranking evaluation
│   ├── eval_results/                 # Evaluation JSON outputs
│   └── run_*.sh                      # Launch scripts
├── data/
│   ├── math500_raw.json              # MATH-500 problems (cached)
│   ├── sanity_500.jsonl              # Sanity-check split
│   └── val_data.jsonl                # Validation split
├── configs/                          # SFT / training configs
├── models/                           # (not tracked) backbone & trained PRMs
└── outputs/                          # (not tracked) training checkpoints
```
