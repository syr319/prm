# Model Inventory

## 1. Self-Trained Models — MUST Backup

These are your own trained checkpoints. They **cannot be downloaded** anywhere.
Each `best_model.pt` bundles the full model weights + training args and is self-contained
(no separate tokenizer files needed — load with the backbone model path at inference time).

### 1.1 Core Results (backup first)

| Path | Size | Description | Key Result |
|------|------|-------------|------------|
| `models/DistillPRM-1.5B/adaptive_t3/best_model.pt` | 8.7 G | **1.5B main model** — adaptive KL+CE, temperature=3 | BoN-32: 63.4% (min-agg, degrades with N) |
| `models/DistillPRM-7B/adaptive_t3/best_model.pt` | 14 G | **7B main model** — adaptive KL+CE, temperature=3 | ProcessBench F1: ~61%, BoN-32: 78.2% |
| `outputs/distillprm-7b-instruct-adaptive-t3/adaptive_t3/best_model.pt` | 14 G | **7B Instruct fine-tuned** — continued from Qwen2.5-Math-7B-Instruct | **Best F1: 61.89%** on ProcessBench |
| `outputs/distillprm-7b-iter2/best_model.pt` | 14 G | **7B iter2** — re-trained with top-25% hard steps (weight=3×) | Latest result |
| `outputs/distillprm-7b-iter2-combined/best_model.pt` | 14 G | **7B iter2-combined** — hard steps from both base and instruct models | Latest result |

**Recommended minimum backup: the 5 rows above (≈ 64 G total).**

### 1.2 Ablation Checkpoints (backup if submitting to a conference)

These are needed to reproduce the ablation table in the paper.

| Path | Size | Description |
|------|------|-------------|
| `models/DistillPRM-1.5B/ce/best_model.pt` | 8.7 G | CE-only loss baseline |
| `models/DistillPRM-1.5B/kl/best_model.pt` | 8.7 G | KL-only loss baseline |
| `models/DistillPRM-1.5B/ablation_no_error_head/best_model.pt` | 2.9 G | No error head (score head only) |
| `models/DistillPRM-7B/adaptive_multidim/best_model.pt` | 14 G | Multi-dimensional scoring head variant |

### 1.3 Intermediate Checkpoints (low priority — can retrain)

| Path | Size | Description |
|------|------|-------------|
| `models/DistillPRM-1.5B/adaptive/best_model.pt` | 8.7 G | Early adaptive run (pre-temperature tuning) |
| `models/DistillPRM-1.5B/adaptive_t2/best_model.pt` | 8.7 G | Temperature=2 variant |
| `models/DistillPRM-1.5B/adaptive_t5/best_model.pt` | 8.7 G | Temperature=5 variant |
| `models/DistillPRM-7B/adaptive_t3_may1_backup/best_model.pt` | 10 G | Backup snapshot from May 1 (superseded by main) |

---

## 2. Open-Source Models — Can Be Downloaded

These models are publicly available on HuggingFace. No backup needed unless
you want faster re-setup (total ≈ 100 G).

### 2.1 Backbone Models (required to run DistillPRM)

| Local Path | Size | HuggingFace ID | Download Command |
|------------|------|----------------|-----------------|
| `models/Qwen2.5-Math-1.5B` | 2.9 G | `Qwen/Qwen2.5-Math-1.5B` | `huggingface-cli download Qwen/Qwen2.5-Math-1.5B --local-dir models/Qwen2.5-Math-1.5B` |
| `models/Qwen2.5-Math-7B` | 15 G | `Qwen/Qwen2.5-Math-7B` | `huggingface-cli download Qwen/Qwen2.5-Math-7B --local-dir models/Qwen2.5-Math-7B` |
| `models/Qwen2.5-Math-7B-Instruct` | 15 G | `Qwen/Qwen2.5-Math-7B-Instruct` | `huggingface-cli download Qwen/Qwen2.5-Math-7B-Instruct --local-dir models/Qwen2.5-Math-7B-Instruct` |
| `models/qwen-1.5b` | 2.9 G | `Qwen/Qwen2.5-Math-1.5B` | Same as above (duplicate, safe to skip) |

> **Note:** The backbone model path is passed as `--student_model` when loading a
> DistillPRM checkpoint. The tokenizer comes from the backbone, not the `.pt` file.

### 2.2 Baseline PRM for Comparison

| Local Path | Size | HuggingFace ID | Download Command |
|------------|------|----------------|-----------------|
| `models/Skywork-PRM-1.5B` | ~2 G | `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` | `huggingface-cli download Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B --local-dir models/Skywork-PRM-1.5B` |

> **Note:** `models/Skywork-PRM-1.5B-real/` contains only tokenizer files — it is a
> partial download artifact. Use the full download above.

### 2.3 Teacher / Reference Models

| Local Path | Size | HuggingFace ID | Notes |
|------------|------|----------------|-------|
| `models/GenPRM-1.5B` | 3.4 G | `RyanLiu112/GenPRM-1.5B` (tentative) | GenPRM paper model, backbone = DeepSeek-R1-Distill-Qwen-1.5B |
| `models/GenPRM-7B` | 15 G | `RyanLiu112/GenPRM-7B` (tentative) | GenPRM paper model, backbone = DeepSeek-R1-Distill-Qwen-7B |

> Verify the exact HF IDs at https://huggingface.co/RyanLiu112 before downloading.

### 2.4 VLM-PRM Models (for the visual reasoning experiments)

| Local Path | Size | HuggingFace ID | Download Command |
|------------|------|----------------|-----------------|
| `models/llava-critic-7b` | 15 G | `lmms-lab/llava-critic-7b` | `huggingface-cli download lmms-lab/llava-critic-7b --local-dir models/llava-critic-7b` |
| `models/Qwen2.5-VL-7B-Instruct` | 16 G | `Qwen/Qwen2.5-VL-7B-Instruct` | `huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/Qwen2.5-VL-7B-Instruct` |
| `models/VisualPRM-8B` | 16 G | `OpenGVLab/VisualPRM-8B` (tentative) | `huggingface-cli download OpenGVLab/VisualPRM-8B --local-dir models/VisualPRM-8B` |

> Verify `VisualPRM-8B` HF ID — the local config shows it was based on InternVL2_5-8B.
> Check https://huggingface.co/OpenGVLab for the exact published name.

---

## 3. Quick Reference — What to Do on a New Machine

```bash
export HF_ENDPOINT=https://hf-mirror.com   # if HuggingFace is slow

# Step 1: download all backbone/open-source models
huggingface-cli download Qwen/Qwen2.5-Math-1.5B       --local-dir models/Qwen2.5-Math-1.5B
huggingface-cli download Qwen/Qwen2.5-Math-7B          --local-dir models/Qwen2.5-Math-7B
huggingface-cli download Qwen/Qwen2.5-Math-7B-Instruct --local-dir models/Qwen2.5-Math-7B-Instruct
huggingface-cli download Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B --local-dir models/Skywork-PRM-1.5B

# Step 2: restore self-trained checkpoints from HuggingFace backup
# Backup repo: https://huggingface.co/shensignal/DistillPRM-checkpoints (private)
# Login first: huggingface-cli login

mkdir -p models/DistillPRM-1.5B/adaptive_t3
huggingface-cli download shensignal/DistillPRM-checkpoints \
    DistillPRM-1.5B/adaptive_t3/best_model.pt \
    --local-dir models/DistillPRM-1.5B/adaptive_t3 --repo-type model

mkdir -p models/DistillPRM-7B/adaptive_t3
huggingface-cli download shensignal/DistillPRM-checkpoints \
    DistillPRM-7B/adaptive_t3/best_model.pt \
    --local-dir models/DistillPRM-7B/adaptive_t3 --repo-type model

mkdir -p outputs/distillprm-7b-instruct-adaptive-t3/adaptive_t3
huggingface-cli download shensignal/DistillPRM-checkpoints \
    DistillPRM-7B-Instruct/adaptive_t3/best_model.pt \
    --local-dir outputs/distillprm-7b-instruct-adaptive-t3/adaptive_t3 --repo-type model

mkdir -p outputs/distillprm-7b-iter2
huggingface-cli download shensignal/DistillPRM-checkpoints \
    DistillPRM-7B-iter2/best_model.pt \
    --local-dir outputs/distillprm-7b-iter2 --repo-type model

mkdir -p outputs/distillprm-7b-iter2-combined
huggingface-cli download shensignal/DistillPRM-checkpoints \
    DistillPRM-7B-iter2-combined/best_model.pt \
    --local-dir outputs/distillprm-7b-iter2-combined --repo-type model
```

---

## 4. Backup Priority Summary

| Priority | Model | Size | Reason |
|----------|-------|------|--------|
| ★★★ | `DistillPRM-1.5B/adaptive_t3/best_model.pt` | 8.7 G | Main 1.5B result |
| ★★★ | `DistillPRM-7B/adaptive_t3/best_model.pt` | 14 G | Main 7B result |
| ★★★ | `distillprm-7b-instruct-adaptive-t3/.../best_model.pt` | 14 G | Best F1 (61.89%) |
| ★★★ | `distillprm-7b-iter2/best_model.pt` | 14 G | Latest iter2 |
| ★★★ | `distillprm-7b-iter2-combined/best_model.pt` | 14 G | Latest iter2-combined |
| ★★☆ | `DistillPRM-1.5B/ce/best_model.pt` | 8.7 G | Ablation |
| ★★☆ | `DistillPRM-1.5B/kl/best_model.pt` | 8.7 G | Ablation |
| ★★☆ | `DistillPRM-1.5B/ablation_no_error_head/best_model.pt` | 2.9 G | Ablation |
| ★★☆ | `DistillPRM-7B/adaptive_multidim/best_model.pt` | 14 G | Ablation |
| ★☆☆ | `DistillPRM-1.5B/adaptive_t2`, `adaptive_t5`, `adaptive` | 8.7 G each | Can retrain |
| ★☆☆ | `DistillPRM-7B/adaptive_t3_may1_backup` | 10 G | Superseded backup |
| — | All models in section 2 | ~85 G total | Re-downloadable |
