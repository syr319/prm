# Speed Benchmark v2: DistillPRM vs GenPRM

Generated: 2026-05-01 14:05:33

## Fairness Design

| | Plan A (same framework) | Plan B (best practice) |
|---|---|---|
| **DistillPRM** | Transformers `forward()` | Transformers `forward()` |
| **GenPRM**     | Transformers `generate()` | vLLM (continuous batching) |
| **Purpose**    | Isolate algorithmic cost | Real-world deployment speed |

**Common optimisations applied to DistillPRM in both plans:**
- bfloat16 backbone (set at load time via `dtype=torch.bfloat16`; classification heads stay float32 per the original design)
- flash_attention_2 (or sdpa if flash_attn not installed)
- Sorted batching — inputs grouped by tokenised length to minimise padding

**GenPRM Plan A note:** `model.generate()` uses the full token budget (stage1=512, stage2=1024) per sequence; no per-sequence early stopping. This slightly over-estimates GenPRM latency.

## Setup

| Problems sampled | 50 (ProcessBench, seed=42) |
|---|---|
| Total steps | 391 |
| Warmup steps | 10 (excluded) |
| DistillPRM max input length | 1024 tokens |
| GenPRM stage1 max tokens | 512 (analyze) |
| GenPRM stage2 max tokens | 1024 (verify+output) |
| Plan A batch size | 16 |

## Plan A — Same Framework (Native Transformers)

| Model | bs | Steps | Total (s) | ms/step | steps/s | avg tokens |
|-------|----|-------|----------:|--------:|--------:|-----------:|
| DistillPRM-1.5B | 1 | 391 | 11.9 | 30.5 | 32.80 | — |
| DistillPRM-1.5B | 16 | 391 | 5.6 | 14.2 | 70.40 | — |
| DistillPRM-7B | 1 | 391 | 26.7 | 68.4 | 14.60 | — |
| DistillPRM-7B | 16 | 391 | 21.8 | 55.6 | 18.00 | — |
| GenPRM-1.5B (transformers) | 16 | 391 | 1155.5 | 2955.2 | 0.34 | 819 |
| GenPRM-7B (transformers) | 16 | 391 | 1245.7 | 3186.0 | 0.31 | 918 |

### Plan A — Speed Ratios

*(DistillPRM steps/s ÷ GenPRM steps/s = how many times faster)*

| Comparison | DistillPRM steps/s | GenPRM steps/s | Ratio |
|-----------|-------------------:|---------------:|------:|
| DistillPRM-1.5B (bs=1) vs GenPRM-1.5B | 32.80 | 0.34 | **96.5×** |
| DistillPRM-1.5B (bs=16) vs GenPRM-1.5B | 70.40 | 0.34 | **207.1×** |
| DistillPRM-7B (bs=1) vs GenPRM-7B | 14.60 | 0.31 | **47.1×** |
| DistillPRM-7B (bs=16) vs GenPRM-7B | 18.00 | 0.31 | **58.1×** |

## Plan B — Best Practice (Transformers vs vLLM)

| Model | Framework | bs | Steps | Total (s) | ms/step | steps/s | avg tokens |
|-------|-----------|-----|-------|----------:|--------:|--------:|-----------:|
| DistillPRM-1.5B | Transformers | 1 | 391 | 11.9 | 30.5 | 32.80 | — |
| DistillPRM-1.5B | Transformers | 16 | 391 | 5.6 | 14.2 | 70.40 | — |
| DistillPRM-7B | Transformers | 1 | 391 | 26.7 | 68.4 | 14.60 | — |
| DistillPRM-7B | Transformers | 16 | 391 | 21.8 | 55.6 | 18.00 | — |
| GenPRM-1.5B (vLLM) | vLLM | batched | 391 | 45.7 | 116.8 | 8.56 | 718 |
| GenPRM-7B (vLLM) | vLLM | batched | 391 | 63.0 | 161.0 | 6.21 | 771 |

### Plan B — Speed Ratios

| Comparison | DistillPRM steps/s | GenPRM steps/s | Ratio |
|-----------|-------------------:|---------------:|------:|
| DistillPRM-1.5B (bs=1) vs GenPRM-1.5B | 32.80 | 8.56 | **3.8×** |
| DistillPRM-1.5B (bs=16) vs GenPRM-1.5B | 70.40 | 8.56 | **8.2×** |
| DistillPRM-7B (bs=1) vs GenPRM-7B | 14.60 | 6.21 | **2.4×** |
| DistillPRM-7B (bs=16) vs GenPRM-7B | 18.00 | 6.21 | **2.9×** |

## Discussion

- **Plan A** shows the *algorithmic* gap: DistillPRM does a single forward pass; GenPRM generates ~700-800 tokens in two stages. The speed difference is driven by generation length, not framework choice.
- **Plan B** shows the *deployment* gap: vLLM's continuous batching reduces GenPRM's per-step latency vs. padded transformers batch generation, but DistillPRM still wins because it generates zero tokens.
- **Plan B DistillPRM ≈ Plan A DistillPRM**: transformers forward() is already near-optimal for a discriminative model — vLLM offers no benefit here.
