# Speed Benchmark: DistillPRM vs GenPRM

Generated: 2026-04-30 21:36:36

## Setup

| Parameter | Value |
|-----------|-------|
| Problems sampled | 50 (ProcessBench, seed=42) |
| Total steps | 391 |
| Warmup steps | 10 (excluded from timing) |
| Max input length (DistillPRM) | 1024 tokens |
| GenPRM Stage 1 max tokens | 512 (analyze) |
| GenPRM Stage 2 max tokens | 1024 (verify+output) |
| Code execution | disabled (generation time only) |

## Inference Speed

| Model | Type | Batch | Steps | Total (s) | ms/step | steps/s | avg tokens |
|-------|------|-------|-------|----------:|--------:|--------:|-----------:|
| DistillPRM-1.5B | Discriminative | 1 | 391 | 9.5 | 24.4 | 40.95 | — |
| DistillPRM-1.5B | Discriminative | 16 | 391 | 9.3 | 23.8 | 41.95 | — |
| DistillPRM-7B | Discriminative | 1 | 391 | 25.2 | 64.5 | 15.51 | — |
| DistillPRM-7B | Discriminative | 16 | 391 | 37.3 | 95.3 | 10.49 | — |
| GenPRM-1.5B | Generative (vLLM) | batched | 391 | 49.3 | 126.1 | 7.93 | 738 |
| GenPRM-7B | Generative (vLLM) | batched | 391 | 63.7 | 162.9 | 6.14 | 771 |

## Speed Ratios

Steps/s(DistillPRM) / Steps/s(GenPRM) — how many times faster DistillPRM is.

| Comparison | DistillPRM steps/s | GenPRM steps/s | Ratio |
|-----------|-------------------:|---------------:|------:|
| DistillPRM-1.5B (bs=1) vs GenPRM-1.5B | 40.95 | 7.93 | **5.2× faster** |
| DistillPRM-1.5B (bs=16) vs GenPRM-1.5B | 41.95 | 7.93 | **5.3× faster** |
| DistillPRM-7B (bs=1) vs GenPRM-7B | 15.51 | 6.14 | **2.5× faster** |
| DistillPRM-7B (bs=16) vs GenPRM-7B | 10.49 | 6.14 | **1.7× faster** |

## Notes

- **DistillPRM**: single `model.forward()` call per batch; timing includes tokenization and GPU transfer.
- **GenPRM**: two batched vLLM calls per step set — (1) analyze until `</analyze>`, (2) verify+output until `</output>`. All steps are processed in parallel within each stage.
- Code execution (Stage 2 `exec()`) is **excluded**; only generation latency is measured.
- Both model families evaluated on the same single GPU.
- Input context grows with step index (early steps are shorter); reported times reflect the natural distribution from ProcessBench.
