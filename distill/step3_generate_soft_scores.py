"""
Step 3: Generate GenPRM-7B soft scores for all training steps via vLLM batch inference.

Design decisions (based on reading GenPRM source code):

1. Model: GenPRM-7B (DeepSeek-R1-Distill-Qwen-7B fine-tuned)
   - Architecture: Qwen2ForCausalLM
   - Chat template: DeepSeek format (<｜User｜> / <｜Assistant｜>)
   - Special tokens: BOS=<｜begin▁of▁sentence｜>, EOS=<｜end▁of▁sentence｜>

2. Prompt format (follows GenPRM's build_prompt + output_template):
   - System prompt → embedded after BOS
   - Multi-turn: <｜User｜>{step_i_text} interleaved with <｜Assistant｜><EOS> (empty prior judgments)
   - Forced prefix at end: "<output>\n**Judgement**: $\\boxed"
   - This forces the model to jump directly to the Yes/No judgment (no CoT, no code)
   - This "output-only" mode is explicitly supported in GenPRM's inference() method

3. Soft score extraction (follows GenPRM's get_reward_score exactly):
   - Find the last Yes/No token in generated token_ids
   - Extract logprobs at that position for Yes and No tokens
   - Return softmax(Yes) = P(Yes) / (P(Yes) + P(No))

4. Context: full multi-turn context (all prior steps as user messages with empty assistant turns)
   - Allows model to assess step difficulty given prior context

5. Efficiency:
   - vLLM batch inference (all prompts passed in one call per checkpoint batch)
   - max_tokens=10 (just need {Yes/No}\n</output>)
   - temperature=0 (greedy, deterministic)
   - checkpoint every CKPT_EVERY steps

Usage:
    python distill/step3_generate_soft_scores.py --test          # 1000 steps, validate
    python distill/step3_generate_soft_scores.py                  # full 171K steps
    python distill/step3_generate_soft_scores.py --resume         # resume from checkpoint
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# vLLM 0.18.1: VLLM_USE_V1 is not supported; disable flashinfer version check
os.environ.pop("VLLM_USE_V1", None)
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
from vllm import LLM, SamplingParams

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
MODEL_PATH   = str(ROOT / "models" / "GenPRM-7B")
PARQUET_PATH = ROOT / "data" / "GenPRM-MATH-Data" / "data" / "train-00000-of-00001.parquet"
OUTPUT_PATH  = ROOT / "data" / "genprm_math_steps_with_soft_scores.json"
CKPT_DIR     = ROOT / "data" / "soft_score_checkpoints"

# ─── Constants ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT   = "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step."
# This prefix forces the model to output Yes/No directly without CoT.
# Exactly corresponds to GenPRM's output_template when analyze=False, verify=False.
OUTPUT_PREFIX   = "<output>\n**Judgement**: $\\boxed"

CKPT_EVERY      = 10_000   # save intermediate checkpoint every N steps
MAX_MODEL_LEN   = 16_384   # vLLM max sequence length

# ─── Regex (same as step1) ────────────────────────────────────────────────────
RE_ANALYZE = re.compile(r"<analyze>(.*?)</analyze>", re.DOTALL)
RE_VERIFY  = re.compile(r"<verify>(.*?)</verify>",   re.DOTALL)
RE_OUTPUT  = re.compile(r"<output>(.*?)</output>",   re.DOTALL)
RE_LABEL   = re.compile(r"\\boxed\{(Yes|No)\}",      re.IGNORECASE)
RE_CODE    = re.compile(r"```python\s*(.*?)\s*```",  re.DOTALL)


# ─── Prompt building ──────────────────────────────────────────────────────────

def build_prompt(messages, tokenizer):
    """
    Identical to GenPRM's build_prompt in src/utils/util.py.
    Uses chat template with add_generation_prompt=False, then strips trailing EOS.
    Result does NOT end with <｜Assistant｜> — the inference code appends the
    response prefix manually (OUTPUT_PREFIX in our case).
    """
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    eos = tokenizer.eos_token
    if prompt.endswith(f"{eos}\n"):
        prompt = prompt[: -len(f"{eos}\n")]
    elif prompt.endswith(eos):
        prompt = prompt[: -len(eos)]
    return prompt


def extract_question_and_step(first_user_content: str):
    """Split first user message into (question, step1_text)."""
    if first_user_content.startswith("Question:"):
        parts = first_user_content.split("\n\n", 1)
        question  = parts[0].removeprefix("Question:").strip()
        step_text = parts[1].strip() if len(parts) > 1 else ""
    else:
        question  = ""
        step_text = first_user_content.strip()
    return question, step_text


def parse_assistant_content(content: str):
    """Extract verification_cot, verification_code, hard_label from assistant message."""
    analyze_m = RE_ANALYZE.search(content)
    verify_m  = RE_VERIFY.search(content)
    output_m  = RE_OUTPUT.search(content)

    verification_cot  = analyze_m.group(1).strip() if analyze_m else ""
    verification_code = ""
    if verify_m:
        code_blocks = RE_CODE.findall(verify_m.group(1))
        verification_code = "\n\n".join(b.strip() for b in code_blocks)

    hard_label = -1
    if output_m:
        m = RE_LABEL.search(output_m.group(1))
        if m:
            hard_label = 1 if m.group(1).lower() == "yes" else 0

    return verification_cot, verification_code, hard_label


# ─── Prompt preparation ───────────────────────────────────────────────────────

def extract_cot_prefix(asst_content: str) -> str:
    """
    Extract the analyze+verify reasoning from an assistant message,
    stripping the final <output> block.

    The training data assistant format is:
      <analyze>...</analyze>
      <verify>...</verify>
      <output>**Judgement**: $\boxed{Yes/No}$</output>

    We want everything up to (but not including) <output>, so that we can
    append OUTPUT_PREFIX and get the Yes/No logprob conditional on the
    full reasoning chain.
    """
    # Split at <output> and keep only the reasoning part
    prefix = asst_content.split("<output>")[0].rstrip()
    return prefix


def prepare_all_prompts(df, tokenizer, max_steps=None):
    """
    Build all scoring prompts using the CoT-prefix approach.

    For each step N in a conversation:
      - Prior steps (0..N-1): empty assistant turns (short, no context leak)
      - Current step N: user message + assistant CoT prefix from training data
        (the actual analyze+verify reasoning GenPRM produced for this step)
      - Appended suffix: OUTPUT_PREFIX  →  forces model to output Yes/No

    Full prompt structure:
      <BOS>{sys}<｜User｜>{step0}<｜Assistant｜><EOS>     ← empty prior
      ...
      <｜User｜>{stepN}<｜Assistant｜>{cot_prefix}<output>\\n**Judgement**: $\\boxed

    This gives a meaningful soft score because:
      - The model sees the actual reasoning chain for the current step
      - The Yes/No logprob reflects confidence GIVEN the full CoT
      - Hard/ambiguous steps tend to have longer, less decisive CoT → lower confidence
      - Clearly correct/wrong steps have decisive CoT → high/low confidence

    Returns:
      prompts : list[str]  — ready-to-use prompt strings
      records : list[dict] — metadata for each step (soft_score=None, filled later)
    """
    prompts = []
    records = []
    total_steps_seen = 0

    for conv_idx in tqdm(range(len(df)), desc="Building prompts"):
        conv = df.iloc[conv_idx]["conversations"]

        total_steps = sum(1 for m in conv if m["role"] == "assistant")

        # Running state for this conversation
        messages_context = [{"role": "system", "content": SYSTEM_PROMPT}]
        question   = ""
        prev_steps = []
        step_idx   = 0

        for i in range(1, len(conv), 2):
            if i >= len(conv) or conv[i]["role"] != "user":
                break
            if i + 1 >= len(conv) or conv[i + 1]["role"] != "assistant":
                break

            user_content = conv[i]["content"]
            asst_content = conv[i + 1]["content"]

            if step_idx == 0:
                question, current_step = extract_question_and_step(user_content)
            else:
                current_step = user_content.strip()

            context = "\n\n".join(prev_steps)
            verification_cot, verification_code, hard_label = parse_assistant_content(asst_content)

            # CoT prefix: the reasoning GenPRM produced for THIS step (from training data)
            # Strip the <output> block — we will append OUTPUT_PREFIX instead
            cot_prefix = extract_cot_prefix(asst_content)

            # Build scoring messages:
            #   prior steps  → empty assistant turns (fast, no label leak)
            #   current step → user + CoT prefix as partial assistant response
            scoring_msgs = list(messages_context) + [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": cot_prefix},
            ]
            # build_prompt strips the trailing EOS so we get:
            #   ...<｜User｜>{stepN}<｜Assistant｜>{cot_prefix}
            # then we append OUTPUT_PREFIX to get the judgment prefix
            prompt = build_prompt(scoring_msgs, tokenizer) + OUTPUT_PREFIX
            prompts.append(prompt)

            records.append({
                "conv_idx":          conv_idx,
                "question":          question,
                "context":           context,
                "current_step":      current_step,
                "verification_cot":  verification_cot,
                "verification_code": verification_code,
                "hard_label":        hard_label,
                "step_index":        step_idx,
                "total_steps":       total_steps,
                "soft_score":        None,
            })

            # Update context for next step: empty prior assistant turns keep prompts short
            messages_context.append({"role": "user",      "content": user_content})
            messages_context.append({"role": "assistant", "content": ""})
            prev_steps.append(current_step)
            step_idx        += 1
            total_steps_seen += 1

            if max_steps and total_steps_seen >= max_steps:
                return prompts, records

    return prompts, records


# ─── Soft score extraction ────────────────────────────────────────────────────

def extract_soft_score(vllm_output, yes_token_id: int, no_token_id: int) -> float:
    """
    Extract P(Yes) / (P(Yes) + P(No)) from a single vLLM RequestOutput.

    Follows GenPRM's get_reward_score() logic exactly:
    - Find the last Yes/No token in the generated sequence
    - Get its logprob and the competing token's logprob from top-20 logprobs
    - Return the softmax probability for Yes

    Falls back to 0.5 if no Yes/No token is found.
    """
    out       = vllm_output.outputs[0]
    text      = out.text
    token_ids = list(out.token_ids)
    lp_list   = out.logprobs   # list[dict[int, Logprob]], one entry per generated token

    # Check that Yes or No appears in the generated text
    boxed_match = re.search(r"(Yes|No)\}", text, re.IGNORECASE)
    if not boxed_match:
        return 0.5

    decision = boxed_match.group(1).capitalize()

    if decision == "Yes":
        # Find last occurrence of yes_token in the token sequence
        try:
            idx = len(token_ids) - 1 - token_ids[::-1].index(yes_token_id)
        except ValueError:
            return 0.5
        lp_map = lp_list[idx]
        if yes_token_id not in lp_map:
            return 0.5
        yes_p = math.exp(lp_map[yes_token_id].logprob)
        try:
            no_p = math.exp(lp_map[no_token_id].logprob)
        except (KeyError, AttributeError):
            # No not in top-20: use minimum logprob as lower bound
            no_p = math.exp(min(v.logprob for v in lp_map.values()))
    else:
        # decision == "No"
        try:
            idx = len(token_ids) - 1 - token_ids[::-1].index(no_token_id)
        except ValueError:
            return 0.5
        lp_map = lp_list[idx]
        if no_token_id not in lp_map:
            return 0.5
        no_p = math.exp(lp_map[no_token_id].logprob)
        try:
            yes_p = math.exp(lp_map[yes_token_id].logprob)
        except (KeyError, AttributeError):
            yes_p = math.exp(min(v.logprob for v in lp_map.values()))

    denom = yes_p + no_p
    return yes_p / denom if denom > 0 else 0.5


# ─── Statistics ───────────────────────────────────────────────────────────────

def print_score_stats(records):
    """Print soft score distribution and sanity checks."""
    scored  = [r for r in records if r["soft_score"] is not None]
    scores  = np.array([r["soft_score"] for r in scored])
    labels  = np.array([r["hard_label"]  for r in scored])

    correct = scores[labels == 1]
    wrong   = scores[labels == 0]

    print(f"\n{'='*55}")
    print(f"SOFT SCORE STATISTICS  (N={len(scored):,})")
    print(f"{'='*55}")
    print(f"Overall:  mean={scores.mean():.3f}  std={scores.std():.3f}  "
          f"median={np.median(scores):.3f}")
    if len(correct):
        print(f"Correct steps (N={len(correct):,}): "
              f"mean={correct.mean():.3f}  median={np.median(correct):.3f}")
    if len(wrong):
        print(f"Wrong steps   (N={len(wrong):,}): "
              f"mean={wrong.mean():.3f}  median={np.median(wrong):.3f}")

    # Score bucket distribution
    print("\nScore bucket distribution:")
    for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        bucket = scores[(scores >= lo) & (scores < hi)]
        pct    = len(bucket) / len(scores) * 100
        bar    = "█" * int(pct / 2)
        print(f"  [{lo:.1f},{hi:.1f}): {len(bucket):6,}  ({pct:5.1f}%)  {bar}")

    # Sanity check: alignment with hard labels
    if len(correct) and len(wrong):
        acc_corr = np.mean(correct >= 0.5)
        acc_wrong = np.mean(wrong < 0.5)
        print(f"\nSanity check (soft score vs hard label):")
        print(f"  Correct steps with soft_score ≥ 0.5: {acc_corr:.1%}")
        print(f"  Wrong steps   with soft_score < 0.5: {acc_wrong:.1%}")

    # Difficulty distribution (score near 0.5 = ambiguous = hard)
    difficulty = 1.0 - 2.0 * np.abs(scores - 0.5)
    print(f"\nDifficulty (1 - 2|score - 0.5|):")
    print(f"  mean={difficulty.mean():.3f}  p90={np.percentile(difficulty, 90):.3f}")
    hard_frac = np.mean(difficulty > 0.5)
    print(f"  Fraction with difficulty > 0.5 (genuinely ambiguous): {hard_frac:.1%}")

    # Fallback rate
    fallback = np.sum(scores == 0.5)
    print(f"\nFallback (score == 0.5, no Yes/No found): {fallback:,} ({fallback/len(scores):.2%})")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test",    action="store_true",
                   help="Run on first --test-n steps only (default: 1000)")
    p.add_argument("--test-n",  type=int, default=1000,
                   help="Number of steps to process in test mode")
    p.add_argument("--resume",  action="store_true",
                   help="Resume from the latest checkpoint in CKPT_DIR")
    p.add_argument("--output",  type=str, default=str(OUTPUT_PATH),
                   help="Output JSON path")
    return p.parse_args()


def find_latest_checkpoint(ckpt_dir: Path):
    """Return (ckpt_path, n_done) for the latest checkpoint, or (None, 0)."""
    if not ckpt_dir.exists():
        return None, 0
    ckpts = sorted(ckpt_dir.glob("ckpt_*.json"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        return None, 0
    latest = ckpts[-1]
    n_done = int(latest.stem.split("_")[1])
    return latest, n_done


def main():
    args = parse_args()
    test_n      = args.test_n if args.test else None
    output_path = Path(args.output)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("GenPRM-7B Soft Score Generation")
    print("=" * 55)
    print(f"Model       : {MODEL_PATH}")
    print(f"Mode        : {'TEST (%d steps)' % test_n if test_n else 'FULL'}")
    print(f"Output      : {output_path}")

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading tokenizer...")
    tokenizer   = AutoTokenizer.from_pretrained(MODEL_PATH)
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[-1]
    no_token_id  = tokenizer.encode("No",  add_special_tokens=False)[-1]
    print(f"  Yes token: id={yes_token_id}  repr={tokenizer.decode([yes_token_id])!r}")
    print(f"  No  token: id={no_token_id}   repr={tokenizer.decode([no_token_id])!r}")

    # ── 2. Build all prompts from parquet ─────────────────────────────────────
    print("\n[2/4] Loading parquet and building prompts...")
    df = pd.read_parquet(PARQUET_PATH)
    prompts, records = prepare_all_prompts(df, tokenizer, max_steps=test_n)
    print(f"  Built {len(prompts):,} prompts from {len(df):,} conversations.")

    # Show sample prompt to verify format
    print(f"\nSample prompt (step 0):")
    print(f"  {prompts[0][:300].replace(chr(10), '\\n')!r}...")
    print(f"  ...{prompts[0][-80:].replace(chr(10), '\\n')!r}")

    if len(prompts) > 1:
        print(f"\nSample prompt (step 1, has context):")
        print(f"  {prompts[1][:300].replace(chr(10), '\\n')!r}...")

    # ── 3. Resume from checkpoint if requested ────────────────────────────────
    start_idx = 0
    if args.resume and not args.test:
        ckpt_path, n_done = find_latest_checkpoint(CKPT_DIR)
        if ckpt_path:
            print(f"\n[Resume] Loading checkpoint: {ckpt_path}  ({n_done:,} steps done)")
            with open(ckpt_path) as f:
                done_records = json.load(f)
            # Overwrite the first n_done records with checkpoint data
            for i, r in enumerate(done_records):
                records[i] = r
            start_idx = n_done
            print(f"  Resuming from step {start_idx:,}")
        else:
            print("\n[Resume] No checkpoint found. Starting from scratch.")

    remaining_prompts  = prompts[start_idx:]
    remaining_count    = len(remaining_prompts)
    print(f"\nSteps to process: {remaining_count:,}")

    if remaining_count == 0:
        print("Nothing to do — all steps already scored.")
        print_score_stats(records)
        return

    # ── 4. Load vLLM model ────────────────────────────────────────────────────
    print("\n[3/4] Loading GenPRM-7B with vLLM...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        enforce_eager=True,            # disable CUDA graphs (avoids flashinfer compile issues)
        enable_chunked_prefill=True,
        max_model_len=MAX_MODEL_LEN,
        dtype="bfloat16",
    )

    # Sampling params: output-only mode, greedy, minimal tokens
    sampling_params = SamplingParams(
        temperature=0,                       # greedy — deterministic soft scores
        max_tokens=10,                       # {Yes}\n</output>\n = ~5 tokens
        stop=["</output>\n", "\n\n"],        # stop after judgment
        include_stop_str_in_output=True,     # include stop string (for regex matching)
        logprobs=20,                         # top-20 logprobs at each position
    )

    # ── 5. Batch inference with checkpointing ─────────────────────────────────
    print("\n[4/4] Running inference...")
    global_offset = start_idx  # index into records[]

    for ckpt_start in range(0, remaining_count, CKPT_EVERY):
        ckpt_end    = min(ckpt_start + CKPT_EVERY, remaining_count)
        batch       = remaining_prompts[ckpt_start:ckpt_end]
        batch_size  = len(batch)
        abs_start   = global_offset + ckpt_start
        abs_end     = global_offset + ckpt_end

        print(f"\n  Checkpoint batch: steps {abs_start:,}–{abs_end:,}  ({batch_size:,} prompts)")
        outputs = llm.generate(batch, sampling_params, use_tqdm=True)

        # Extract soft scores
        batch_scores = [
            extract_soft_score(out, yes_token_id, no_token_id)
            for out in outputs
        ]

        # Fill scores into records
        for i, score in enumerate(batch_scores):
            records[abs_start + i]["soft_score"] = score

        # Quick stats for this batch
        arr = np.array(batch_scores)
        print(f"  Batch stats: mean={arr.mean():.3f}  "
              f"fallbacks={(arr == 0.5).sum()} ({(arr == 0.5).mean():.1%})")

        # Save checkpoint (skip in test mode to avoid cluttering disk)
        if not args.test:
            ckpt_file = CKPT_DIR / f"ckpt_{abs_end}.json"
            with open(ckpt_file, "w", encoding="utf-8") as f:
                json.dump(records[:abs_end], f, ensure_ascii=False)
            print(f"  Saved checkpoint: {ckpt_file}")

    # ── 6. Final statistics and save ──────────────────────────────────────────
    print_score_stats(records)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(records):,} records to {output_path}")

    if not args.test:
        # Clean up intermediate checkpoints
        for ckpt in CKPT_DIR.glob("ckpt_*.json"):
            ckpt.unlink()
        print(f"Cleaned up checkpoint directory.")


if __name__ == "__main__":
    main()
