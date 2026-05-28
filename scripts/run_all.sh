#!/bin/bash
# Master script: run the full MM-Vet BoN experiment end-to-end.
# Each step is idempotent (checkpointing) — safe to re-run if interrupted.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Environment ──────────────────────────────────────────────────────────────
export HF_ENDPOINT=https://hf-mirror.com
export FLASHINFER_DISABLE_VERSION_CHECK=1

# Load API key from settings.env if not already in environment
if [ -z "$DASHSCOPE_API_KEY" ]; then
    ENV_FILE="$PROJECT_DIR/.claude/settings.env"
    if [ -f "$ENV_FILE" ]; then
        export $(grep -v '^#' "$ENV_FILE" | xargs)
        echo "Loaded DASHSCOPE_API_KEY from $ENV_FILE"
    else
        echo "WARNING: DASHSCOPE_API_KEY not set and $ENV_FILE not found."
    fi
fi

cd "$PROJECT_DIR"

echo "============================================================"
echo " OpenPRM Experiment — MM-Vet BoN Baseline"
echo "============================================================"
echo ""

# ── Step 1: Download ─────────────────────────────────────────────────────────
if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
    echo "[Step 1/4] Downloading models and data..."
    bash "$SCRIPT_DIR/01_download.sh"
else
    echo "[Step 1/4] Skipping download (SKIP_DOWNLOAD=1)"
fi
echo ""

# ── Step 2: Generate candidates ───────────────────────────────────────────────
echo "[Step 2/4] Generating 8 candidates per question with Qwen2.5-VL-7B..."
python3 "$SCRIPT_DIR/02_generate_candidates.py" --batch-size 16
echo ""

# ── Step 3a: Score with LLaVA-Critic ─────────────────────────────────────────
echo "[Step 3a/4] Scoring candidates with LLaVA-Critic-7B..."
python3 "$SCRIPT_DIR/03a_score_llava_critic.py"
echo ""

# ── Step 3b: Score with VisualPRM ─────────────────────────────────────────────
echo "[Step 3b/4] Scoring candidates with VisualPRM-8B..."
python3 "$SCRIPT_DIR/03b_score_visualprm.py"
echo ""

# ── Step 4: Evaluate with Qwen-VL-Max ────────────────────────────────────────
echo "[Step 4/4] Evaluating results with Qwen-VL-Max API..."
python3 "$SCRIPT_DIR/04_evaluate_mmvet.py"
echo ""

echo "============================================================"
echo " All done! Results are in: $PROJECT_DIR/results/"
echo "============================================================"
