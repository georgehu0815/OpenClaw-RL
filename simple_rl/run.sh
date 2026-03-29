#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh — Startup script for simple-rl online RL training
#
# Prerequisites:
#   1. Docker Desktop running                   (brew install --cask docker)
#   2. oMLX policy server running at $POLICY_URL
#   3. Python 3.12+ with openai, wandb installed
#
# Usage:
#   # Minimal dry-run (2 rounds, no real LLM)
#   POLICY_URL=http://localhost:8080/v1 bash run.sh --dry-run
#
#   # Full training run
#   POLICY_URL=http://localhost:8080/v1 bash run.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"


# === simple-rl startup ===
# [✓] Docker daemon reachable
# [✓] Policy server reachable at http://localhost:8080/v1

# Configuration:
#   POLICY_URL       = http://localhost:8080/v1
#   POLICY_MODEL     = Qwen3.5-0.8B-8bit
#   MAX_CONCURRENT   = 4
#   N_SAMPLES        = 8
#   ROLLOUT_BATCH    = 4
#   MAX_TURNS        = 20
#   DATASET          = /Volumes/ExternalSSD/train/OpenClaw-RL/simple_rl/data/sample_tasks.jsonl
#   LOG_DIR          = /Volumes/ExternalSSD/train/OpenClaw-RL/simple_rl/logs


# ── Defaults (override via env or args) ───────────────────────────────────────
POLICY_URL="${POLICY_URL:-http://localhost:8080/v1}"
POLICY_MODEL="${POLICY_MODEL:-Qwen3.5-0.8B-8bit}"
# #   POLICY_MODEL     = Qwen3.5-0.8B-8bit
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
N_SAMPLES="${N_SAMPLES:-8}"
ROLLOUT_BATCH="${ROLLOUT_BATCH:-4}"
MAX_TURNS="${MAX_TURNS:-20}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
DATASET="${DATASET:-$SCRIPT_DIR/data/sample_tasks.jsonl}"
MAX_ROUNDS=10   # 0 = run forever
        # --max_rounds 100
DRY_RUN=0
PRM_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; MAX_ROUNDS=2; N_SAMPLES=2; ROLLOUT_BATCH=2; MAX_TURNS=3 ;;
        --prm)       PRM_FLAG="--prm_enable" ;;
        --rounds)    MAX_ROUNDS="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# ── Checks ─────────────────────────────────────────────────────────────────────
echo "=== simple-rl startup ==="

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running. Start Docker Desktop first."
    exit 1
fi
echo "[✓] Docker daemon reachable"

if ! curl -s --max-time 2 "${POLICY_URL}/models" >/dev/null 2>&1; then
    if [[ $DRY_RUN -eq 0 ]]; then
        echo "WARNING: Cannot reach policy server at $POLICY_URL"
        echo "         Start oMLX: mlx_lm.server --model $POLICY_MODEL --port 8080"
        echo "         (training will fail at first LLM call)"
    fi
else
    echo "[✓] Policy server reachable at $POLICY_URL"
fi

echo ""
echo "Configuration:"
echo "  POLICY_URL       = $POLICY_URL"
echo "  POLICY_MODEL     = $POLICY_MODEL"
echo "  MAX_CONCURRENT   = $MAX_CONCURRENT"
echo "  N_SAMPLES        = $N_SAMPLES"
echo "  ROLLOUT_BATCH    = $ROLLOUT_BATCH"
echo "  MAX_TURNS        = $MAX_TURNS"
echo "  DATASET          = $DATASET"
echo "  LOG_DIR          = $LOG_DIR"
[[ $DRY_RUN -eq 1 ]] && echo "  MODE             = DRY-RUN (max_rounds=2)"
echo ""

# ── Run ─────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

exec python3 -m simple_rl.train_async \
    --dataset         "$DATASET"       \
    --policy_url      "$POLICY_URL"    \
    --policy_model    "$POLICY_MODEL"  \
    --max_concurrent  "$MAX_CONCURRENT" \
    --n_samples       "$N_SAMPLES"     \
    --rollout_batch_size "$ROLLOUT_BATCH" \
    --max_turns       "$MAX_TURNS"     \
    --log_dir         "$LOG_DIR"       \
    --max_rounds      "$MAX_ROUNDS"    \
    $PRM_FLAG
