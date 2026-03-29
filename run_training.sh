#!/usr/bin/env bash
# run_training.sh — Launch simple-rl online RL training with GRPO + W&B
#
# Usage:
#   bash run_training.sh            # full training
#   bash run_training.sh --dry-run  # 2 rounds, no real LLM needed

set -euo pipefail

# ── W&B credentials ───────────────────────────────────────────────────────────
# Copy your key from https://wandb.ai/authorize
export WANDB_API_KEY="your-wandb-key-here"
export WANDB_PROJECT="terminal-rl-simple"
export WANDB_ENTITY="bochuxt7-iot"

# ── GRPO training (set POLICY_MODEL_PATH to enable real gradient updates) ─────
export POLICY_MODEL_PATH="mlx-community/Qwen3.5-0.8B-8bit"  # must match POLICY_MODEL below
export MLX_TUNE_PYTHON="/Volumes/ExternalSSD/train/mlx-tune/.venv/bin/python3"
export GRPO_LORA_RANK="16"
export GRPO_LR="1e-6"
export GRPO_NUM_GEN="4"

# ── Policy server ─────────────────────────────────────────────────────────────
export POLICY_URL="http://localhost:8080/v1"
export POLICY_MODEL="Qwen3.5-0.8B-8bit"
export POLICY_MAX_CONCURRENT="1"   # oMLX handles one request at a time

# ── Launch (pass --prm to enable per-step PRM scoring) ────────────────────────
POLICY_URL="$POLICY_URL" bash simple_rl/run.sh --prm "$@"
