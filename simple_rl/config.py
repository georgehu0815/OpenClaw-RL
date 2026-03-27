"""
Central configuration for simple-rl.
All values are overridable via environment variables.
"""
import os

# ── Policy model (oMLX OpenAI-compat server) ──────────────────────────────────
POLICY_URL   = os.getenv("POLICY_URL",   "http://localhost:8080/v1")
POLICY_MODEL = os.getenv("POLICY_MODEL", "Qwen3-8B-4bit")

# ── PRM model (optional step-scoring slot) ────────────────────────────────────
PRM_URL    = os.getenv("PRM_URL",    "http://localhost:8081/v1")
PRM_MODEL  = os.getenv("PRM_MODEL",  "Qwen3-4B-4bit")
PRM_ENABLE = bool(int(os.getenv("PRM_ENABLE", "0")))
PRM_M      = int(os.getenv("PRM_M", "3"))          # majority-vote count per step

# ── Environment / Docker ──────────────────────────────────────────────────────
DOCKER_IMAGE    = os.getenv("DOCKER_IMAGE",    "ubuntu:22.04")
MAX_CONCURRENT  = int(os.getenv("MAX_CONCURRENT",  "4"))
IDLE_TIMEOUT    = int(os.getenv("IDLE_TIMEOUT",    "600"))   # seconds
EXEC_TIMEOUT    = float(os.getenv("EXEC_TIMEOUT",  "30.0"))  # per-command timeout
EVAL_TIMEOUT    = float(os.getenv("EVAL_TIMEOUT",  "60.0"))  # evaluator timeout

# ── Episode / agent ───────────────────────────────────────────────────────────
MAX_TURNS   = int(os.getenv("MAX_TURNS",   "20"))
CONTEXT_LEN = int(os.getenv("CONTEXT_LEN", "16384"))  # approx char budget

# ── GRPO / training ───────────────────────────────────────────────────────────
N_SAMPLES_PER_PROMPT = int(os.getenv("N_SAMPLES_PER_PROMPT", "8"))
ROLLOUT_BATCH_SIZE   = int(os.getenv("ROLLOUT_BATCH_SIZE",   "4"))
KL_LOSS_COEF         = float(os.getenv("KL_LOSS_COEF",       "0.01"))
LORA_RANK            = int(os.getenv("LORA_RANK",            "16"))

# ── Data / logging ────────────────────────────────────────────────────────────
DATASET_PATH  = os.getenv("DATASET_PATH",  "data/sample_tasks.jsonl")
LOG_DIR       = os.getenv("LOG_DIR",       "logs")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "")   # empty = disable W&B
