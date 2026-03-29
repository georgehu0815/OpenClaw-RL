"""
mlx_grpo_bridge.py — Orchestrates mlx-tune GRPOTrainer from the simple_rl loop.

Architecture
────────────
  train_async.py
      │  calls run_grpo_update(batches, ...)
      ▼
  mlx_grpo_bridge  (system Python — no mlx dependency)
      │  serialises GRPOJob JSON + dataset JSONL
      │  launches subprocess ──────────────────────────────────────────────────┐
      ▼                                                                         │
  waits for result JSON  ◀──────────────────────────────────────────────────── │
      │                       grpo_worker.py (mlx-tune venv Python)            │
      │  reads step_losses,   ├─ loads model via FastLanguageModel             │
      │  adapter_path         ├─ runs WandbGRPOTrainer (logs per step to W&B) │
      ▼                       └─ writes result JSON                            │
  returns GRPOUpdateResult                                                     │
                                                                               ┘

Stub mode (POLICY_MODEL_PATH not set)
  run_grpo_update() returns immediately with is_stub=True — useful for
  integration tests and dry-run mode.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import config
from .rollout_buffer import GRPOBatch

logger = logging.getLogger(__name__)


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class GRPOUpdateResult:
    round_num:    int
    status:       str                              # "success" | "error" | "stub"
    adapter_path: str  = ""
    step_losses:  List[Dict] = field(default_factory=list)   # [{"step": N, "loss": F}]
    error:        Optional[str] = None
    is_stub:      bool = False                     # True when POLICY_MODEL_PATH unset


# ── Dataset conversion ────────────────────────────────────────────────────────

def batches_to_dataset(batches: List[GRPOBatch]) -> List[Dict]:
    """
    Convert GRPOBatch list to mlx-tune dataset format.

    Each sample becomes:
      {"prompt": task_instruction, "answer": str(trajectory_score)}

    The reward_fn in grpo_worker reads float(answer) as the pre-computed score.
    This is an offline-GRPO approximation: the model learns to produce
    outputs similar to high-scoring trajectories.
    """
    dataset = []
    for batch in batches:
        for sample in batch.samples:
            traj = sample.trajectory
            instruction = (
                traj.task.get("instruction") or traj.task.get("prompt", "")
            )
            dataset.append({
                "prompt":     instruction,
                "answer":     str(round(traj.score, 6)),
                # Store advantage too — useful for debugging / future use
                "_advantage": round(sample.advantage, 6),
                "_task_id":   batch.task_id,
            })
    return dataset


# ── Main entry point ──────────────────────────────────────────────────────────

def run_grpo_update(
    batches:        List[GRPOBatch],
    round_num:      int,
    log_dir:        str             = config.LOG_DIR,
    model_path:     str             = config.POLICY_MODEL_PATH,
    mlx_python:     str             = config.MLX_TUNE_PYTHON,
    output_dir:     str             = config.GRPO_OUTPUT_DIR,
    lora_rank:      int             = config.GRPO_LORA_RANK,
    learning_rate:  float           = config.GRPO_LR,
    num_generations:int             = config.GRPO_NUM_GEN,
    beta:           float           = config.GRPO_BETA,
    loss_type:      str             = config.GRPO_LOSS_TYPE,
    max_steps:      int             = config.GRPO_MAX_STEPS,
    logging_steps:  int             = config.GRPO_LOGGING_STEPS,
    wandb_project:  str             = config.WANDB_PROJECT,
    wandb_entity:   str             = config.WANDB_ENTITY,
    wandb_api_key:  str             = config.WANDB_API_KEY,
    wandb_run_id:      Optional[str]   = None,
    prev_adapter_path: str             = "",       # adapter from previous round; "" = start fresh
    timeout:           int             = 3600,     # subprocess timeout seconds
) -> GRPOUpdateResult:
    """
    Submit pre-collected GRPO batches to mlx-tune GRPOTrainer.

    Returns a GRPOUpdateResult with:
      - status: "success" | "error" | "stub"
      - adapter_path: path to saved LoRA adapters
      - step_losses: per-step loss history [{step, loss}, ...]
      - is_stub: True if model_path was empty (no real training done)

    If prev_adapter_path is set, the worker loads those adapters before
    training so each round continues from the previous checkpoint.
    """
    if not batches:
        logger.warning("run_grpo_update called with empty batches, skipping.")
        return GRPOUpdateResult(round_num=round_num, status="stub", is_stub=True)

    # ── Stub mode ─────────────────────────────────────────────────────────────
    if not model_path:
        logger.info(
            "POLICY_MODEL_PATH not set — GRPO update skipped (stub mode). "
            "Set POLICY_MODEL_PATH to enable real training."
        )
        return GRPOUpdateResult(round_num=round_num, status="stub", is_stub=True)

    # ── Build job payload ─────────────────────────────────────────────────────
    dataset = batches_to_dataset(batches)
    job_id = f"r{round_num:04d}_{uuid.uuid4().hex[:6]}"
    work_dir = Path(log_dir) / "grpo_jobs"
    work_dir.mkdir(parents=True, exist_ok=True)

    job_path    = work_dir / f"job_{job_id}.json"
    result_path = work_dir / f"result_{job_id}.json"

    job = {
        "round_num":       round_num,
        "model_path":      model_path,
        "lora_rank":       lora_rank,
        "dataset":         dataset,
        "output_dir":      output_dir,
        "loss_type":       loss_type,
        "beta":            beta,
        "num_generations": num_generations,
        "temperature":     0.7,
        "learning_rate":   learning_rate,
        "max_steps":       max_steps,
        "logging_steps":   logging_steps,
        "wandb_project":   wandb_project,
        "wandb_entity":    wandb_entity,
        "wandb_api_key":   wandb_api_key,
        "wandb_run_id":    wandb_run_id or "",
        "adapter_path":    prev_adapter_path,   # "" on round 1; else previous round's output
    }
    print(f">>>GRPO job payload (model_path {model_path}): {json.dumps(job, indent=2)} - mlx_grpo_bridge.py:155")
    job_path.write_text(json.dumps(job, indent=2))
    logger.info("GRPO job written: %s (%d samples)", job_path, len(dataset))

    # ── Launch worker subprocess ───────────────────────────────────────────────
    worker_module = "simple_rl.grpo_worker"
    repo_root = Path(__file__).parent.parent   # OpenClaw-RL/
    cmd = [mlx_python, "-m", worker_module, str(job_path)]

    logger.info("Launching GRPO worker: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=False,    # let stdout/stderr print live
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return GRPOUpdateResult(
            round_num=round_num, status="error",
            error=f"GRPO worker timed out after {timeout}s",
        )
    except FileNotFoundError:
        return GRPOUpdateResult(
            round_num=round_num, status="error",
            error=f"mlx_python not found: {mlx_python}",
        )

    if proc.returncode != 0:
        return GRPOUpdateResult(
            round_num=round_num, status="error",
            error=f"Worker exited with code {proc.returncode}",
        )

    # ── Read result ────────────────────────────────────────────────────────────
    if not result_path.exists():
        return GRPOUpdateResult(
            round_num=round_num, status="error",
            error=f"Worker completed but result file missing: {result_path}",
        )

    raw = json.loads(result_path.read_text())
    result = GRPOUpdateResult(
        round_num=round_num,
        status=raw.get("status", "error"),
        adapter_path=raw.get("adapter_path", ""),
        step_losses=raw.get("step_losses", []),
        error=raw.get("error"),
        is_stub=False,
    )

    if result.status == "success":
        logger.info(
            "GRPO round %d complete | adapter=%s | steps=%d",
            round_num, result.adapter_path, len(result.step_losses),
        )
    else:
        logger.error("GRPO round %d failed: %s", round_num, result.error)

    return result
