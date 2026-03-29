"""
train_async.py — Top-level async RL training loop.

Architecture
────────────
  dataset (JSONL)
      │
      ▼
  Async RL Loop   ── dispatches tasks ──▶  AgentLoop × MAX_CONCURRENT
      │                                        │
      │                               LocalEnvPool (Docker)
      │                                        │
      ◀── Trajectory (messages + score) ───────┘
      │
      ▼
  RolloutBuffer  (asyncio.Queue)
      │   accumulate N_SAMPLES_PER_PROMPT per task
      │   compute GRPO advantages
      ▼
  GRPOBatch  ──▶  _submit_to_mlx_tune()  [stub — wire to mlx-tune]

Usage
─────
    # minimal
    python -m simple_rl.train_async --dataset data/sample_tasks.jsonl

    # full options
    python -m simple_rl.train_async \\
        --dataset data/sample_tasks.jsonl \\
        --policy_url http://localhost:8080/v1 \\
        --policy_model Qwen3.5-0.8B-8bit \\
        --max_concurrent 1 \\
        --n_samples 8 \\
        --rollout_batch_size 4 \\
        --max_turns 20 \\
        --prm_enable \\
        --log_dir logs \\
        --wandb_project terminal-rl-simple \\
        --max_rounds 100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

from . import config
from .agent_loop import Trajectory, run_episode
from .local_env_pool import LocalEnvPool
from .rollout_buffer import GRPOBatch, RolloutBuffer

logger = logging.getLogger(__name__)

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> List[dict]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping bad JSON on line %d: %s", lineno, exc)
    if not tasks:
        raise ValueError(f"No tasks loaded from {path}")
    logger.info("Loaded %d tasks from %s", len(tasks), path)
    return tasks


# ── GRPO training submission ──────────────────────────────────────────────────

# Tracks the adapter path produced by the most recent successful GRPO round so
# the next round can resume from it rather than starting from the base model.
_last_adapter_path: str = ""


def _submit_to_mlx_tune(
    batches:      List[GRPOBatch],
    round_num:    int,
    log_dir:      str,
    wandb_run_id: Optional[str] = None,
) -> None:
    """
    Submit pre-collected GRPO batches to mlx-tune GRPOTrainer.

    Always writes grpo_batch_r{N}.jsonl for inspection.  Then, if
    POLICY_MODEL_PATH is set, launches grpo_worker.py under the mlx-tune venv
    to run a real gradient update and log per-step losses to W&B.

    Adapter continuity: the adapter path from round N is passed to round N+1
    so training accumulates across rounds rather than restarting from the base
    model every time.
    """
    global _last_adapter_path
    from .mlx_grpo_bridge import run_grpo_update

    all_scores = [s.trajectory.score for b in batches for s in b.samples]
    all_advs   = [s.advantage        for b in batches for s in b.samples]
    n = len(all_scores)
    if n == 0:
        return

    mean_score = sum(all_scores) / n
    mean_adv   = sum(all_advs)   / n
    logger.info(
        "=== Training round %d | n_samples=%d mean_score=%.3f mean_adv=%.3f ===",
        round_num, n, mean_score, mean_adv,
    )

    # ── Always write JSONL snapshot ───────────────────────────────────────────
    out_path = Path(log_dir) / f"grpo_batch_r{round_num:04d}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for batch in batches:
            for sample in batch.samples:
                record = {
                    "task_id":     batch.task_id,
                    "advantage":   sample.advantage,
                    "score":       sample.trajectory.score,
                    "n_turns":     sample.trajectory.n_turns,
                    "messages":    sample.trajectory.messages,
                    "turn_scores": sample.trajectory.turn_scores,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Batch written to %s", out_path)

    # ── Launch real GRPOTrainer (if model path configured) ────────────────────
    if _last_adapter_path:
        logger.info("GRPO round %d: resuming from adapter %s", round_num, _last_adapter_path)

    result = run_grpo_update(
        batches=batches,
        round_num=round_num,
        log_dir=log_dir,
        wandb_run_id=wandb_run_id,
        prev_adapter_path=_last_adapter_path,
    )

    if result.is_stub:
        logger.info(
            "GRPO round %d: stub mode (set POLICY_MODEL_PATH to enable training)",
            round_num,
        )
        return

    if result.status == "success":
        _last_adapter_path = result.adapter_path   # carry forward to next round
        logger.info(
            "GRPO round %d: adapter saved to %s | %d steps logged",
            round_num, result.adapter_path, len(result.step_losses),
        )
        if _WANDB_AVAILABLE and config.WANDB_PROJECT and result.step_losses:
            final_loss = result.step_losses[-1]["loss"]
            _wandb.log({
                "grpo/round":      round_num,
                "grpo/final_loss": final_loss,
                "grpo/adapter":    result.adapter_path,
                "grpo/mean_score": mean_score,
                "grpo/mean_adv":   mean_adv,
            })
    else:
        logger.error("GRPO round %d failed: %s", round_num, result.error)


# ── Main training loop ─────────────────────────────────────────────────────────

async def train(
    dataset_path: str        = config.DATASET_PATH,
    policy_url: str          = config.POLICY_URL,
    policy_model: str        = config.POLICY_MODEL,
    policy_api_key: str      = config.POLICY_API_KEY,
    max_concurrent: int      = config.MAX_CONCURRENT,
    n_samples_per_prompt: int = config.N_SAMPLES_PER_PROMPT,
    rollout_batch_size: int  = config.ROLLOUT_BATCH_SIZE,
    max_turns: int           = config.MAX_TURNS,
    prm_enable: bool         = config.PRM_ENABLE,
    log_dir: str             = config.LOG_DIR,
    wandb_project: str       = config.WANDB_PROJECT,
    wandb_entity: str        = config.WANDB_ENTITY,
    wandb_api_key: str       = config.WANDB_API_KEY,
    max_rounds: int          = 0,   # 0 = run forever
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    tasks = load_dataset(dataset_path)

    # W&B
    if _WANDB_AVAILABLE and wandb_project:
        if wandb_api_key:
            _wandb.login(key=wandb_api_key)
        _wandb.init(
            project=wandb_project,
            entity=wandb_entity or None,
            config={
                "policy_model":        policy_model,
                "n_samples_per_prompt": n_samples_per_prompt,
                "rollout_batch_size":  rollout_batch_size,
                "max_turns":           max_turns,
                "prm_enable":          prm_enable,
            },
        )

    # PRM client
    prm = None
    if prm_enable:
        from .prm_client import PRMClient
        prm = PRMClient(log_dir=log_dir)
        logger.info("PRM scoring enabled (model=%s, m=%d, log=%s/prm_steps.jsonl)",
                    config.PRM_MODEL, config.PRM_M, log_dir)
    else:
        logger.info("PRM scoring disabled.")
        print("\n\n==PRM scoring disabled. To enable, set prm_enable and configure PRM_URL, PRM_MODEL, etc. in config.py - train_async.py:231")

    env_pool = LocalEnvPool(max_concurrent=max_concurrent)
    await env_pool.start()

    buffer = RolloutBuffer(
        n_samples_per_prompt=n_samples_per_prompt,
        rollout_batch_size=rollout_batch_size,
    )

    traj_log_path = Path(log_dir) / "trajectories.jsonl"
    traj_log = open(traj_log_path, "a", encoding="utf-8")
    logger.info("Trajectory log: %s", traj_log_path)

    step = 0
    round_num = 0
    pending_batches: List[GRPOBatch] = []
    pending_tasks: list[asyncio.Task] = []
    task_idx = 0

    try:
        while True:
            # ── Exit condition ───────────────────────────────────────────────
            if max_rounds > 0 and round_num >= max_rounds:
                logger.info("Reached max_rounds=%d — stopping.", max_rounds)
                break

            # ── Fill concurrency slots with new episodes ──────────────────
            while len(pending_tasks) < max_concurrent:
                task = tasks[task_idx % len(tasks)]
                task_idx += 1
                t = asyncio.create_task(
                    run_episode(
                        task,
                        env_pool,
                        policy_url=policy_url,
                        policy_model=policy_model,
                        policy_api_key=policy_api_key,
                        max_turns=max_turns,
                    ),
                    name=f"episode-{step}",
                )
                t._task_meta = task   # type: ignore[attr-defined]
                pending_tasks.append(t)

            # ── Wait for any episode to finish ────────────────────────────
            done_set, pending_set = await asyncio.wait(
                pending_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            pending_tasks = list(pending_set)

            # ── Process completed episodes ────────────────────────────────
            for done in done_set:
                step += 1
                traj: Trajectory = await done
                task_meta = getattr(done, "_task_meta", traj.task)
                task_id   = task_meta.get("task_name", f"task_{task_idx}")

                # Log
                log_entry = traj.to_log_dict()
                log_entry["step"] = step
                traj_log.write(json.dumps(log_entry) + "\n")
                traj_log.flush()
                logger.info("step=%d %s", step, log_entry)

                if _WANDB_AVAILABLE and wandb_project:
                    _wandb.log({
                        "score":             traj.score,
                        "turns":             traj.n_turns,
                        "prompt_tokens":     traj.prompt_tokens,
                        "completion_tokens": traj.completion_tokens,
                        "step":              step,
                    })
                    print(f"\n\n====W&B logged step {step} with score {traj.score:.3f} - train_async.py:304")
                else:
                    print(f"\n\n====W&B not available, but would have logged step {step} with score {traj.score:.3f} - train_async.py:306")

                # Buffer
                batch = await buffer.add(task_id, traj, prm)
                if batch is not None:
                    pending_batches.append(batch)
                    buffer.put_batch(batch)

            # ── Training round: fire when enough batches are ready ────────
            if len(pending_batches) >= rollout_batch_size:
                round_num += 1
                train_batches = pending_batches[:rollout_batch_size]
                pending_batches = pending_batches[rollout_batch_size:]

                _wb_run_id = (
                    _wandb.run.id
                    if (_WANDB_AVAILABLE and wandb_project and _wandb.run)
                    else None
                )
                _submit_to_mlx_tune(train_batches, round_num, log_dir,
                                    wandb_run_id=_wb_run_id)

                mean_score = sum(
                    s.trajectory.score
                    for b in train_batches
                    for s in b.samples
                ) / max(1, sum(len(b.samples) for b in train_batches))

                if _WANDB_AVAILABLE and wandb_project:
                    _wandb.log({"round_mean_score": mean_score, "round": round_num})
                else:
                    print(f"\n\n====W&B not available, but would have logged round {round_num} with mean_score {mean_score:.3f} - train_async.py:337")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        # Cancel any in-flight episodes
        for t in pending_tasks:
            t.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        await env_pool.stop()
        traj_log.close()

        if _WANDB_AVAILABLE and wandb_project:
            _wandb.finish()

        logger.info("Training finished. rounds=%d steps=%d", round_num, step)


# ── CLI entry point ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simple async RL training loop (Apple Silicon / MLX).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",            default=config.DATASET_PATH)
    p.add_argument("--policy_url",         default=config.POLICY_URL)
    p.add_argument("--policy_model",       default=config.POLICY_MODEL)
    p.add_argument("--policy_api_key",     default=config.POLICY_API_KEY)
    p.add_argument("--max_concurrent",     type=int,   default=config.MAX_CONCURRENT)
    p.add_argument("--n_samples",          type=int,   default=config.N_SAMPLES_PER_PROMPT,
                   dest="n_samples_per_prompt")
    p.add_argument("--rollout_batch_size", type=int,   default=config.ROLLOUT_BATCH_SIZE)
    p.add_argument("--max_turns",          type=int,   default=config.MAX_TURNS)
    p.add_argument("--prm_enable",         action="store_true")
    p.add_argument("--log_dir",            default=config.LOG_DIR)
    p.add_argument("--wandb_project",      default=config.WANDB_PROJECT)
    p.add_argument("--wandb_entity",       default=config.WANDB_ENTITY)
    p.add_argument("--wandb_api_key",      default=config.WANDB_API_KEY)
    p.add_argument("--max_rounds",         type=int,   default=0,
                   help="Stop after N training rounds (0 = run forever)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(
        train(
            dataset_path=args.dataset,
            policy_url=args.policy_url,
            policy_model=args.policy_model,
            policy_api_key=args.policy_api_key,
            max_concurrent=args.max_concurrent,
            n_samples_per_prompt=args.n_samples_per_prompt,
            rollout_batch_size=args.rollout_batch_size,
            max_turns=args.max_turns,
            prm_enable=args.prm_enable,
            log_dir=args.log_dir,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_api_key=args.wandb_api_key,
            max_rounds=args.max_rounds,
        )
    )


if __name__ == "__main__":
    main()
