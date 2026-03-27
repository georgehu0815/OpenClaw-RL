"""
rollout_buffer.py — asyncio.Queue-based rollout buffer with GRPO advantage.

Replaces the Ray object store from the old terminal-rl stack.

Usage
─────
    buffer = RolloutBuffer(n_samples_per_prompt=8, rollout_batch_size=4)

    # add trajectories one at a time (from concurrent episodes)
    batch = await buffer.add("task_123", traj)   # returns None until full group
    if batch is not None:
        buffer.put_batch(batch)                   # enqueue for training

    # training side: pull a full round
    round_batches = await buffer.get_training_batches(rollout_batch_size=4)

GRPO advantage
──────────────
Within each prompt group (n_samples_per_prompt trajectories from the same task):

    A_i = (r_i − mean(r)) / (std(r) + ε)

If PRM step scores are available they are stored per-turn on the Trajectory
but advantage is always computed from the final episode score.
"""
from __future__ import annotations

import asyncio
import logging
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .agent_loop import Trajectory

logger = logging.getLogger(__name__)


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class GRPOSample:
    """One trajectory annotated with its GRPO advantage."""
    trajectory: Trajectory
    advantage: float = 0.0


@dataclass
class GRPOBatch:
    """
    A group of trajectories from the same task prompt, ready for a gradient
    update.  All samples share the same prompt; advantages are normalised
    within the group.
    """
    task_id: str
    samples: List[GRPOSample] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.trajectory.score for s in self.samples) / len(self.samples)

    @property
    def mean_advantage(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.advantage for s in self.samples) / len(self.samples)


# ── Buffer ─────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Thread-safe (single asyncio event loop) rollout buffer.

    - Accumulates trajectories per task until ``n_samples_per_prompt`` are
      collected for that task.
    - Computes GRPO advantages within each group.
    - Optionally invokes a PRMClient for per-step scoring before computing
      advantages.
    - Enqueues completed GRPOBatches for the training loop to consume.
    """

    def __init__(
        self,
        n_samples_per_prompt: int = 8,
        rollout_batch_size: int = 4,
        eps: float = 1e-8,
    ) -> None:
        self._n = n_samples_per_prompt
        self._batch_size = rollout_batch_size
        self._eps = eps

        # pending[task_id] → list of trajectories not yet grouped
        self._pending: Dict[str, List[Trajectory]] = {}
        self._lock = asyncio.Lock()

        # Completed GRPOBatches waiting for the training loop
        self._queue: asyncio.Queue[GRPOBatch] = asyncio.Queue()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def add(
        self,
        task_id: str,
        traj: Trajectory,
        prm_client=None,
    ) -> Optional[GRPOBatch]:
        """
        Add one trajectory for *task_id*.  When enough samples are collected
        (n_samples_per_prompt), computes GRPO advantages and returns the batch.
        Otherwise returns None.

        Pass a PRMClient as *prm_client* to annotate each trajectory with
        per-turn step scores before advantage computation.
        """
        async with self._lock:
            group = self._pending.setdefault(task_id, [])
            group.append(traj)
            if len(group) < self._n:
                return None
            trajs = self._pending.pop(task_id)

        # PRM scoring (outside lock — it's async I/O)
        if prm_client is not None:
            await asyncio.gather(
                *(self._score_with_prm(t, prm_client) for t in trajs),
                return_exceptions=True,
            )

        batch = self._make_batch(task_id, trajs)
        logger.info(
            "Batch ready: task=%s n=%d mean_score=%.3f",
            task_id, len(trajs), batch.mean_score,
        )
        return batch

    def put_batch(self, batch: GRPOBatch) -> None:
        """Enqueue a completed batch for the training loop."""
        self._queue.put_nowait(batch)

    async def get_training_batches(
        self, batch_size: Optional[int] = None
    ) -> List[GRPOBatch]:
        """
        Collect *batch_size* GRPOBatches (blocking until all are available).
        Defaults to rollout_batch_size set at construction.
        """
        n = batch_size if batch_size is not None else self._batch_size
        batches: List[GRPOBatch] = []
        while len(batches) < n:
            batch = await self._queue.get()
            batches.append(batch)
        return batches

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _make_batch(self, task_id: str, trajs: List[Trajectory]) -> GRPOBatch:
        """Compute GRPO advantages and assemble a GRPOBatch."""
        scores = [t.score for t in trajs]
        mean_r = statistics.mean(scores)
        std_r  = statistics.stdev(scores) if len(scores) > 1 else 0.0

        samples = [
            GRPOSample(
                trajectory=t,
                advantage=(t.score - mean_r) / (std_r + self._eps),
            )
            for t in trajs
        ]
        return GRPOBatch(task_id=task_id, samples=samples)

    @staticmethod
    async def _score_with_prm(traj: Trajectory, prm_client) -> None:
        """Populate traj.turn_scores via the PRM client (best-effort)."""
        try:
            traj.turn_scores = await prm_client.score_trajectory(traj)
        except Exception as exc:
            logger.warning("PRM scoring failed for task=%s: %s",
                           traj.task.get("task_name", "?"), exc)
