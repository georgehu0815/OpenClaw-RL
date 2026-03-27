"""
Tests for rollout_buffer.py

All unit tests — no Docker, no LLM.
"""
from __future__ import annotations

import asyncio
import statistics
from unittest.mock import AsyncMock, MagicMock

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simple_rl.agent_loop import Trajectory
from simple_rl.rollout_buffer import GRPOBatch, GRPOSample, RolloutBuffer


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_traj(task_name: str, score: float) -> Trajectory:
    task = {"task_name": task_name, "instruction": "test"}
    t = Trajectory(task=task, score=score, n_turns=1)
    return t


# ── GRPOBatch ──────────────────────────────────────────────────────────────────

class TestGRPOBatch:

    @pytest.mark.unit
    def test_mean_score(self):
        samples = [
            GRPOSample(trajectory=_make_traj("t", s), advantage=0.0)
            for s in [0.4, 0.6, 0.8]
        ]
        batch = GRPOBatch(task_id="t", samples=samples)
        assert batch.mean_score == pytest.approx(0.6)

    @pytest.mark.unit
    def test_mean_advantage(self):
        samples = [
            GRPOSample(trajectory=_make_traj("t", 0.5), advantage=a)
            for a in [-1.0, 0.0, 1.0]
        ]
        batch = GRPOBatch(task_id="t", samples=samples)
        assert batch.mean_advantage == pytest.approx(0.0)

    @pytest.mark.unit
    def test_empty_batch(self):
        batch = GRPOBatch(task_id="empty")
        assert batch.mean_score == 0.0
        assert batch.mean_advantage == 0.0


# ── RolloutBuffer ──────────────────────────────────────────────────────────────

class TestRolloutBuffer:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_returns_none_before_full(self):
        """add() should return None until n_samples_per_prompt trajectories."""
        buf = RolloutBuffer(n_samples_per_prompt=4)
        for _ in range(3):
            result = await buf.add("task_a", _make_traj("task_a", 0.5))
            assert result is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_returns_batch_when_full(self):
        """add() should return a GRPOBatch when n_samples_per_prompt are collected."""
        buf = RolloutBuffer(n_samples_per_prompt=4)
        result = None
        for i in range(4):
            result = await buf.add("task_b", _make_traj("task_b", float(i) / 3))
        assert isinstance(result, GRPOBatch)
        assert len(result.samples) == 4

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_grpo_advantage_formula(self):
        """Advantages should equal (r - mean) / (std + eps)."""
        scores = [0.2, 0.5, 0.8, 1.0]
        buf = RolloutBuffer(n_samples_per_prompt=4, eps=1e-8)
        batch = None
        for s in scores:
            batch = await buf.add("task_c", _make_traj("task_c", s))

        assert batch is not None
        mean_r = statistics.mean(scores)
        std_r  = statistics.stdev(scores)

        for sample in batch.samples:
            expected_adv = (sample.trajectory.score - mean_r) / (std_r + 1e-8)
            assert sample.advantage == pytest.approx(expected_adv, abs=1e-6)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_grpo_advantage_identical_scores(self):
        """When all scores are equal, advantages should be ~0 (std≈0 case)."""
        buf = RolloutBuffer(n_samples_per_prompt=3)
        batch = None
        for _ in range(3):
            batch = await buf.add("task_d", _make_traj("task_d", 0.7))

        assert batch is not None
        for sample in batch.samples:
            # std = 0 → advantage ≈ 0 / eps ≈ 0 (eps prevents div-by-zero)
            assert abs(sample.advantage) < 1e5   # not inf/nan

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_independent_tasks_dont_mix(self):
        """Trajectories from different task_ids should not cross-contaminate."""
        buf = RolloutBuffer(n_samples_per_prompt=2)

        # Add one traj for task_x (should not trigger a batch)
        result_x = await buf.add("task_x", _make_traj("task_x", 0.5))
        assert result_x is None

        # Fill task_y separately
        await buf.add("task_y", _make_traj("task_y", 0.3))
        result_y = await buf.add("task_y", _make_traj("task_y", 0.7))
        assert isinstance(result_y, GRPOBatch)
        assert result_y.task_id == "task_y"

        # task_x still pending
        result_x2 = await buf.add("task_x", _make_traj("task_x", 0.5))
        assert isinstance(result_x2, GRPOBatch)
        assert result_x2.task_id == "task_x"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_put_and_get_batch(self):
        """put_batch() + get_training_batches() should round-trip."""
        buf = RolloutBuffer(n_samples_per_prompt=2, rollout_batch_size=2)
        batch1 = GRPOBatch(task_id="t1")
        batch2 = GRPOBatch(task_id="t2")

        buf.put_batch(batch1)
        buf.put_batch(batch2)

        retrieved = await asyncio.wait_for(
            buf.get_training_batches(batch_size=2), timeout=1.0
        )
        assert len(retrieved) == 2
        task_ids = {b.task_id for b in retrieved}
        assert task_ids == {"t1", "t2"}

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_prm_scoring_called(self):
        """When prm_client is provided, score_trajectory should be called per traj."""
        buf = RolloutBuffer(n_samples_per_prompt=2)

        prm = MagicMock()
        prm.score_trajectory = AsyncMock(return_value=[0.7, 0.8])

        batch = None
        for _ in range(2):
            batch = await buf.add("task_e", _make_traj("task_e", 0.6), prm_client=prm)

        assert batch is not None
        assert prm.score_trajectory.call_count == 2
        # turn_scores should be populated
        for sample in batch.samples:
            assert sample.trajectory.turn_scores == [0.7, 0.8]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_queue_depth(self):
        """queue_depth should reflect number of batches queued."""
        buf = RolloutBuffer(n_samples_per_prompt=2)
        assert buf.queue_depth == 0
        buf.put_batch(GRPOBatch(task_id="t"))
        assert buf.queue_depth == 1
        await buf.get_training_batches(batch_size=1)
        assert buf.queue_depth == 0
