"""
Tests that GRPO training batches are correctly assembled and submitted.

Covers:
  1. RolloutBuffer — accumulation, advantage computation, trigger threshold
  2. _submit_to_mlx_tune — grpo_batch_r*.jsonl schema and content
  3. train() — calls _submit_to_mlx_tune at the right time

All unit tests; no Docker, no LLM required.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simple_rl.agent_loop import Trajectory
from simple_rl.rollout_buffer import GRPOBatch, GRPOSample, RolloutBuffer
import simple_rl.train_async as ta


# ── Helpers ────────────────────────────────────────────────────────────────────

def _traj(score: float, task_name: str = "t1", n_turns: int = 2) -> Trajectory:
    return Trajectory(
        task={"task_name": task_name, "instruction": "do X"},
        messages=[
            {"role": "system",    "content": "sys"},
            {"role": "user",      "content": "task"},
            {"role": "assistant", "content": "<bash>ls</bash>"},
            {"role": "user",      "content": "<observation>file.txt</observation>"},
        ],
        score=score,
        n_turns=n_turns,
        turn_scores=[0.8, 0.6],
    )


def _batch(task_id: str = "t1", scores=(0.0, 1.0)) -> GRPOBatch:
    """Build a GRPOBatch directly from scores."""
    import statistics
    trajs = [_traj(s, task_name=task_id) for s in scores]
    mean_r = statistics.mean(scores)
    std_r  = statistics.stdev(scores) if len(scores) > 1 else 0.0
    eps = 1e-8
    samples = [
        GRPOSample(trajectory=t, advantage=(t.score - mean_r) / (std_r + eps))
        for t in trajs
    ]
    return GRPOBatch(task_id=task_id, samples=samples)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ── 1. RolloutBuffer ──────────────────────────────────────────────────────────

class TestRolloutBuffer:

    @pytest.mark.unit
    async def test_returns_none_until_full(self):
        """Buffer returns None until n_samples_per_prompt trajectories arrive."""
        buf = RolloutBuffer(n_samples_per_prompt=3, rollout_batch_size=1)
        assert await buf.add("t1", _traj(0.5)) is None
        assert await buf.add("t1", _traj(0.8)) is None
        batch = await buf.add("t1", _traj(1.0))
        assert batch is not None

    @pytest.mark.unit
    async def test_returns_batch_on_completion(self):
        """Completed batch has correct task_id and sample count."""
        n = 4
        buf = RolloutBuffer(n_samples_per_prompt=n, rollout_batch_size=1)
        batch = None
        for i in range(n):
            batch = await buf.add("find_python", _traj(float(i) / n))
        assert batch is not None
        assert batch.task_id == "find_python"
        assert len(batch.samples) == n

    @pytest.mark.unit
    async def test_advantage_formula(self):
        """A_i = (r_i - mean) / (std + eps) for each sample."""
        import statistics
        scores = [0.0, 0.5, 1.0]
        buf = RolloutBuffer(n_samples_per_prompt=len(scores), rollout_batch_size=1)
        batch = None
        for s in scores:
            batch = await buf.add("t1", _traj(s))

        mean_r = statistics.mean(scores)
        std_r  = statistics.stdev(scores)
        eps = 1e-8
        for sample in batch.samples:
            expected = (sample.trajectory.score - mean_r) / (std_r + eps)
            assert abs(sample.advantage - expected) < 1e-6

    @pytest.mark.unit
    async def test_advantages_sum_near_zero(self):
        """Normalised advantages sum to ~0 within a group."""
        scores = [0.2, 0.4, 0.6, 0.8]
        buf = RolloutBuffer(n_samples_per_prompt=len(scores), rollout_batch_size=1)
        batch = None
        for s in scores:
            batch = await buf.add("t1", _traj(s))
        total = sum(s.advantage for s in batch.samples)
        assert abs(total) < 1e-6

    @pytest.mark.unit
    async def test_separate_tasks_accumulate_independently(self):
        """Trajectories from different tasks don't mix."""
        buf = RolloutBuffer(n_samples_per_prompt=2, rollout_batch_size=1)
        # One trajectory per task — neither should trigger yet
        assert await buf.add("task_a", _traj(0.5)) is None
        assert await buf.add("task_b", _traj(0.8)) is None
        # Complete task_a
        batch_a = await buf.add("task_a", _traj(1.0))
        assert batch_a is not None
        assert batch_a.task_id == "task_a"


# ── 2. _submit_to_mlx_tune ────────────────────────────────────────────────────

class TestSubmitToMlxTune:

    @pytest.mark.unit
    def test_writes_grpo_batch_file(self):
        """grpo_batch_r{N:04d}.jsonl is created in log_dir."""
        with tempfile.TemporaryDirectory() as tmp:
            ta._submit_to_mlx_tune([_batch()], round_num=1, log_dir=tmp)
            files = list(Path(tmp).glob("grpo_batch_r*.jsonl"))
            assert len(files) == 1
            assert files[0].name == "grpo_batch_r0001.jsonl"

    @pytest.mark.unit
    def test_round_number_in_filename(self):
        """Round number is zero-padded to 4 digits."""
        with tempfile.TemporaryDirectory() as tmp:
            ta._submit_to_mlx_tune([_batch()], round_num=42, log_dir=tmp)
            assert (Path(tmp) / "grpo_batch_r0042.jsonl").exists()

    @pytest.mark.unit
    def test_schema_fields_present(self):
        """Every record has: task_id, advantage, score, n_turns, messages, turn_scores."""
        required = {"task_id", "advantage", "score", "n_turns", "messages", "turn_scores"}
        with tempfile.TemporaryDirectory() as tmp:
            ta._submit_to_mlx_tune([_batch()], round_num=1, log_dir=tmp)
            records = _read_jsonl(Path(tmp) / "grpo_batch_r0001.jsonl")
            for rec in records:
                assert required <= rec.keys(), f"Missing: {required - rec.keys()}"

    @pytest.mark.unit
    def test_one_record_per_sample(self):
        """File has one JSON line per GRPOSample."""
        batch = _batch(scores=(0.0, 0.5, 1.0))
        with tempfile.TemporaryDirectory() as tmp:
            ta._submit_to_mlx_tune([batch], round_num=1, log_dir=tmp)
            records = _read_jsonl(Path(tmp) / "grpo_batch_r0001.jsonl")
            assert len(records) == len(batch.samples)

    @pytest.mark.unit
    def test_advantage_values_written(self):
        """Advantage values in the file match the batch."""
        batch = _batch(scores=(0.0, 1.0))
        expected_advs = sorted(s.advantage for s in batch.samples)
        with tempfile.TemporaryDirectory() as tmp:
            ta._submit_to_mlx_tune([batch], round_num=1, log_dir=tmp)
            records = _read_jsonl(Path(tmp) / "grpo_batch_r0001.jsonl")
            written_advs = sorted(r["advantage"] for r in records)
            for e, w in zip(expected_advs, written_advs):
                assert abs(e - w) < 1e-9

    @pytest.mark.unit
    def test_messages_preserved(self):
        """messages list is serialised correctly (not truncated or mangled)."""
        traj = _traj(score=1.0)
        batch = GRPOBatch(
            task_id="t1",
            samples=[GRPOSample(trajectory=traj, advantage=0.0)],
        )
        with tempfile.TemporaryDirectory() as tmp:
            ta._submit_to_mlx_tune([batch], round_num=1, log_dir=tmp)
            records = _read_jsonl(Path(tmp) / "grpo_batch_r0001.jsonl")
            assert records[0]["messages"] == traj.messages

    @pytest.mark.unit
    def test_multiple_batches_in_one_round(self):
        """Multiple GRPOBatches are all written to the same file."""
        batches = [_batch("t1"), _batch("t2")]
        total_samples = sum(len(b.samples) for b in batches)
        with tempfile.TemporaryDirectory() as tmp:
            ta._submit_to_mlx_tune(batches, round_num=1, log_dir=tmp)
            records = _read_jsonl(Path(tmp) / "grpo_batch_r0001.jsonl")
            assert len(records) == total_samples

    @pytest.mark.unit
    def test_noop_on_empty_batches(self):
        """No file written when all batches have zero samples."""
        empty = GRPOBatch(task_id="empty", samples=[])
        with tempfile.TemporaryDirectory() as tmp:
            ta._submit_to_mlx_tune([empty], round_num=1, log_dir=tmp)
            files = list(Path(tmp).glob("grpo_batch_r*.jsonl"))
            assert len(files) == 0


# ── 3. train() triggers submission ────────────────────────────────────────────

class TestTrainSubmitTrigger:

    @pytest.mark.unit
    async def test_submit_called_after_enough_batches(self):
        """_submit_to_mlx_tune is called once rollout_batch_size batches accumulate."""
        n_samples = 2
        rollout_batch_size = 1

        mock_pool = MagicMock()
        mock_pool.start = AsyncMock()
        mock_pool.stop  = AsyncMock()
        mock_pool.allocate = AsyncMock(return_value="lease-1")
        mock_pool.close    = AsyncMock()
        mock_pool.evaluate = AsyncMock(return_value=1.0)
        mock_pool.exec     = AsyncMock(return_value="output")

        # Policy returns TASK_COMPLETE immediately
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "TASK_COMPLETE"
        mock_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        tasks = [{"task_name": "t1", "instruction": "do X"}] * (n_samples * 2)

        with tempfile.TemporaryDirectory() as tmp, \
             patch("simple_rl.train_async.LocalEnvPool", return_value=mock_pool), \
             patch("simple_rl.agent_loop.openai.AsyncOpenAI") as mock_oai, \
             patch.object(ta, "_submit_to_mlx_tune") as mock_submit, \
             patch.object(ta, "load_dataset", return_value=tasks), \
             patch.object(ta, "_WANDB_AVAILABLE", False):

            mock_oai.return_value.chat.completions.create = AsyncMock(return_value=mock_resp)

            await ta.train(
                dataset_path="dummy.jsonl",
                max_concurrent=1,
                n_samples_per_prompt=n_samples,
                rollout_batch_size=rollout_batch_size,
                wandb_project="",
                log_dir=tmp,
                max_rounds=1,
            )

        assert mock_submit.called, "_submit_to_mlx_tune was never called"
        call_args = mock_submit.call_args
        batches_arg = call_args.args[0]
        assert len(batches_arg) >= rollout_batch_size
