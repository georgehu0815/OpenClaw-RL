"""
Tests that PRMClient writes prm_steps.jsonl with the correct schema.

All tests are unit tests (no real PRM server required).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simple_rl.agent_loop import Trajectory
from simple_rl.prm_client import PRMClient, PRMStepResult


def _fake_trajectory(task_name: str = "find_python", n_turns: int = 2) -> Trajectory:
    """Build a trajectory with assistant+observation pairs."""
    msgs = [
        {"role": "system",  "content": "You are an agent."},
        {"role": "user",    "content": "Find python binary."},
    ]
    for i in range(n_turns):
        msgs.append({"role": "assistant",  "content": f"<bash>which python{i}</bash>"})
        msgs.append({"role": "user",       "content": f"<observation>/usr/bin/python{i}</observation>"})
    traj = Trajectory(
        task={"task_name": task_name, "instruction": "Find python binary."},
        messages=msgs,
        score=1.0,
        n_turns=n_turns,
    )
    return traj


def _mock_prm_client(tmp_dir: str, votes_per_step=None) -> PRMClient:
    """PRMClient with a mocked OpenAI backend, writing to tmp_dir."""
    votes_per_step = votes_per_step or [0.8, 0.9, 0.7]

    async def _fake_create(**kwargs):
        resp = MagicMock()
        resp.choices[0].message.content = str(votes_per_step[0])
        return resp

    client = PRMClient(log_dir=tmp_dir, session="testsession")
    # Replace the internal openai client
    fake_openai = MagicMock()
    fake_openai.chat.completions.create = AsyncMock(
        side_effect=[
            _make_resp(v) for v in votes_per_step * 20  # enough for any number of calls
        ]
    )
    client._client = fake_openai
    client._m = len(votes_per_step)
    return client


def _make_resp(vote: float) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = str(vote)
    return resp


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestPRMJsonl:

    @pytest.mark.unit
    async def test_file_created(self):
        """prm_steps.jsonl is created when log_dir is set."""
        with tempfile.TemporaryDirectory() as tmp:
            prm = _mock_prm_client(tmp, votes_per_step=[0.8, 0.9])
            traj = _fake_trajectory(n_turns=1)
            await prm.score_trajectory(traj)

            log_path = os.path.join(tmp, "prm_steps.jsonl")
            assert os.path.exists(log_path), "prm_steps.jsonl was not created"

    @pytest.mark.unit
    async def test_one_line_per_turn(self):
        """Each agent turn produces exactly one JSONL line."""
        n_turns = 3
        with tempfile.TemporaryDirectory() as tmp:
            prm = _mock_prm_client(tmp, votes_per_step=[0.8, 0.9, 0.7])
            traj = _fake_trajectory(n_turns=n_turns)
            await prm.score_trajectory(traj)

            lines = _read_jsonl(os.path.join(tmp, "prm_steps.jsonl"))
            assert len(lines) == n_turns

    @pytest.mark.unit
    async def test_schema_fields_present(self):
        """Every line has: session, task, turn, score, votes, representative_eval."""
        required = {"session", "task", "turn", "score", "votes", "representative_eval"}
        with tempfile.TemporaryDirectory() as tmp:
            prm = _mock_prm_client(tmp, votes_per_step=[0.8, 0.6])
            traj = _fake_trajectory(n_turns=2)
            await prm.score_trajectory(traj)

            for line in _read_jsonl(os.path.join(tmp, "prm_steps.jsonl")):
                missing = required - line.keys()
                assert not missing, f"Missing fields: {missing}"

    @pytest.mark.unit
    async def test_session_and_task_values(self):
        """session matches PRMClient session; task matches trajectory task_name."""
        with tempfile.TemporaryDirectory() as tmp:
            prm = _mock_prm_client(tmp, votes_per_step=[0.8])
            traj = _fake_trajectory(task_name="count_words", n_turns=1)
            await prm.score_trajectory(traj)

            line = _read_jsonl(os.path.join(tmp, "prm_steps.jsonl"))[0]
            assert line["session"] == "testsession"
            assert line["task"]    == "count_words"

    @pytest.mark.unit
    async def test_turn_index_sequential(self):
        """turn field is 0-based and sequential."""
        n_turns = 4
        with tempfile.TemporaryDirectory() as tmp:
            prm = _mock_prm_client(tmp, votes_per_step=[0.5, 0.6, 0.7])
            traj = _fake_trajectory(n_turns=n_turns)
            await prm.score_trajectory(traj)

            lines = _read_jsonl(os.path.join(tmp, "prm_steps.jsonl"))
            assert [l["turn"] for l in lines] == list(range(n_turns))

    @pytest.mark.unit
    async def test_votes_list_and_score_average(self):
        """votes is a list; score is the average of votes."""
        votes = [0.8, 0.6, 1.0]
        expected_score = sum(votes) / len(votes)
        with tempfile.TemporaryDirectory() as tmp:
            prm = _mock_prm_client(tmp, votes_per_step=votes)
            traj = _fake_trajectory(n_turns=1)
            await prm.score_trajectory(traj)

            line = _read_jsonl(os.path.join(tmp, "prm_steps.jsonl"))[0]
            assert isinstance(line["votes"], list)
            assert len(line["votes"]) == len(votes)
            assert abs(line["score"] - expected_score) < 1e-6

    @pytest.mark.unit
    async def test_representative_eval_contains_task(self):
        """representative_eval contains the task description."""
        with tempfile.TemporaryDirectory() as tmp:
            prm = _mock_prm_client(tmp, votes_per_step=[0.8])
            traj = _fake_trajectory(n_turns=1)
            await prm.score_trajectory(traj)

            line = _read_jsonl(os.path.join(tmp, "prm_steps.jsonl"))[0]
            assert "Find python binary" in line["representative_eval"]

    @pytest.mark.unit
    async def test_turn_scores_set_on_trajectory(self):
        """score_trajectory also populates traj.turn_scores as List[float]."""
        with tempfile.TemporaryDirectory() as tmp:
            prm = _mock_prm_client(tmp, votes_per_step=[0.8, 0.9])
            traj = _fake_trajectory(n_turns=2)
            scores = await prm.score_trajectory(traj)

            assert len(scores) == 2
            assert all(isinstance(s, float) for s in scores)

    @pytest.mark.unit
    async def test_no_file_when_log_dir_none(self):
        """No file is written when log_dir=None."""
        prm = PRMClient(log_dir=None)
        prm._client = MagicMock()
        prm._client.chat.completions.create = AsyncMock(
            return_value=_make_resp(0.7)
        )
        prm._m = 1
        traj = _fake_trajectory(n_turns=1)
        await prm.score_trajectory(traj)
        assert prm._log_path is None

    @pytest.mark.unit
    async def test_appends_across_trajectories(self):
        """Multiple trajectories append to the same file, not overwrite."""
        with tempfile.TemporaryDirectory() as tmp:
            prm = _mock_prm_client(tmp, votes_per_step=[0.8])
            for task in ["task_a", "task_b"]:
                traj = _fake_trajectory(task_name=task, n_turns=1)
                await prm.score_trajectory(traj)

            lines = _read_jsonl(os.path.join(tmp, "prm_steps.jsonl"))
            assert len(lines) == 2
            assert {l["task"] for l in lines} == {"task_a", "task_b"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
