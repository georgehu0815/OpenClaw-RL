"""
Tests for agent_loop.py

All tests use a mock LocalEnvPool and mock OpenAI client —
no Docker or running LLM required.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simple_rl.agent_loop import (
    DONE_SIGNAL,
    Trajectory,
    _parse_bash_cmd,
    _trim_messages,
    run_episode,
)


# ── Helper: mock pool ──────────────────────────────────────────────────────────

def _make_pool(exec_output: str = "ok\n", eval_score: float = 1.0):
    pool = MagicMock()
    pool.allocate  = AsyncMock(return_value="fake-lease-id")
    pool.exec      = AsyncMock(return_value=exec_output)
    pool.evaluate  = AsyncMock(return_value=eval_score)
    pool.close     = AsyncMock(return_value=None)
    return pool


def _make_openai_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    """Build a minimal fake openai ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# ── parse_bash_cmd ─────────────────────────────────────────────────────────────

class TestParseBashCmd:

    @pytest.mark.unit
    def test_xml_tags(self):
        text = "Let me check:\n<bash>\nls /tmp\n</bash>"
        assert _parse_bash_cmd(text) == "ls /tmp"

    @pytest.mark.unit
    def test_markdown_fence(self):
        text = "Running:\n```bash\necho hello\n```"
        assert _parse_bash_cmd(text) == "echo hello"

    @pytest.mark.unit
    def test_sh_fence(self):
        text = "```sh\npwd\n```"
        assert _parse_bash_cmd(text) == "pwd"

    @pytest.mark.unit
    def test_no_command(self):
        assert _parse_bash_cmd("I am thinking…") is None

    @pytest.mark.unit
    def test_case_insensitive_tag(self):
        text = "<BASH>echo hello</BASH>"
        assert _parse_bash_cmd(text) == "echo hello"

    @pytest.mark.unit
    def test_multiline_cmd(self):
        cmd = "for i in 1 2 3; do\n  echo $i\ndone"
        text = f"<bash>\n{cmd}\n</bash>"
        assert _parse_bash_cmd(text) == cmd


# ── _trim_messages ─────────────────────────────────────────────────────────────

class TestTrimMessages:

    @pytest.mark.unit
    def test_within_budget(self):
        msgs = [
            {"role": "system",    "content": "sys"},
            {"role": "user",      "content": "task"},
            {"role": "assistant", "content": "cmd"},
            {"role": "user",      "content": "obs"},
        ]
        result = _trim_messages(msgs, budget=10_000)
        assert result == msgs

    @pytest.mark.unit
    def test_trims_old_turns(self):
        big = "x" * 3000
        msgs = [
            {"role": "system",    "content": "sys"},          # always kept
            {"role": "user",      "content": "task"},          # always kept
            {"role": "assistant", "content": big},             # old — may be dropped
            {"role": "user",      "content": big},             # old — may be dropped
            {"role": "assistant", "content": "new cmd"},
            {"role": "user",      "content": "new obs"},
        ]
        result = _trim_messages(msgs, budget=500)
        # system and first user always survive
        assert result[0]["content"] == "sys"
        assert result[1]["content"] == "task"
        total = sum(len(m["content"]) for m in result)
        assert total <= 500

    @pytest.mark.unit
    def test_short_messages_unchanged(self):
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user",   "content": "u"},
        ]
        assert _trim_messages(msgs, budget=5) == msgs


# ── run_episode ────────────────────────────────────────────────────────────────

class TestRunEpisode:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_task_complete_signal_ends_loop(self, hello_task):
        """Agent saying TASK_COMPLETE should end the episode early."""
        pool = _make_pool(eval_score=1.0)
        responses = [
            _make_openai_response("<bash>echo 'Hello World' > /tmp/hello.txt</bash>"),
            _make_openai_response(f"Done. {DONE_SIGNAL}"),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=responses)

        with patch("simple_rl.agent_loop.openai.AsyncOpenAI", return_value=mock_client):
            traj = await run_episode(hello_task, pool, policy_url="http://fake/v1")

        assert traj.done is True
        assert traj.score == pytest.approx(1.0)
        assert traj.n_turns == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_max_turns_respected(self, hello_task):
        """Episode should stop at max_turns even without TASK_COMPLETE."""
        pool = _make_pool(eval_score=0.0)
        # Every response issues a command (never completes)
        responses = [
            _make_openai_response("<bash>ls</bash>")
        ] * 100  # plenty of responses

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=responses)

        with patch("simple_rl.agent_loop.openai.AsyncOpenAI", return_value=mock_client):
            traj = await run_episode(
                hello_task, pool, policy_url="http://fake/v1", max_turns=3
            )

        assert traj.n_turns == 3
        assert traj.done is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_bash_cmd_returns_hint(self, hello_task):
        """Missing <bash> tag should inject a hint, not crash."""
        pool = _make_pool(eval_score=0.0)
        responses = [
            _make_openai_response("I am thinking about the task..."),
            _make_openai_response(DONE_SIGNAL),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=responses)

        with patch("simple_rl.agent_loop.openai.AsyncOpenAI", return_value=mock_client):
            traj = await run_episode(hello_task, pool, policy_url="http://fake/v1")

        # The hint message should be in the trajectory
        hint_msgs = [
            m for m in traj.messages
            if m.get("role") == "user" and "AGENT_HINT" in m.get("content", "")
        ]
        assert len(hint_msgs) >= 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_policy_error_records_and_returns(self, hello_task):
        """OpenAI API error should set traj.error and still return a Trajectory."""
        import openai as _openai
        pool = _make_pool()

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=_openai.APIError("server down", request=MagicMock(), body=None)
        )

        with patch("simple_rl.agent_loop.openai.AsyncOpenAI", return_value=mock_client):
            traj = await run_episode(hello_task, pool, policy_url="http://fake/v1")

        assert traj.error is not None
        assert isinstance(traj, Trajectory)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_token_tracking(self, hello_task):
        """Token counts from the API should accumulate in the trajectory."""
        pool = _make_pool(eval_score=0.5)
        responses = [
            _make_openai_response("<bash>ls</bash>", prompt_tokens=50, completion_tokens=20),
            _make_openai_response(DONE_SIGNAL,      prompt_tokens=60, completion_tokens=10),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=responses)

        with patch("simple_rl.agent_loop.openai.AsyncOpenAI", return_value=mock_client):
            traj = await run_episode(hello_task, pool, policy_url="http://fake/v1")

        assert traj.prompt_tokens == 110
        assert traj.completion_tokens == 30

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_called_on_exception(self, hello_task):
        """Pool.close() must be called even if the policy errors out."""
        import openai as _openai
        pool = _make_pool()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=_openai.APIError("boom", request=MagicMock(), body=None)
        )

        with patch("simple_rl.agent_loop.openai.AsyncOpenAI", return_value=mock_client):
            await run_episode(hello_task, pool, policy_url="http://fake/v1")

        pool.close.assert_called_once()
