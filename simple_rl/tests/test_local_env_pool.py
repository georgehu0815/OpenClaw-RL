"""
Tests for local_env_pool.py

Unit tests mock TerminalEnv; integration tests spin real Docker containers.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simple_rl.local_env_pool import LocalEnvPool


def _make_mock_env(score: float = 0.8):
    """Create a mock TerminalEnv that succeeds."""
    env = MagicMock()
    env.container_name = "mock-container"
    env.start = AsyncMock(return_value=None)
    env.exec  = AsyncMock(return_value="mock output")
    env.evaluate = AsyncMock(return_value=score)
    env.close = AsyncMock(return_value=None)
    return env


# ── Unit tests ─────────────────────────────────────────────────────────────────

class TestLocalEnvPoolUnit:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_allocate_and_close(self, trivial_task):
        """allocate() should give a lease_id; close() should release the slot."""
        pool = LocalEnvPool(max_concurrent=2)
        await pool.start()

        mock_env = _make_mock_env()
        with patch("simple_rl.local_env_pool.TerminalEnv", return_value=mock_env):
            lease_id = await pool.allocate(trivial_task)

        assert isinstance(lease_id, str) and len(lease_id) == 32
        assert pool.active_count == 1

        await pool.close(lease_id)
        assert pool.active_count == 0

        await pool.stop()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_exec_delegates_to_env(self, trivial_task):
        """exec() should call env.exec with the given command."""
        pool = LocalEnvPool(max_concurrent=2)
        await pool.start()

        mock_env = _make_mock_env()
        with patch("simple_rl.local_env_pool.TerminalEnv", return_value=mock_env):
            lease_id = await pool.allocate(trivial_task)

        out = await pool.exec(lease_id, "ls /tmp")
        assert out == "mock output"
        mock_env.exec.assert_called_once_with("ls /tmp", timeout=pytest.approx(30.0, abs=5))

        await pool.close(lease_id)
        await pool.stop()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_uses_task_grader(self, trivial_task):
        """evaluate() should call env.evaluate with the task's grader script."""
        pool = LocalEnvPool(max_concurrent=2)
        await pool.start()

        mock_env = _make_mock_env(score=0.5)
        with patch("simple_rl.local_env_pool.TerminalEnv", return_value=mock_env):
            lease_id = await pool.allocate(trivial_task)

        score = await pool.evaluate(lease_id)
        assert score == pytest.approx(0.5)
        mock_env.evaluate.assert_called_once_with("echo 0.5")

        await pool.close(lease_id)
        await pool.stop()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_max_concurrent_blocks(self, trivial_task):
        """allocate() should block when all slots are in use."""
        pool = LocalEnvPool(max_concurrent=1)
        await pool.start()

        mock_env = _make_mock_env()
        with patch("simple_rl.local_env_pool.TerminalEnv", return_value=mock_env):
            lease1 = await pool.allocate(trivial_task)

            # Second allocate should block — wrap in timeout
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(pool.allocate(trivial_task), timeout=0.1)

        await pool.close(lease1)
        await pool.stop()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unknown_lease_raises(self):
        """exec/evaluate/close on an unknown lease_id should raise KeyError."""
        pool = LocalEnvPool(max_concurrent=2)
        await pool.start()

        with pytest.raises(KeyError):
            await pool.exec("deadbeef" * 4, "ls")

        await pool.stop()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stop_closes_all_containers(self, trivial_task):
        """stop() should close all active containers."""
        pool = LocalEnvPool(max_concurrent=3)
        await pool.start()

        mocks = [_make_mock_env() for _ in range(3)]
        leases = []
        for mock_env in mocks:
            with patch("simple_rl.local_env_pool.TerminalEnv", return_value=mock_env):
                lid = await pool.allocate(trivial_task)
                leases.append(lid)

        assert pool.active_count == 3
        await pool.stop()
        assert pool.active_count == 0

        for mock_env in mocks:
            mock_env.close.assert_called_once()


# ── Integration tests ──────────────────────────────────────────────────────────

class TestLocalEnvPoolIntegration:

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_exec_and_evaluate(self, hello_task):
        """End-to-end: allocate real Docker container, exec, evaluate, close."""
        pool = LocalEnvPool(max_concurrent=1)
        await pool.start()
        try:
            lease_id = await pool.allocate(hello_task)
            # Solve the task
            await pool.exec(lease_id, "echo 'Hello World' > /tmp/hello.txt")
            score = await pool.evaluate(lease_id)
            assert score == pytest.approx(1.0)
            await pool.close(lease_id)
        finally:
            await pool.stop()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_parallel_allocations(self, hello_task):
        """Two concurrent containers should run in parallel without conflict."""
        pool = LocalEnvPool(max_concurrent=2)
        await pool.start()
        try:
            t1, t2 = hello_task.copy(), hello_task.copy()
            t1["task_name"] = "parallel_a"
            t2["task_name"] = "parallel_b"

            lid1, lid2 = await asyncio.gather(
                pool.allocate(t1), pool.allocate(t2)
            )
            assert pool.active_count == 2

            # Both envs are isolated
            await asyncio.gather(
                pool.exec(lid1, "echo 'Hello World' > /tmp/hello.txt"),
                pool.exec(lid2, "echo 'Hello World' > /tmp/hello.txt"),
            )
            s1, s2 = await asyncio.gather(
                pool.evaluate(lid1), pool.evaluate(lid2)
            )
            assert s1 == pytest.approx(1.0)
            assert s2 == pytest.approx(1.0)

            await asyncio.gather(pool.close(lid1), pool.close(lid2))
        finally:
            await pool.stop()
