"""
Tests for terminal_env.py

Unit tests mock subprocess; integration tests require Docker.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# Make the package importable from the repo root
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simple_rl.terminal_env import TerminalEnv


# ── Unit tests (no Docker) ─────────────────────────────────────────────────────

class TestTerminalEnvUnit:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_start_success(self):
        """start() should set _running=True when docker run succeeds."""
        env = TerminalEnv(docker_image="ubuntu:22.04", container_name="test-unit-start")
        fake_proc = MagicMock()
        fake_proc.returncode = 0
        fake_proc.communicate = AsyncMock(return_value=(b"container_id\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            await env.start({})

        assert env._running is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_start_failure_raises(self):
        """start() should raise RuntimeError when docker run fails."""
        env = TerminalEnv(docker_image="ubuntu:22.04", container_name="test-unit-fail")
        fake_proc = MagicMock()
        fake_proc.returncode = 1
        fake_proc.communicate = AsyncMock(return_value=(b"", b"No such image"))

        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            with pytest.raises(RuntimeError, match="Failed to start"):
                await env.start({})

        assert env._running is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_exec_not_running(self):
        """exec() on a non-started env returns an error string."""
        env = TerminalEnv()
        result = await env.exec("ls /tmp")
        assert "not running" in result.lower() or "ENV_ERROR" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_exec_returns_output(self):
        """exec() returns combined stdout+stderr from the container."""
        env = TerminalEnv()
        env._running = True

        fake_proc = MagicMock()
        fake_proc.communicate = AsyncMock(return_value=(b"hello\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            out = await env.exec("echo hello")

        assert "hello" in out

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_exec_timeout_returns_message(self):
        """exec() should return a [TIMEOUT] string (not raise) on timeout."""
        env = TerminalEnv()
        env._running = True

        async def _slow_communicate():
            await asyncio.sleep(999)

        fake_proc = MagicMock()
        fake_proc.communicate = _slow_communicate

        kill_proc = MagicMock()
        kill_proc.wait = AsyncMock(return_value=None)

        with patch("asyncio.create_subprocess_exec", side_effect=[fake_proc, kill_proc]):
            out = await env.exec("sleep 999", timeout=0.01)

        assert "TIMEOUT" in out

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_parses_float(self):
        """evaluate() should parse a float from the last non-empty output line."""
        env = TerminalEnv()
        env._running = True

        fake_proc = MagicMock()
        fake_proc.communicate = AsyncMock(return_value=(b"running tests...\n0.75\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            score = await env.evaluate("./run_tests.sh")

        assert score == pytest.approx(0.75)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_clamps_to_01(self):
        """evaluate() should clamp scores to [0.0, 1.0]."""
        env = TerminalEnv()
        env._running = True

        fake_proc = MagicMock()
        fake_proc.communicate = AsyncMock(return_value=(b"99.9\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            score = await env.evaluate("echo 99.9")

        assert score == 1.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_no_float_returns_zero(self):
        """evaluate() should return 0.0 when no parseable float is found."""
        env = TerminalEnv()
        env._running = True

        fake_proc = MagicMock()
        fake_proc.communicate = AsyncMock(return_value=(b"ERROR: test failed\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            score = await env.evaluate("echo error")

        assert score == 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_not_running(self):
        """close() on a non-running env should be a no-op."""
        env = TerminalEnv()
        # Should not raise
        await env.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_running(self):
        """close() should call docker rm -f and set _running=False."""
        env = TerminalEnv(container_name="test-close")
        env._running = True

        fake_proc = MagicMock()
        fake_proc.wait = AsyncMock(return_value=None)

        with patch("asyncio.create_subprocess_exec", return_value=fake_proc) as mock_exec:
            await env.close()

        assert env._running is False
        cmd_args = mock_exec.call_args[0]
        assert "rm" in cmd_args and "-f" in cmd_args


# ── Integration tests (require Docker) ────────────────────────────────────────

class TestTerminalEnvIntegration:

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, hello_task):
        """Start container, run a command, evaluate grader, close."""
        env = TerminalEnv(docker_image="ubuntu:22.04")
        await env.start(hello_task)
        try:
            # Create the expected file
            out = await env.exec("echo 'Hello World' > /tmp/hello.txt")
            assert "TIMEOUT" not in out and "EXEC_ERROR" not in out

            # Evaluate
            score = await env.evaluate(hello_task["grader"])
            assert score == pytest.approx(1.0)
        finally:
            await env.close()

        assert env._running is False

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_grader_fail_when_file_missing(self, hello_task):
        """Grader should return 0.0 if the agent did nothing."""
        env = TerminalEnv(docker_image="ubuntu:22.04")
        await env.start(hello_task)
        try:
            score = await env.evaluate(hello_task["grader"])
        finally:
            await env.close()
        assert score == pytest.approx(0.0)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_output_cap(self, hello_task):
        """exec() should cap output at _MAX_OUTPUT_BYTES."""
        from simple_rl.terminal_env import _MAX_OUTPUT_BYTES
        env = TerminalEnv(docker_image="ubuntu:22.04")
        await env.start(hello_task)
        try:
            # Generate > 4096 bytes of output
            out = await env.exec("python3 -c \"print('x' * 10000)\"")
            assert len(out) <= _MAX_OUTPUT_BYTES
        finally:
            await env.close()
