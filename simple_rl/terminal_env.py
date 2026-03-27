"""
terminal_env.py — Thin wrapper around a single Docker container.

Each episode gets its own TerminalEnv.  The env is started via ``start()``,
used for bash execution via ``exec()``, scored via ``evaluate()``, then
cleaned up via ``close()``.

No terminal_bench / CAMEL dependency — pure asyncio + subprocess.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from . import config

logger = logging.getLogger(__name__)

# Maximum bytes returned per exec() call (prevents context overflow)
_MAX_OUTPUT_BYTES = 4096


@dataclass
class TerminalEnv:
    """One Docker container = one RL episode environment."""

    docker_image: str = field(default_factory=lambda: config.DOCKER_IMAGE)
    container_name: str = field(
        default_factory=lambda: f"simple-rl-{uuid.uuid4().hex[:10]}"
    )
    _running: bool = field(default=False, init=False, repr=False)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self, task: dict) -> None:
        """Launch the Docker container in detached mode."""
        cmd = [
            "docker", "run",
            "--rm",             # auto-remove after stop
            "-d",               # detached
            "--name", self.container_name,
            "--network", "none",  # no outbound network (sandboxed)
            self.docker_image,
            "sleep", "infinity",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            raise RuntimeError(
                f"Failed to start container {self.container_name}: {err}"
            )
        self._running = True
        logger.info("Started container %s", self.container_name)

    async def exec(self, bash_cmd: str, timeout: float = config.EXEC_TIMEOUT) -> str:
        """
        Run a bash command inside the container.

        Returns combined stdout+stderr, capped at _MAX_OUTPUT_BYTES.
        On timeout, returns a timeout message (does not raise).
        """
        if not self._running:
            return "[ENV_ERROR] container is not running"

        cmd = ["docker", "exec", self.container_name, "bash", "-c", bash_cmd]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            out = stdout.decode(errors="replace") + stderr.decode(errors="replace")
            if len(out) > _MAX_OUTPUT_BYTES:
                out = out[-_MAX_OUTPUT_BYTES:]  # keep tail (most recent output)
            return out
        except asyncio.TimeoutError:
            # Kill the hanging exec (best effort)
            try:
                kill_proc = await asyncio.create_subprocess_exec(
                    "docker", "exec", self.container_name, "pkill", "-9", "bash",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await kill_proc.wait()
            except Exception:
                pass
            return f"[TIMEOUT] Command exceeded {timeout}s limit."
        except Exception as exc:
            return f"[EXEC_ERROR] {type(exc).__name__}: {exc}"

    async def evaluate(
        self, grader_script: str, timeout: float = config.EVAL_TIMEOUT
    ) -> float:
        """
        Run a grader bash script inside the container.

        The script must print a float in [0, 1] on its last non-empty line.
        Returns 0.0 if the script fails or prints nothing parseable.
        """
        if not grader_script.strip():
            return 0.0

        raw = await self.exec(grader_script, timeout=timeout)
        for line in reversed(raw.strip().splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                score = float(line)
                return max(0.0, min(1.0, score))
            except ValueError:
                continue
        logger.warning(
            "Grader for %s returned no parseable float: %r",
            self.container_name,
            raw[-200:],
        )
        return 0.0

    async def close(self) -> None:
        """Stop and remove the container (force)."""
        if not self._running:
            return
        self._running = False
        cmd = ["docker", "rm", "-f", self.container_name]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=15.0)
            logger.info("Removed container %s", self.container_name)
        except asyncio.TimeoutError:
            logger.warning("Timeout removing container %s", self.container_name)
        except Exception as exc:
            logger.warning("Error removing container %s: %s", self.container_name, exc)
