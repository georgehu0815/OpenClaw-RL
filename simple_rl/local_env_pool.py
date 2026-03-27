"""
local_env_pool.py — In-process pool of TerminalEnv Docker containers.

Replaces the Router Server + Pool Server + remote-worker tier from the
old terminal-rl architecture.  Everything runs on the same Mac; no HTTP
hops between components.

Public API (mirrors the old HTTP pool server, but called in-process):

    pool = LocalEnvPool(max_concurrent=4)
    await pool.start()

    lease_id = await pool.allocate(task)
    obs      = await pool.exec(lease_id, "ls /tmp")
    score    = await pool.evaluate(lease_id)
    await pool.close(lease_id)

    await pool.stop()          # cleanup at shutdown
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from . import config
from .terminal_env import TerminalEnv

logger = logging.getLogger(__name__)


@dataclass
class _Slot:
    env: TerminalEnv
    task: dict
    lease_id: str
    last_used: float = field(default_factory=time.monotonic)


class LocalEnvPool:
    """
    Manages a bounded pool of Docker containers for parallel RL episodes.

    Capacity is enforced via an asyncio.Semaphore so callers simply
    ``await pool.allocate(task)`` and block until a slot is free.
    """

    def __init__(
        self,
        max_concurrent: int = config.MAX_CONCURRENT,
        docker_image: str = config.DOCKER_IMAGE,
        idle_timeout: float = config.IDLE_TIMEOUT,
    ) -> None:
        self._max = max_concurrent
        self._docker_image = docker_image
        self._idle_timeout = idle_timeout

        self._sem: asyncio.Semaphore = asyncio.Semaphore(max_concurrent)
        self._slots: Dict[str, _Slot] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self._reaper: Optional[asyncio.Task] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the idle-container reaper background task."""
        self._reaper = asyncio.create_task(self._idle_reaper(), name="env-pool-reaper")
        logger.info("LocalEnvPool started (max_concurrent=%d)", self._max)

    async def stop(self) -> None:
        """Cancel reaper and forcefully close all containers."""
        if self._reaper is not None:
            self._reaper.cancel()
            try:
                await self._reaper
            except asyncio.CancelledError:
                pass

        async with self._lock:
            lease_ids = list(self._slots.keys())

        # Close all slots (outside lock to allow concurrent closes)
        await asyncio.gather(
            *(self._close_slot(lid) for lid in lease_ids),
            return_exceptions=True,
        )
        logger.info("LocalEnvPool stopped")

    # ── Public API ─────────────────────────────────────────────────────────────

    async def allocate(self, task: dict) -> str:
        """
        Acquire a concurrency slot, start a fresh Docker container, and
        return an opaque lease_id for this episode.

        Blocks until a slot is available (up to max_concurrent parallel envs).
        """
        await self._sem.acquire()
        lease_id = uuid.uuid4().hex
        env = TerminalEnv(docker_image=self._docker_image)
        try:
            await env.start(task)
        except Exception:
            self._sem.release()
            raise

        slot = _Slot(env=env, task=task, lease_id=lease_id)
        async with self._lock:
            self._slots[lease_id] = slot

        logger.debug("Allocated lease=%s container=%s", lease_id, env.container_name)
        return lease_id

    async def exec(
        self, lease_id: str, bash_cmd: str, timeout: float = config.EXEC_TIMEOUT
    ) -> str:
        """Run a bash command in the leased container and return stdout+stderr."""
        slot = await self._get_slot(lease_id)
        slot.last_used = time.monotonic()
        return await slot.env.exec(bash_cmd, timeout=timeout)

    async def evaluate(self, lease_id: str) -> float:
        """
        Run the task's grader script inside the container.
        Returns a score in [0.0, 1.0].
        """
        slot = await self._get_slot(lease_id)
        grader = slot.task.get("grader", "")
        return await slot.env.evaluate(grader)

    async def close(self, lease_id: str) -> None:
        """Release the slot: stop container, release semaphore."""
        await self._close_slot(lease_id)

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _get_slot(self, lease_id: str) -> _Slot:
        async with self._lock:
            slot = self._slots.get(lease_id)
        if slot is None:
            raise KeyError(f"Unknown lease_id: {lease_id}")
        return slot

    async def _close_slot(self, lease_id: str) -> None:
        async with self._lock:
            slot = self._slots.pop(lease_id, None)

        if slot is None:
            return  # already closed

        try:
            await slot.env.close()
        except Exception as exc:
            logger.warning("Error closing env for lease=%s: %s", lease_id, exc)
        finally:
            self._sem.release()

        logger.debug("Released lease=%s", lease_id)

    async def _idle_reaper(self) -> None:
        """Periodically remove containers that have been idle too long."""
        while True:
            try:
                await asyncio.sleep(60)
                now = time.monotonic()
                async with self._lock:
                    stale = [
                        lid
                        for lid, slot in self._slots.items()
                        if (now - slot.last_used) > self._idle_timeout
                    ]
                if stale:
                    logger.info("Reaping %d idle container(s)", len(stale))
                    await asyncio.gather(
                        *(self._close_slot(lid) for lid in stale),
                        return_exceptions=True,
                    )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("Reaper error: %s", exc)

    # ── Diagnostics ────────────────────────────────────────────────────────────

    @property
    def active_count(self) -> int:
        return len(self._slots)

    @property
    def available_slots(self) -> int:
        return self._max - len(self._slots)
