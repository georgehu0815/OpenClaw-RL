"""
Tests for W&B integration in train_async.py.

Unit tests mock the wandb library (no network calls).
Integration test hits the real W&B API — requires WANDB_API_KEY in env.

Run unit tests only (default):
    pytest simple_rl/tests/test_wandb.py

Run integration test:
    WANDB_API_KEY=<key> pytest simple_rl/tests/test_wandb.py -m integration -v
"""
from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simple_rl import config
import simple_rl.train_async as ta


from contextlib import contextmanager

@contextmanager
def _patch_wandb(mock_wandb):
    """
    Inject mock_wandb into train_async regardless of whether the real wandb
    package is installed.  Works by patching sys.modules AND the module-level
    names that train_async uses (_wandb, _WANDB_AVAILABLE).
    """
    with patch.dict("sys.modules", {"wandb": mock_wandb}), \
         patch.object(ta, "_wandb", mock_wandb, create=True), \
         patch.object(ta, "_WANDB_AVAILABLE", True, create=True):
        yield


def _make_mock_wandb():
    """Return a MagicMock that stands in for the wandb module."""
    mock = MagicMock()
    mock.init.return_value = MagicMock(
        get_url=lambda: "https://wandb.ai/bochuxt7-iot/terminal-rl-simple/runs/abc123"
    )
    return mock


def _make_mock_pool():
    """Return a LocalEnvPool mock with async start/stop."""
    pool = MagicMock()
    pool.start = AsyncMock()
    pool.stop  = AsyncMock()
    return pool


async def _run_train_headless(mock_pool, **train_kwargs):
    """Call ta.train() with minimal deps patched so it exits after init."""
    with patch.object(ta, "load_dataset", return_value=[{"task_name": "t1", "instruction": "x"}]), \
         patch("simple_rl.train_async.LocalEnvPool", return_value=mock_pool), \
         patch("simple_rl.train_async.RolloutBuffer"):
        try:
            await ta.train(dataset_path="dummy.jsonl", max_rounds=1, **train_kwargs)
        except Exception:
            pass  # only care about wandb call assertions


# ── Unit tests (no network) ────────────────────────────────────────────────────

class TestWandbUnit:

    @pytest.mark.unit
    def test_config_values(self):
        """Config exports the expected W&B defaults."""
        assert config.WANDB_PROJECT == os.getenv("WANDB_PROJECT", "terminal-rl-simple")
        assert config.WANDB_ENTITY  == os.getenv("WANDB_ENTITY",  "bochuxt7-iot")
        assert isinstance(config.WANDB_API_KEY, str)

    @pytest.mark.unit
    async def test_login_called_when_key_present(self):
        """wandb.login(key=…) is called when wandb_api_key is non-empty."""
        mock_wandb = _make_mock_wandb()
        mock_pool  = _make_mock_pool()

        with _patch_wandb(mock_wandb):
            await _run_train_headless(
                mock_pool,
                wandb_project="terminal-rl-simple",
                wandb_entity="bochuxt7-iot",
                wandb_api_key="fake-key-123",
            )

        mock_wandb.login.assert_called_once_with(key="fake-key-123")

    @pytest.mark.unit
    async def test_login_skipped_when_no_key(self):
        """wandb.login() is NOT called when wandb_api_key is empty."""
        mock_wandb = _make_mock_wandb()
        mock_pool  = _make_mock_pool()

        with _patch_wandb(mock_wandb):
            await _run_train_headless(
                mock_pool,
                wandb_project="terminal-rl-simple",
                wandb_entity="bochuxt7-iot",
                wandb_api_key="",
            )

        mock_wandb.login.assert_not_called()

    @pytest.mark.unit
    async def test_init_entity_and_project_passed(self):
        """wandb.init receives the correct entity and project."""
        mock_wandb = _make_mock_wandb()
        mock_pool  = _make_mock_pool()

        with _patch_wandb(mock_wandb):
            await _run_train_headless(
                mock_pool,
                wandb_project="terminal-rl-simple",
                wandb_entity="bochuxt7-iot",
                wandb_api_key="",
            )

        kw = mock_wandb.init.call_args.kwargs
        assert kw["project"] == "terminal-rl-simple"
        assert kw["entity"]  == "bochuxt7-iot"

    @pytest.mark.unit
    async def test_wandb_disabled_when_project_empty(self):
        """wandb.init is never called when wandb_project=''."""
        mock_wandb = _make_mock_wandb()
        mock_pool  = _make_mock_pool()

        with _patch_wandb(mock_wandb):
            await _run_train_headless(
                mock_pool,
                wandb_project="",
                wandb_api_key="",
            )

        mock_wandb.init.assert_not_called()

    @pytest.mark.unit
    async def test_finish_called_on_exit(self):
        """wandb.finish() is always called when a run was started."""
        mock_wandb = _make_mock_wandb()
        mock_pool  = _make_mock_pool()

        with _patch_wandb(mock_wandb):
            await _run_train_headless(
                mock_pool,
                wandb_project="terminal-rl-simple",
                wandb_entity="bochuxt7-iot",
                wandb_api_key="",
            )

        mock_wandb.finish.assert_called_once()


# ── Integration test (real W&B API) ───────────────────────────────────────────

@pytest.mark.integration
class TestWandbIntegration:

    def test_live_run(self):
        """
        Creates a real W&B run, logs a smoke-test point, and verifies the
        run appears under bochuxt7-iot/terminal-rl-simple.

        Requires: WANDB_API_KEY env var set.
        """
        import wandb

        api_key = os.environ.get("WANDB_API_KEY", config.WANDB_API_KEY)
        if not api_key:
            pytest.skip("WANDB_API_KEY not set")

        wandb.login(key=api_key)
        run = wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY or None,
            config={"test": True},
            tags=["smoke-test"],
        )

        # Log the same fields train_async logs
        wandb.log({"score": 1.0, "turns": 3, "prompt_tokens": 100, "completion_tokens": 20, "step": 0})
        wandb.log({"round_mean_score": 0.8, "round": 1})

        url = run.url
        wandb.finish()

        assert url is not None, "run.url returned None"
        assert "bochuxt7-iot" in url,        f"Wrong entity in URL: {url}"
        assert "terminal-rl-simple" in url,  f"Wrong project in URL: {url}"
        print(f"\nLive run URL: {url}")
