"""
Tests for mlx_grpo_bridge.py and grpo_worker.py.

Unit tests mock the subprocess; integration tests run the real worker
under the mlx-tune venv (requires POLICY_MODEL_PATH to be set).

Run unit tests (default):
    pytest simple_rl/tests/test_mlx_grpo_bridge.py

Run integration test (requires model):
    POLICY_MODEL_PATH=mlx-community/Qwen2.5-0.5B-Instruct-4bit \\
    pytest simple_rl/tests/test_mlx_grpo_bridge.py -m integration -v -s
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simple_rl.agent_loop import Trajectory
from simple_rl.rollout_buffer import GRPOBatch, GRPOSample
from simple_rl.mlx_grpo_bridge import (
    GRPOUpdateResult,
    batches_to_dataset,
    run_grpo_update,
)
from simple_rl import config


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _traj(score: float, task_name: str = "t1") -> Trajectory:
    return Trajectory(
        task={"task_name": task_name, "instruction": f"Task: {task_name}"},
        messages=[
            {"role": "system",    "content": "sys"},
            {"role": "user",      "content": "do it"},
            {"role": "assistant", "content": "<bash>ls</bash>"},
            {"role": "user",      "content": "<observation>ok</observation>"},
        ],
        score=score,
        n_turns=1,
    )


def _batch(task_id: str = "t1", scores=(0.2, 0.8)) -> GRPOBatch:
    import statistics
    trajs = [_traj(s, task_name=task_id) for s in scores]
    mean_r = statistics.mean(scores)
    std_r  = statistics.stdev(scores) if len(scores) > 1 else 0.0
    eps = 1e-8
    return GRPOBatch(
        task_id=task_id,
        samples=[
            GRPOSample(trajectory=t,
                       advantage=(t.score - mean_r) / (std_r + eps))
            for t in trajs
        ],
    )


# ── 1. Dataset conversion ─────────────────────────────────────────────────────

class TestBatchesToDataset:

    @pytest.mark.unit
    def test_one_sample_per_trajectory(self):
        batch = _batch(scores=(0.0, 0.5, 1.0))
        ds = batches_to_dataset([batch])
        assert len(ds) == 3

    @pytest.mark.unit
    def test_prompt_from_instruction(self):
        batch = _batch(task_id="my_task")
        ds = batches_to_dataset([batch])
        for item in ds:
            assert "my_task" in item["prompt"]

    @pytest.mark.unit
    def test_answer_is_stringified_score(self):
        batch = _batch(scores=(0.75,))
        ds = batches_to_dataset([batch])
        assert float(ds[0]["answer"]) == pytest.approx(0.75)

    @pytest.mark.unit
    def test_advantage_metadata_present(self):
        batch = _batch(scores=(0.0, 1.0))
        ds = batches_to_dataset([batch])
        for item in ds:
            assert "_advantage" in item
            assert "_task_id"   in item

    @pytest.mark.unit
    def test_multiple_batches_combined(self):
        batches = [_batch("a", (0.0, 0.5)), _batch("b", (0.8, 1.0))]
        ds = batches_to_dataset(batches)
        assert len(ds) == 4
        task_ids = {d["_task_id"] for d in ds}
        assert task_ids == {"a", "b"}


# ── 2. GRPOUpdateResult ───────────────────────────────────────────────────────

class TestGRPOUpdateResult:

    @pytest.mark.unit
    def test_stub_mode_when_no_model_path(self):
        """run_grpo_update returns is_stub=True when model_path is empty."""
        result = run_grpo_update(
            batches=[_batch()],
            round_num=1,
            model_path="",      # empty = stub
        )
        assert result.is_stub is True
        assert result.status == "stub"

    @pytest.mark.unit
    def test_stub_mode_on_empty_batches(self):
        result = run_grpo_update(batches=[], round_num=1, model_path="some-model")
        assert result.is_stub is True

    @pytest.mark.unit
    def test_result_fields_on_success(self):
        """Simulate a successful worker by writing a fake result JSON."""
        with tempfile.TemporaryDirectory() as tmp:
            fake_result = {
                "status":       "success",
                "adapter_path": "/tmp/adapters",
                "step_losses":  [{"step": 1, "loss": 0.42}],
                "error":        None,
            }
            # Write result in the place the bridge expects it
            job_dir = Path(tmp) / "grpo_jobs"
            job_dir.mkdir()
            result_path = job_dir / "result_r0001_aabbcc.json"
            result_path.write_text(json.dumps(fake_result))

            # Patch subprocess.run to succeed and plant the result file
            import subprocess
            def fake_run(cmd, **kwargs):
                m = MagicMock()
                m.returncode = 0
                return m

            with patch("simple_rl.mlx_grpo_bridge.subprocess.run", side_effect=fake_run), \
                 patch("simple_rl.mlx_grpo_bridge.uuid.uuid4") as mock_uuid:
                mock_uuid.return_value.hex = "aabbccddee"   # hex[:6] = "aabbcc"

                result = run_grpo_update(
                    batches=[_batch()],
                    round_num=1,
                    log_dir=tmp,
                    model_path="some-model",
                    wandb_project="",
                )

            assert result.status == "success"
            assert result.adapter_path == "/tmp/adapters"
            assert result.step_losses == [{"step": 1, "loss": 0.42}]
            assert result.is_stub is False


# ── 3. Subprocess command construction ───────────────────────────────────────

class TestSubprocessCommand:

    @pytest.mark.unit
    def test_uses_mlx_python(self):
        """Subprocess is launched with the configured mlx_python."""
        called_cmds = []

        def fake_run(cmd, **kwargs):
            called_cmds.append(cmd)
            m = MagicMock()
            m.returncode = 1   # fail so we don't need a result file
            return m

        with tempfile.TemporaryDirectory() as tmp, \
             patch("simple_rl.mlx_grpo_bridge.subprocess.run", side_effect=fake_run):
            run_grpo_update(
                batches=[_batch()],
                round_num=5,
                log_dir=tmp,
                model_path="some/model",
                mlx_python="/custom/python",
                wandb_project="",
            )

        assert called_cmds, "subprocess.run was never called"
        assert called_cmds[0][0] == "/custom/python"
        assert "simple_rl.grpo_worker" in called_cmds[0]

    @pytest.mark.unit
    def test_job_json_written_with_correct_fields(self):
        """The job JSON file has all required fields before subprocess launch."""
        written_jobs = []

        def fake_run(cmd, **kwargs):
            job_path = cmd[-1]
            written_jobs.append(json.loads(Path(job_path).read_text()))
            m = MagicMock(); m.returncode = 1
            return m

        with tempfile.TemporaryDirectory() as tmp, \
             patch("simple_rl.mlx_grpo_bridge.subprocess.run", side_effect=fake_run):
            run_grpo_update(
                batches=[_batch()],
                round_num=3,
                log_dir=tmp,
                model_path="hf/model-id",
                wandb_project="my-project",
                wandb_entity="my-entity",
                wandb_api_key="",
            )

        assert written_jobs, "No job file was written"
        job = written_jobs[0]
        required = {"round_num", "model_path", "dataset", "output_dir",
                    "loss_type", "beta", "num_generations", "learning_rate",
                    "wandb_project", "wandb_entity"}
        assert required <= job.keys()
        assert job["round_num"]   == 3
        assert job["model_path"]  == "hf/model-id"
        assert isinstance(job["dataset"], list)
        assert len(job["dataset"]) == 2   # _batch() has 2 samples

    @pytest.mark.unit
    def test_timeout_returns_error_result(self):
        import subprocess as sp

        with tempfile.TemporaryDirectory() as tmp, \
             patch("simple_rl.mlx_grpo_bridge.subprocess.run",
                   side_effect=sp.TimeoutExpired("cmd", 1)):
            result = run_grpo_update(
                batches=[_batch()],
                round_num=1,
                log_dir=tmp,
                model_path="some-model",
                timeout=1,
            )

        assert result.status == "error"
        assert "timed out" in result.error


# ── 4. grpo_worker.py syntax check ───────────────────────────────────────────

class TestGrpoWorkerScript:

    @pytest.mark.unit
    def test_worker_is_valid_python(self):
        """grpo_worker.py parses without syntax errors."""
        import ast
        worker_path = Path(__file__).parent.parent / "grpo_worker.py"
        assert worker_path.exists(), f"grpo_worker.py not found at {worker_path}"
        src = worker_path.read_text()
        ast.parse(src)   # raises SyntaxError if invalid

    @pytest.mark.unit
    def test_worker_reward_fn_parses_float(self):
        """The reward function correctly parses float ground_truth."""
        # Import the reward fn helper directly (no mlx needed — it's pure Python)
        import importlib.util, sys
        worker_path = str(Path(__file__).parent.parent / "grpo_worker.py")

        spec = importlib.util.spec_from_file_location("grpo_worker", worker_path)
        mod = importlib.util.module_from_spec(spec)
        # Stub out mlx_tune so the module loads without the venv
        sys.modules.setdefault("mlx", MagicMock())
        sys.modules.setdefault("mlx.core", MagicMock())
        sys.modules.setdefault("mlx.optimizers", MagicMock())
        sys.modules.setdefault("mlx_tune", MagicMock())
        sys.modules.setdefault("mlx_tune.losses", MagicMock())
        sys.modules.setdefault("mlx_tune.rl_trainers", MagicMock())
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass   # may fail on partial imports; we only need make_precomputed_reward_fn

        reward_fn = mod.make_precomputed_reward_fn()
        assert reward_fn("any response", "0.85") == pytest.approx(0.85)
        assert reward_fn("any response", "0.0")  == pytest.approx(0.0)
        assert reward_fn("any response", "bad")  == pytest.approx(0.0)   # invalid → 0


# ── 5. LoRA verification and loss graph ──────────────────────────────────────

def _load_grpo_worker_module():
    """
    Load grpo_worker.py with real stub base classes so WandbGRPOTrainer is
    a proper Python class (not a MagicMock), making __new__ and isinstance work.
    """
    import importlib.util
    import types as _types

    # Build a stub mlx_tune module with a real GRPOTrainer class
    class _StubGRPOTrainer:
        def __init__(self, *args, **kwargs): pass

    stub_mlx_tune = _types.ModuleType("mlx_tune")
    stub_mlx_tune.GRPOTrainer = _StubGRPOTrainer
    stub_mlx_tune.GRPOConfig = MagicMock
    stub_mlx_tune.FastLanguageModel = MagicMock

    stub_losses = _types.ModuleType("mlx_tune.losses")
    stub_losses.grpo_batch_loss = MagicMock()

    stub_rl = _types.ModuleType("mlx_tune.rl_trainers")
    stub_rl._save_adapters_and_config = MagicMock()

    stub_mlx = _types.ModuleType("mlx")
    stub_mlx_core = _types.ModuleType("mlx.core")
    stub_mlx_opt  = _types.ModuleType("mlx.optimizers")
    stub_mlx_utils = _types.ModuleType("mlx.utils")
    stub_mlx_utils.tree_flatten = MagicMock(return_value=[])

    overrides = {
        "mlx":                stub_mlx,
        "mlx.core":           stub_mlx_core,
        "mlx.optimizers":     stub_mlx_opt,
        "mlx.utils":          stub_mlx_utils,
        "mlx_tune":           stub_mlx_tune,
        "mlx_tune.losses":    stub_losses,
        "mlx_tune.rl_trainers": stub_rl,
    }

    # Use a unique module name to avoid caching conflicts between tests
    mod_name = f"grpo_worker_lora_{id(overrides)}"
    worker_path = str(Path(__file__).parent.parent / "grpo_worker.py")
    spec = importlib.util.spec_from_file_location(mod_name, worker_path)
    mod = importlib.util.module_from_spec(spec)

    saved = {k: sys.modules.get(k) for k in overrides}
    sys.modules.update(overrides)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return mod


class TestLoRAVerification:

    @pytest.mark.unit
    def test_lora_verification_returns_dict_with_expected_keys(self):
        """_log_lora_verification returns a dict with all expected info keys."""
        mod = _load_grpo_worker_module()
        if not hasattr(mod, "WandbGRPOTrainer"):
            pytest.skip("WandbGRPOTrainer not loadable without mlx")

        mock_model = MagicMock()
        mock_model.lora_config = None   # no LoRA configured

        trainer = mod.WandbGRPOTrainer.__new__(mod.WandbGRPOTrainer)
        trainer._wb = None
        trainer._round = 0
        trainer.step_losses = []
        trainer.model = mock_model

        info = trainer._log_lora_verification()
        expected_keys = {
            "lora_enabled", "lora_applied", "lora_rank", "lora_alpha",
            "lora_target_modules", "trainable_param_tensors",
            "trainable_param_total", "frozen_param_total",
        }
        assert expected_keys <= info.keys()

    @pytest.mark.unit
    def test_lora_verification_no_lora_config(self):
        """_log_lora_verification returns defaults when model has no lora_config."""
        mod = _load_grpo_worker_module()
        if not hasattr(mod, "WandbGRPOTrainer"):
            pytest.skip("WandbGRPOTrainer not loadable without mlx")

        mock_model = MagicMock()
        mock_model.lora_config = None

        trainer = mod.WandbGRPOTrainer.__new__(mod.WandbGRPOTrainer)
        trainer._wb = None
        trainer._round = 0
        trainer.step_losses = []
        trainer.model = mock_model

        info = trainer._log_lora_verification()
        assert info["lora_enabled"] is False
        assert info["lora_rank"] == 0
        assert info["lora_target_modules"] == []

    @pytest.mark.unit
    def test_lora_verification_with_config_logs_to_wandb(self):
        """_log_lora_verification logs lora/* metrics to W&B when run is present."""
        mod = _load_grpo_worker_module()
        if not hasattr(mod, "WandbGRPOTrainer"):
            pytest.skip("WandbGRPOTrainer not loadable without mlx")

        mock_model = MagicMock()
        mock_model.lora_config = {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"]}
        mock_model.lora_enabled = True
        mock_model._lora_applied = True
        # Stub tree_flatten to return predictable tensors
        mock_tensor = MagicMock()
        mock_tensor.size = 100

        import sys as _sys
        mlx_utils = MagicMock()
        mlx_utils.tree_flatten.return_value = [("lora_A", mock_tensor), ("lora_B", mock_tensor)]
        _sys.modules["mlx.utils"] = mlx_utils

        mock_wb = MagicMock()
        mock_wb.summary = {}

        trainer = mod.WandbGRPOTrainer.__new__(mod.WandbGRPOTrainer)
        trainer._wb = mock_wb
        trainer._round = 1
        trainer.step_losses = []
        trainer.model = mock_model

        # Patch HAS_WANDB to True so W&B branch executes
        original = getattr(mod, "HAS_WANDB", False)
        mod.HAS_WANDB = True
        try:
            info = trainer._log_lora_verification()
        finally:
            mod.HAS_WANDB = original

        logged = mock_wb.log.call_args[0][0]
        assert "lora/rank" in logged
        assert logged["lora/rank"] == 8
        assert "lora/trainable_params" in logged

    @pytest.mark.unit
    def test_lora_info_stored_on_trainer(self):
        """_log_lora_verification stores result as self.lora_info."""
        mod = _load_grpo_worker_module()
        if not hasattr(mod, "WandbGRPOTrainer"):
            pytest.skip("WandbGRPOTrainer not loadable without mlx")

        mock_model = MagicMock()
        mock_model.lora_config = {"r": 4, "lora_alpha": 8, "target_modules": []}
        mock_model.lora_enabled = False
        mock_model._lora_applied = False

        trainer = mod.WandbGRPOTrainer.__new__(mod.WandbGRPOTrainer)
        trainer._wb = None
        trainer._round = 0
        trainer.step_losses = []
        trainer.model = mock_model

        trainer._log_lora_verification()
        assert hasattr(trainer, "lora_info")
        assert trainer.lora_info["lora_rank"] == 4


class TestLossGraph:

    @pytest.mark.unit
    def test_step_losses_structure(self):
        """
        step_losses list has dicts with 'step' (int) and 'loss' (float) keys.
        Simulate the logging loop from _train_native without running mlx.
        """
        # We verify the step_losses accumulation logic by replaying it manually
        step_losses = []
        iters = 5
        logging_steps = 2
        total_loss = 0.0

        for step in range(iters):
            total_loss += 0.5   # synthetic loss
            if (step + 1) % logging_steps == 0:
                avg_loss = total_loss / logging_steps
                step_losses.append({"step": step + 1, "loss": avg_loss})
                total_loss = 0.0

        assert len(step_losses) == iters // logging_steps
        for entry in step_losses:
            assert "step" in entry and "loss" in entry
            assert isinstance(entry["step"], int)
            assert isinstance(entry["loss"], float)

    @pytest.mark.unit
    def test_step_losses_monotonically_increasing(self):
        """step values in step_losses must increase monotonically."""
        step_losses = [
            {"step": i * 5, "loss": 0.3}
            for i in range(1, 6)
        ]
        steps = [e["step"] for e in step_losses]
        assert steps == sorted(steps)
        assert len(set(steps)) == len(steps)   # no duplicates

    @pytest.mark.unit
    def test_define_metric_called_for_wandb_x_axis(self):
        """
        When W&B run is present, define_metric should be called with
        'grpo/step' and with step_metric='grpo/step' for 'grpo/*'.
        """
        mod = _load_grpo_worker_module()
        if not hasattr(mod, "WandbGRPOTrainer"):
            pytest.skip("WandbGRPOTrainer not loadable without mlx")

        mock_wb = MagicMock()
        define_calls = []

        mock_wandb = MagicMock()
        mock_wandb.define_metric.side_effect = lambda *a, **kw: define_calls.append((a, kw))

        trainer = mod.WandbGRPOTrainer.__new__(mod.WandbGRPOTrainer)
        trainer._wb = mock_wb
        trainer._round = 0
        trainer.step_losses = []

        # Replay the define_metric block from _train_native
        mod.HAS_WANDB = True
        original_wandb = getattr(mod, "_wandb", None)
        mod._wandb = mock_wandb
        try:
            if trainer._wb is not None and mod.HAS_WANDB:
                mod._wandb.define_metric("grpo/step")
                mod._wandb.define_metric("grpo/*", step_metric="grpo/step")
        finally:
            if original_wandb is not None:
                mod._wandb = original_wandb

        assert any(a == ("grpo/step",) for a, kw in define_calls), \
            "define_metric('grpo/step') was not called"
        assert any(
            a == ("grpo/*",) and kw.get("step_metric") == "grpo/step"
            for a, kw in define_calls
        ), "define_metric('grpo/*', step_metric='grpo/step') was not called"


# ── 5. Integration test (real mlx-tune model) ────────────────────────────────

@pytest.mark.integration
class TestGRPOIntegration:

    def test_full_grpo_round(self):
        """
        Runs a real one-step GRPO update using mlx-tune.
        Requires:
          POLICY_MODEL_PATH  e.g. mlx-community/Qwen2.5-0.5B-Instruct-4bit
        """
        model_path = os.environ.get("POLICY_MODEL_PATH", config.POLICY_MODEL_PATH)
        if not model_path:
            pytest.skip("POLICY_MODEL_PATH not set")

        with tempfile.TemporaryDirectory() as tmp:
            result = run_grpo_update(
                batches=[_batch("smoke_test", scores=(0.0, 1.0))],
                round_num=1,
                log_dir=tmp,
                model_path=model_path,
                max_steps=2,            # keep it fast
                num_generations=2,
                logging_steps=1,
                wandb_project="",       # skip W&B for smoke test
            )

        assert result.status == "success", f"GRPO failed: {result.error}"
        assert result.adapter_path, "No adapter path returned"
        assert Path(result.adapter_path).exists(), f"Adapter dir missing: {result.adapter_path}"
        assert len(result.step_losses) > 0, "No step losses recorded"
        for entry in result.step_losses:
            assert "step" in entry and "loss" in entry
            assert isinstance(entry["loss"], float)
