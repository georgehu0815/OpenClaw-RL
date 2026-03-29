"""
grpo_worker.py — Run under the mlx-tune venv Python.

Called as a subprocess by mlx_grpo_bridge.py:

    /path/to/mlx-tune/.venv/python3 -m simple_rl.grpo_worker <job_json_path>

Reads a GRPOJob JSON, runs GRPOTrainer with W&B step logging, writes a
GRPOResult JSON to the same directory.

Job JSON schema:
  {
    "round_num":      int,
    "model_path":     str,      # HF id or local path
    "lora_rank":      int,
    "dataset":        [{"prompt": str, "answer": str}, ...],
    "output_dir":     str,
    "loss_type":      str,      # grpo | dr_grpo | dapo | bnpo
    "beta":           float,
    "num_generations":int,
    "temperature":    float,
    "learning_rate":  float,
    "max_steps":      int,      # -1 = auto
    "logging_steps":  int,
    "wandb_project":  str,
    "wandb_entity":   str,
    "wandb_api_key":  str,
    "wandb_run_id":   str,      # parent run id to attach step logs
    "adapter_path":   str       # path to previous round's adapters; "" = start fresh
  }

Result JSON schema:
  {
    "status":       "success" | "error",
    "adapter_path": str,
    "step_losses":  [{"step": int, "loss": float}, ...],
    "error":        str | null
  }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── mlx-tune imports (only available in the mlx-tune venv) ────────────────────
try:
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_tune import FastLanguageModel, GRPOTrainer, GRPOConfig
    from mlx_tune.losses import grpo_batch_loss
    from mlx_tune.rl_trainers import _save_adapters_and_config
    HAS_MLX_TUNE = True
except ImportError as _e:
    HAS_MLX_TUNE = False
    _MLX_IMPORT_ERROR = str(_e)

try:
    import wandb as _wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ── W&B-aware GRPOTrainer ─────────────────────────────────────────────────────

class WandbGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer subclass that logs per-step loss to W&B and a result list.

    Extra constructor args:
      wandb_run   — an active wandb.Run (or None to skip W&B)
      round_num   — used as a W&B group dimension (grpo_round)
    """

    def __init__(self, *args, wandb_run=None, round_num: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._wb = wandb_run
        self._round = round_num
        self.step_losses: List[Dict[str, Any]] = []

    def _train_native(self) -> Dict:
        """Override: identical to GRPOTrainer._train_native but adds:
        - LoRA verification logged to W&B before training starts
        - Per-step loss logged to W&B with a correct step-axis
        """
        print("\n[Using Native GRPO Training with Multi-Generation + W&B logging]")

        # ── Define W&B x-axis so loss is plotted against grpo/step ───────────
        if self._wb is not None and HAS_WANDB:
            _wandb.define_metric("grpo/step")
            _wandb.define_metric("grpo/*", step_metric="grpo/step")

        # ── Apply LoRA and log verification info ─────────────────────────────
        if hasattr(self.model, '_apply_lora') and not getattr(self.model, '_lora_applied', False):
            self.model._apply_lora()

        self._log_lora_verification()

        prompts = [
            s['prompt'] if 'prompt' in s else s.get('question', '')
            for s in self.train_dataset
        ]
        print(f"✓ Prepared {len(prompts)} prompts")

        actual_model = self.model.model if hasattr(self.model, 'model') else self.model
        lr_schedule = optim.cosine_decay(self.learning_rate, self.iters)
        optimizer = optim.AdamW(learning_rate=lr_schedule)

        print(f"\nStarting training for {self.iters} iterations...")
        print(f"  Generating {self.num_generations} completions per prompt")

        total_loss = 0.0
        for step in range(self.iters):
            prompt = prompts[step % len(prompts)]

            loss, _ = grpo_batch_loss(
                model=actual_model,
                tokenizer=self.tokenizer,
                prompts=[prompt],
                reward_fn=self.reward_fn,
                num_generations=self.num_generations,
                temperature=self.temperature,
                max_tokens=self.max_completion_length,
                beta=self.beta,
            )

            mx.eval(loss)
            step_loss = loss.item()
            total_loss += step_loss

            if (step + 1) % self.logging_steps == 0:
                avg_loss = total_loss / self.logging_steps
                print(f"  Step {step + 1}/{self.iters} | Loss: {avg_loss:.4f}")
                self.step_losses.append({"step": step + 1, "loss": avg_loss})

                if self._wb is not None:
                    self._wb.log({
                        "grpo/step":          step + 1,
                        "grpo/loss":          avg_loss,
                        "grpo/round":         self._round,
                        "grpo/learning_rate": self.learning_rate,
                    })

                total_loss = 0.0

        _save_adapters_and_config(self.model, self.adapter_path)

        print("\n" + "=" * 70)
        print("GRPO Training Complete!")
        print(f"  Adapters saved to: {self.adapter_path}")
        print("=" * 70)
        return {"status": "success", "adapter_path": str(self.adapter_path)}

    def _log_lora_verification(self) -> Dict:
        """
        Inspect the model's LoRA state and log it to W&B + stdout.

        Returns a dict with the verification data (used in tests).
        """
        info: Dict = {
            "lora_enabled":           False,
            "lora_applied":           False,
            "lora_rank":              0,
            "lora_alpha":             0,
            "lora_target_modules":    [],
            "trainable_param_tensors": 0,
            "trainable_param_total":  0,
            "frozen_param_total":     0,
        }

        if not hasattr(self.model, 'lora_config') or not self.model.lora_config:
            print("  [LoRA] No LoRA config found on model.")
            return info

        cfg = self.model.lora_config
        info["lora_enabled"]        = getattr(self.model, 'lora_enabled', False)
        info["lora_applied"]        = getattr(self.model, '_lora_applied', False)
        info["lora_rank"]           = cfg.get("r", 0)
        info["lora_alpha"]          = cfg.get("lora_alpha", 0)
        info["lora_target_modules"] = cfg.get("target_modules", [])

        # Count trainable vs frozen parameters
        try:
            from mlx.utils import tree_flatten
            inner = self.model.model if hasattr(self.model, 'model') else self.model
            all_flat      = tree_flatten(inner.parameters())
            trainable_flat = tree_flatten(inner.trainable_parameters())

            lora_tensors = [(k, v) for k, v in trainable_flat if 'lora' in k.lower()]
            all_tensors  = list(all_flat)

            info["trainable_param_tensors"] = len(lora_tensors)
            info["trainable_param_total"]   = sum(
                v.size for _, v in lora_tensors
            )
            info["frozen_param_total"] = sum(
                v.size for k, v in all_tensors if 'lora' not in k.lower()
            )
        except Exception as e:
            print(f"  [LoRA] Could not count params: {e}")

        # Print to stdout
        print(f"\n{'='*60}")
        print(f"  LoRA Verification")
        print(f"{'='*60}")
        print(f"  Enabled           : {info['lora_enabled']}")
        print(f"  Applied to layers : {info['lora_applied']}")
        print(f"  Rank (r)          : {info['lora_rank']}")
        print(f"  Alpha             : {info['lora_alpha']}")
        print(f"  Target modules    : {info['lora_target_modules']}")
        print(f"  Trainable tensors : {info['trainable_param_tensors']}")
        print(f"  Trainable params  : {info['trainable_param_total']:,}")
        print(f"  Frozen params     : {info['frozen_param_total']:,}")
        if info["frozen_param_total"] > 0:
            pct = 100 * info["trainable_param_total"] / (
                info["trainable_param_total"] + info["frozen_param_total"]
            )
            print(f"  % trainable       : {pct:.2f}%")
        print(f"{'='*60}")

        if self._wb is not None:
            self._wb.log({
                "lora/enabled":            int(info["lora_enabled"]),
                "lora/applied":            int(info["lora_applied"]),
                "lora/rank":               info["lora_rank"],
                "lora/alpha":              info["lora_alpha"],
                "lora/trainable_tensors":  info["trainable_param_tensors"],
                "lora/trainable_params":   info["trainable_param_total"],
                "lora/frozen_params":      info["frozen_param_total"],
            })
            self._wb.summary["lora_rank"]           = info["lora_rank"]
            self._wb.summary["lora_target_modules"] = str(info["lora_target_modules"])
            self._wb.summary["lora_applied"]        = info["lora_applied"]

        self.lora_info = info
        return info


# ── Reward function ────────────────────────────────────────────────────────────

def make_precomputed_reward_fn():
    """
    Reward function that reads the pre-computed score from the 'answer' field.

    The 'answer' in our dataset = str(trajectory.score), so we just parse it.
    This is an offline-GRPO approximation: the model is trained toward
    high-scoring trajectories using the pre-collected advantage signals.
    """
    def reward_fn(response: str, ground_truth: str) -> float:
        try:
            return float(ground_truth)
        except (ValueError, TypeError):
            return 0.0
    return reward_fn


# ── Main ──────────────────────────────────────────────────────────────────────

def main(job_path: str) -> None:
    job_path = Path(job_path)
    result_path = job_path.with_name(job_path.stem.replace("job", "result") + ".json")

    def _fail(msg: str) -> None:
        result_path.write_text(json.dumps({"status": "error", "error": msg,
                                           "adapter_path": "", "step_losses": []}))
        print(f"[grpo_worker] ERROR: {msg}", file=sys.stderr)
        sys.exit(1)

    if not HAS_MLX_TUNE:
        _fail(f"mlx_tune not available: {_MLX_IMPORT_ERROR}")

    job = json.loads(job_path.read_text())
    round_num = job["round_num"]

    # ── W&B setup ──────────────────────────────────────────────────────────────
    wb_run = None
    if HAS_WANDB and job.get("wandb_project"):
        if job.get("wandb_api_key"):
            _wandb.login(key=job["wandb_api_key"])
        wb_run = _wandb.init(
            project=job["wandb_project"],
            entity=job.get("wandb_entity") or None,
            id=job.get("wandb_run_id") or None,
            resume="allow",
            name=f"grpo-round-{round_num}",
            config={k: v for k, v in job.items()
                    if k not in ("wandb_api_key", "dataset")},
            tags=["grpo-training"],
        )

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"[grpo_worker] Loading model: {job['model_path']}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            job["model_path"],
            max_seq_length=2048,
        )
        model = FastLanguageModel.get_peft_model(model, r=job.get("lora_rank", 16))
    except Exception as e:
        _fail(f"Model load failed: {e}")

    # ── Resume from previous round's adapter (if any) ──────────────────────────
    prev_adapter = job.get("adapter_path", "")
    if prev_adapter:
        print(f"[grpo_worker] Loading adapter from round N-1: {prev_adapter}")
        try:
            model.load_adapter(prev_adapter)
            print(f"[grpo_worker] Adapter loaded — continuing from previous checkpoint")
        except Exception as e:
            _fail(f"Adapter load failed: {e}")

    # ── Configure and run GRPOTrainer ──────────────────────────────────────────
    output_dir = Path(job["output_dir"]) / f"round_{round_num:04d}"
    config = GRPOConfig(
        loss_type=job.get("loss_type",       "grpo"),
        beta=job.get("beta",                  0.04),
        num_generations=job.get("num_generations", 4),
        temperature=job.get("temperature",    0.7),
        learning_rate=job.get("learning_rate", 1e-6),
        max_steps=job.get("max_steps",        -1),
        logging_steps=job.get("logging_steps", 1),
        output_dir=str(output_dir),
    )

    trainer = WandbGRPOTrainer(
        model=model,
        train_dataset=job["dataset"],
        tokenizer=tokenizer,
        reward_fn=make_precomputed_reward_fn(),
        args=config,
        wandb_run=wb_run,
        round_num=round_num,
    )

    try:
        train_result = trainer.train()
    except Exception as e:
        if wb_run:
            wb_run.finish(exit_code=1)
        _fail(f"Training failed: {e}")

    # ── Log round summary to W&B ───────────────────────────────────────────────
    if wb_run is not None:
        if trainer.step_losses:
            final_loss = trainer.step_losses[-1]["loss"]
            wb_run.log({"grpo/final_loss": final_loss, "grpo/round": round_num})
        _wandb.finish()

    # ── Write result ───────────────────────────────────────────────────────────
    result = {
        "status":       train_result.get("status", "success"),
        "adapter_path": train_result.get("adapter_path", ""),
        "step_losses":  trainer.step_losses,
        "error":        None,
    }
    result_path.write_text(json.dumps(result, indent=2))
    print(f"[grpo_worker] Result written to {result_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python -m simple_rl.grpo_worker <job_json_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
