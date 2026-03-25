# run_qwen35_4b_openclaw_combine_lora.sh — Explanation

This is the **original CUDA/Linux training script** (not the MLX version) — it's the combined RL + OPD launcher for Qwen3.5-4B on a multi-GPU machine.

---

## 1. Cleanup (lines 7–16)
Kills any leftover `sglang`, `ray`, and `python` processes before starting fresh. Skippable via `SKIP_CLUSTER_CLEANUP=1`.

---

## 2. GPU Allocation (lines 24–33)
Splits 4 GPUs across 3 roles:

| Role | Default GPUs |
|------|-------------|
| Actor (training) | 2 |
| Rollout (inference) | 1 |
| PRM (reward scorer) | 1 |

Exits with an error if the sum exceeds `NUM_GPUS`.

---

## 3. Paths & Model Config (lines 44–60)
- `HF_CKPT` — base Qwen3.5-4B model checkpoint
- `REF_LOAD` — reference model for KL penalty (defaults to same as base)
- `SAVE_CKPT` — where to write training checkpoints
- `PRM_MODEL_PATH` — which model acts as the PRM judge
- Context length 32K, 85% GPU memory reserved for static KV cache

---

## 4. Rollout Config (lines 73–86)
- Runs `generate_rollout_openclaw_combine` — the combined rollout function
- Batch size 16, 1 sample per prompt (online, not grouped)
- Max response 8K tokens, temperature 0.6
- Virtually infinite rollouts (`--num-rollout 100000000`) — trains until you stop it

---

## 5. Combined Loss (lines 94–105)
The core of the script — uses `combine_loss.combine_loss_function`:

```
A_combined = w_rl × GRPO_advantage + w_opd × OPD_advantage
```

- Both weights default to `1.0` (equal blend)
- GRPO clip range: [0.2, 0.28]
- KL loss coefficient: `0.0` (disabled — pure combined signal)
- Entropy coefficient: `0.0`

---

## 6. LoRA Config (lines 116–121)
- Rank 16, alpha 32
- Targets all 7 projection layers: `q/k/v/o_proj` + `gate/up/down_proj` (MLP too)
- Much lower memory than full fine-tuning

---

## 7. PRM Setup (lines 137–145)
- Enables the PRM judge on its own dedicated GPU
- `PRM_M=1` — 1 evaluation per step (increase for majority voting)
- Temperature 0.6, max 4096 tokens per evaluation

---

## 8. Ray + SGLang Launch (lines 170–201)
- Starts a local Ray head node with all GPUs
- Submits the job to `slime/train_async.py` — the async RL training loop
- Backend: **FSDP** (not Megatron TP — lower GPU requirement)
- SGLang serves the rollout model; mlx-tune equivalent here is PyTorch+FSDP

---

## Key Difference from the MLX Version
This script is the **original CUDA stack** — Ray + SGLang + FSDP instead of oMLX + mlx-tune. The logic (combined RL + OPD, PRM scoring, rollout buffer) is identical, but it requires a Linux GPU machine. The MLX version in `GUIDE-MLX.md` runs the same algorithm on a Mac with unified memory.
