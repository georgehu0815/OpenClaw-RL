# run_qwen35_4b_openclaw_opd.sh — Explanation

This is the **OPD-only** (On-Policy Distillation) full fine-tuning script for Qwen3.5-4B, using Megatron-LM with tensor parallelism on 8 GPUs. It trains the student model to mimic a teacher model's token distributions, using live OpenClaw conversations as the data source — with no GRPO advantage signal, only distillation.

---

## vs. Combined and LoRA scripts — Key Differences

| | This script (OPD only) | Combined (full FT) | LoRA version |
|---|---|---|---|
| Loss signal | **OPD only** (teacher KL) | RL + OPD blended | RL + OPD blended |
| Training backend | Megatron (TP=4) | Megatron (TP=4) | FSDP |
| GPU count | 8 | 8 | 4 |
| Weight updates | All parameters | All parameters | LoRA adapters only |
| Rollout function | `openclaw_opd_rollout` | `openclaw_combine_rollout` | `openclaw_combine_rollout` |
| Custom server | `openclaw_opd_api_server` | `openclaw_combine_api_server` | `openclaw_combine_api_server` |
| RoPE base | **5,000,000** | 10,000,000 | (not set) |
| PYTHONPATH | No `openclaw-opd` extra path | Includes `openclaw-opd` | Includes `openclaw-opd` |

---

## 1. Cleanup (lines 6–15)
Force-kills `sglang`, `ray`, and `python` before starting. Skip with `SKIP_CLUSTER_CLEANUP=1`.

---

## 2. GPU Allocation (lines 23–32)
Splits 8 GPUs across 3 roles:

| Role | Default GPUs |
|------|-------------|
| Actor (training) | 4 |
| Rollout (SGLang inference) | 2 |
| PRM (reward/teacher scorer) | 2 |

---

## 3. Model Config Source (line 42)
```bash
source "${SLIME_ROOT}/scripts/models/qwen3.5-4B.sh"
```
Loads Megatron model architecture args (hidden size, layers, attention heads, etc.) from a shared definition file.

---

## 4. Paths & Server Config (lines 44–62)
- `HF_CKPT` — base Qwen3.5-4B checkpoint (defaults to local `models/` dir)
- `REF_LOAD` — reference model (same as base by default)
- `PRM_MODEL_PATH` — the **teacher model** used for distillation (defaults to same base model)
- `TP=2` — SGLang rollout engine uses 2-GPU tensor parallelism
- Context 32K, 80% GPU memory static allocation
- `OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY=1` — caps concurrent teacher log-prob queries

---

## 5. Checkpoint Args (lines 64–71)
- `--megatron-to-hf-mode bridge` — Megatron ↔ HuggingFace weight conversion
- `--rotary-base 5000000` — **5M RoPE base**, half of the combined script's 10M. Affects how positional encoding scales with sequence length.
- Saves every 100 steps

---

## 6. Rollout Config (lines 73–86)
- Uses `openclaw_opd_rollout.generate_rollout_openclaw_opd` — the OPD-specific rollout function
- Batch 16, temperature 0.6, max 8K response / 32K context
- Runs indefinitely until stopped

---

## 7. The Core Difference — OPD Loss (lines 105–111)

This is what makes this script different from all others:

```python
--advantage-estimator on_policy_distillation
```

Instead of GRPO's `A = (r − mean) / std`, the loss is a **token-level KL divergence** between teacher and student:

```
loss = KL(teacher_logits || student_logits)
     = log_p_teacher(t) − log_p_student(t)   # token-level signal
```

- No GRPO clipping, no reward normalization — purely distillation
- `--kl-loss-coef 0.0` — the separate KL regularization term is disabled (the OPD loss *is* the KL)
- `--entropy-coef 0.0` — no entropy bonus

The teacher's log-probs come from the PRM slot (running the teacher model), queried concurrently with rollout generation.

---

## 8. Megatron Parallelism (lines 88–103)
Identical to the combined full FT script:
```
Tensor parallel:    4 GPUs per layer
Sequence parallel:  on
Pipeline parallel:  1 (no stages)
```
Full activation recomputation enabled. `max-tokens-per-gpu 32768`.

---

## 9. Optimizer (lines 113–123)
Same as combined full FT — CPU offload of optimizer states to save GPU memory:
- `--optimizer-cpu-offload`
- `--overlap-cpu-optimizer-d2h-h2d`
- `--use-precision-aware-optimizer`
- LR `1e-5`, weight decay `0.1`, constant schedule

---

## 10. Custom API Server (lines 149–152)
```bash
--custom-generate-function-path openclaw_opd_api_server.generate
--custom-rm-path openclaw_opd_api_server.reward_func
```
Uses the OPD-specific server (`openclaw_opd_api_server`), not the combine server. This server handles querying the teacher model for log-probs and computing the distillation reward signal.

---

## 11. PRM / Teacher Setup (lines 139–147)
The 2 PRM GPUs here serve a dual role — they run the **teacher model** for distillation log-probs, not just a binary reward scorer:
- 2 GPUs, 2-GPU tensor parallel per engine
- `PRM_M=1` — 1 teacher evaluation per step
- Max 8192 tokens per evaluation

---

## 12. Ray + Megatron Launch (lines 180–209)
- Starts Ray with 8 GPUs
- `PYTHONPATH` includes `Megatron-LM/` and `${SCRIPT_DIR}` — but **not** `openclaw-opd` separately (unlike the combined script), since this script *is* already inside `openclaw-opd/`
- Submits `slime/train_async.py` with the OPD args

---

## When to Use This Script

| Goal | Script |
|------|--------|
| Teacher-guided distillation only, max quality | **This script** |
| Blend distillation + RL reward | Combined full FT script |
| Same but fewer GPUs | Combined LoRA script |
| Mac / Apple Silicon | MLX stack in `GUIDE-MLX.md` |

**OPD-only is best when:** you have a strong teacher model and want the student to closely match its behavior, without the variance of a binary reward signal. The combined method adds RL on top for cases where you also have implicit feedback (thumbs up/down).
