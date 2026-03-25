# run_qwen35_4b_openclaw_combine.sh — Explanation

This is the **full fine-tuning** variant of the combined RL + OPD script, using **Megatron-LM with tensor parallelism** — the heavier, higher-throughput sibling of the LoRA version. Requires 8 GPUs instead of 4.

---

## vs. the LoRA version — Key Differences

| | This script (full FT) | LoRA version |
|---|---|---|
| Training backend | **Megatron-LM** (TP=4) | FSDP |
| GPU count | **8** | 4 |
| Weight updates | **All parameters** | ~1% (LoRA adapters only) |
| Checkpoint interval | Every 100 steps | Every 1 step |
| Memory strategy | CPU optimizer offload + overlap | Gradient checkpointing |
| Model config | Loaded from `qwen3.5-4B.sh` | Inline args only |
| PRM max tokens | 8192 | 4096 |

---

## 1. Cleanup (lines 6–15)
Same as the LoRA version — force-kills `sglang`, `ray`, and `python` before starting. Skip with `SKIP_CLUSTER_CLEANUP=1`.

---

## 2. GPU Allocation (lines 23–32)
Splits 8 GPUs across 3 roles:

| Role | Default GPUs |
|------|-------------|
| Actor (training) | 4 |
| Rollout (SGLang inference) | 2 |
| PRM (reward scorer) | 2 |

Exits if the sum exceeds `NUM_GPUS`.

---

## 3. Model Config Source (line 42)
```bash
source "${SLIME_ROOT}/scripts/models/qwen3.5-4B.sh"
```
Loads Megatron-specific model architecture args (hidden size, num layers, num heads, etc.) from a shared model definition file — unlike the LoRA script which has no such import.

---

## 4. Paths & Server Config (lines 44–64)
- `HF_CKPT` — base Qwen3.5-4B checkpoint
- `REF_LOAD` — reference model for KL (defaults to base)
- `SAVE_CKPT` — checkpoint output directory
- `TP=2` — SGLang rollout engine uses tensor parallelism across 2 GPUs
- Context 32K, 80% GPU memory for static KV cache
- Combined loss weights: `W_RL=1.0`, `W_OPD=1.0` (equal blend, tunable)

---

## 5. Checkpoint Args (lines 66–73)
- `--megatron-to-hf-mode bridge` — converts between Megatron and HuggingFace weight formats
- `--rotary-base 10000000` — Qwen3.5 uses a large RoPE base (10M vs standard 10K)
- Saves every **100 steps** (vs every 1 step in LoRA version — full weights are large)

---

## 6. Rollout Config (lines 75–88)
Identical to the LoRA version:
- `generate_rollout_openclaw_combine` rollout function
- Batch 16, temperature 0.6, max 8K response tokens, 32K context
- Runs indefinitely until stopped

---

## 7. Megatron Parallelism (lines 90–105)
The key difference from the LoRA script — full Megatron tensor parallelism:
```
Tensor parallel size:    4  (splits each layer across 4 GPUs)
Sequence parallel:       on (splits activations across TP group)
Pipeline parallel:       1  (no pipeline stages)
Context parallel:        1
Expert parallel:         1
```
- **Full activation recomputation** (`recompute-granularity full`) trades compute for memory
- `max-tokens-per-gpu 32768` — 4× higher than the LoRA version (more GPU memory available)
- `log-probs-chunk-size 1024` — chunked log-prob computation to avoid OOM

---

## 8. Combined Loss (lines 107–118)
Identical formula to the LoRA version:
```
A_combined = w_rl × GRPO_advantage + w_opd × OPD_advantage
```
- GRPO clip: [0.2, 0.28]
- KL loss disabled (`coef=0.0`)
- Entropy regularization disabled

---

## 9. Optimizer (lines 120–130)
Adam with memory-saving features not present in the LoRA version:
- `--optimizer-cpu-offload` — moves optimizer states (momentum, variance) to CPU RAM
- `--overlap-cpu-optimizer-d2h-h2d` — overlaps CPU↔GPU transfers with compute
- `--use-precision-aware-optimizer` — mixed-precision optimizer for BF16 stability
- LR `1e-5`, weight decay `0.1`, constant schedule

---

## 10. Misc Training Flags (lines 161–167)
- No dropout (attention or hidden)
- AllReduce gradients accumulated in FP32 for stability
- Attention softmax in FP32
- FlashAttention backend

---

## 11. PRM Setup (lines 146–154)
Same as LoRA version but with 2 GPUs per engine and up to 8192 tokens per evaluation (vs 4096). Allows evaluating longer reasoning chains.

---

## 12. Ray + Megatron Launch (lines 187–218)
- Starts Ray head with 8 GPUs
- `PYTHONPATH` includes `Megatron-LM/` — required for Megatron backend (absent in LoRA script)
- Submits `slime/train_async.py` with `--train-backend` defaulting to Megatron (no `--train-backend fsdp` flag)
- Passes `${MODEL_ARGS[@]}` sourced from the Qwen3.5-4B model definition

---

## When to Use Which Script

| Scenario | Use |
|----------|-----|
| 8+ GPU server, max model quality | **This script** (full FT, Megatron) |
| 4-GPU server or limited VRAM | LoRA version (FSDP) |
| Mac / Apple Silicon | Neither — use the MLX stack in `GUIDE-MLX.md` |
