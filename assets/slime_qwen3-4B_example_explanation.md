# slime/docs/en/examples/qwen3-4B.md — Explanation

This is the **official Slime framework quickstart guide** for training Qwen3-4B with GRPO on 8×H100 GPUs. It documents the upstream `slime` training loop that OpenClaw-RL is built on top of — this is the vanilla math-reasoning RL example, not the OpenClaw-specific version.

---
math reasoning rl->slimr rl traing -> grpo on gpu
## What is Slime?

[Slime](https://github.com/THUDM/slime) (from THUDM) is the async RL training framework at the core of OpenClaw-RL. It orchestrates the loop between:
- **Megatron-LM** — model training (actor + reference)
- **SGLang** — fast rollout inference
- **Ray** — distributed scheduling between training and inference nodes

OpenClaw-RL extends Slime by adding custom rollout functions (`openclaw_*_rollout`), custom reward functions, PRM integration, and OPD loss.

---

## Setup Steps

### 1. Docker Image
Uses the pre-built `slimerl/slime:latest` image — contains Megatron-LM, SGLang, and all CUDA dependencies.

### 2. Downloads
```
Model:   Qwen/Qwen3-4B          → /root/Qwen3-4B
Train:   dapo-math-17k dataset  → /root/dapo-math-17k   (17k math problems)
Eval:    aime-2024 dataset      → /root/aime-2024        (competition math)
```

### 3. Checkpoint Conversion
```bash
python tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B_torch_dist
```
Converts HuggingFace safetensors → Megatron's distributed tensor format. Required before training — Megatron cannot read HF checkpoints directly.

---

## Script Parameter Groups

### MODEL_ARGS
Loaded from `scripts/models/qwen3-4B.sh` — Megatron architecture config (hidden size, num layers, num heads, etc.). Unlike HuggingFace, Megatron cannot infer architecture from the checkpoint, so it must be passed explicitly.

> ⚠️ `--rotary-base` must match the specific model variant. Even same-architecture models (e.g. Qwen3-4B vs Qwen3.5-4B) may use different RoPE bases.

---

### CKPT_ARGS
| Arg | Purpose |
|-----|---------|
| `--hf-checkpoint` | HF format — used by SGLang for inference and tokenizer |
| `--ref-load` | Reference model for KL divergence (frozen, Megatron format) |
| `--load` | Resume actor from this checkpoint |
| `--save` | Save actor checkpoints here |
| `--save-interval 20` | Save every 20 training steps |

---

### ROLLOUT_ARGS
| Arg | Value | Meaning |
|-----|-------|---------|
| `--prompt-data` | `dapo-math-17k.jsonl` | Training prompts |
| `--rm-type` | `deepscaler` | Built-in reward model for math correctness |
| `--num-rollout` | 3000 | Total rollout rounds before training ends |
| `--rollout-batch-size` | 32 | Prompts per rollout round |
| `--n-samples-per-prompt` | 8 | Responses generated per prompt (GRPO group size) |
| `--rollout-temperature` | 1.0 | High temperature for diverse sampling |
| `--num-steps-per-rollout` | 1 | 1 gradient update per rollout batch |

Each rollout produces `32 × 8 = 256` samples, which GRPO uses to compute group advantages.

---

### EVAL_ARGS
Evaluates on AIME-2024 every 5 training steps:
- 16 samples per prompt (more than training's 8 — higher accuracy estimate)
- Max response 16K tokens (2× training's 8K — lets the model reason longer)
- `top-p 1.0` — unrestricted nucleus sampling

---

### PERF_ARGS (Megatron Parallelism)
```
Tensor parallel:    2 GPUs per layer (TP=2)
Sequence parallel:  on
Pipeline parallel:  1 (no stages)
```
- Full activation recomputation (`recompute-granularity full`) — trades FLOPs for memory
- `--use-dynamic-batch-size` + `--max-tokens-per-gpu 9216` — packs variable-length sequences up to 9216 tokens/GPU instead of fixed micro-batch sizes

> Note: OpenClaw-RL's full FT scripts use TP=4 and max-tokens-per-gpu=32768 (3.5× higher) to accommodate longer conversation contexts.

---

### GRPO_ARGS
Standard GRPO configuration:
```
advantage = (reward - mean(group_rewards)) / std(group_rewards)
```
| Arg | Value | Note |
|-----|-------|------|
| `--eps-clip` | 0.2 | PPO lower clip bound |
| `--eps-clip-high` | 0.28 | PPO upper clip bound (asymmetric) |
| `--kl-loss-coef` | 0.0 | KL penalty disabled |
| `--entropy-coef` | 0.0 | Entropy bonus disabled |

---

### OPTIMIZER_ARGS
Adam with constant LR schedule:
- LR `1e-6` — **10× lower than OpenClaw scripts** (math RL is more stable than conversation RL)
- Weight decay `0.1`, β1=0.9, β2=0.98
- No CPU offload (unlike OpenClaw full FT scripts)

---

### SGLANG_ARGS
- `--rollout-num-gpus-per-engine 2` — SGLang uses TP=2 for inference
- `--sglang-mem-fraction-static 0.7` — 70% GPU memory for KV cache

---

## Advanced Features

### Dynamic Sampling (DAPO-style)
Addresses the **"all correct or all wrong" problem** in GRPO — if a prompt produces all correct or all wrong answers, the reward std is 0 and the advantage is undefined.

**How it works:**
1. Over-sample `over_sampling_batch_size=64` prompts instead of 32
2. For each prompt's 8 responses, filter using `check_reward_nonzero_std`
3. Keep only prompts where responses have non-zero reward variance
4. Stop once 32 valid prompts are collected

```python
def check_reward_nonzero_std(args, samples, **kwargs):
    rewards = [sample.reward for sample in samples]
    return torch.tensor(rewards).std() > 0.0  # discard all-same reward groups
```

---

### Partial Rollout
During dynamic sampling, many in-progress requests are aborted when the batch is full. `--partial-rollout` saves these incomplete generations to a buffer and reuses them in the next rollout — reducing wasted inference compute.

---

### Decoupled Training and Inference
By default (`--colocate`), training and inference share the same 8 GPUs, timesliced. For better throughput, split them:
```bash
--actor-num-gpus-per-node 2   # 2 GPUs for training
--rollout-num-gpus 6          # 6 GPUs for inference
```
Training and inference now run on dedicated resources.

> ⚠️ High SGLang concurrency (>160 requests) can exceed CUDA graph limits. Fix with `--sglang-server-concurrency 160` or `--sglang-cuda-graph-bs`.

---

### Asynchronous Training (`train_async.py`)
Switching from `train.py` → `train_async.py` enables **pipeline parallelism between rollout and training**:
- While training on batch N, inference is generating batch N+1
- Uses Ray's `.remote` / `ray.get` for async coordination
- Eliminates GPU idle time from training↔inference synchronization

OpenClaw-RL uses `train_async.py` in all its scripts.

---

## Relationship to OpenClaw-RL Scripts

| Feature | This guide (vanilla Slime) | OpenClaw-RL scripts |
|---------|---------------------------|---------------------|
| Task | Math reasoning (DAPO-17k) | Live OpenClaw conversations |
| Reward | `deepscaler` (math correctness) | Custom `openclaw_*_api_server.reward_func` |
| Rollout fn | Built-in | `openclaw_*_rollout.generate_rollout_*` |
| Loss | GRPO only | GRPO, OPD, or Combined |
| PRM | Not used | Enabled (`--prm-enable`) |
| TP size | 2 | 4 |
| LR | 1e-6 | 1e-5 |
| Context | 8K train / 16K eval | 32K |
