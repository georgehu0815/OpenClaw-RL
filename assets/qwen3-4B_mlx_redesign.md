# Qwen3-4B Training — CUDA Stack → MLX Stack Migration

> Redesign of `slime/docs/en/examples/qwen3-4B.md` for the MLX-native stack.
> Replaces: **Slime → Ray → SGLang → Megatron-LM**
> With: **oMLX Policy Slot | oMLX PRM Slot | mlx-tune**

---

## Stack Mapping

| CUDA Role | CUDA Component | MLX Replacement |
|-----------|---------------|-----------------|
| Distributed scheduler | Ray | Python `asyncio` (built into oMLX) |
| Rollout inference | SGLang | **oMLX — Policy Slot** |
| Reward / teacher scoring | SGLang (separate server) | **oMLX — PRM Slot** |
| Model training | Megatron-LM (TP=4, FSDP) | **mlx-tune GRPOTrainer** |
| Checkpoint format | Megatron distributed tensors | MLX safetensors (HF-native) |
| GPU cluster | 8× H100 | Apple Silicon (16–128 GB unified memory) |

---

## Environment Setup

### CUDA version
```bash
docker pull slimerl/slime:latest
git clone https://github.com/THUDM/slime.git
pip install -e . --no-deps
```

### MLX version
```bash
# Install mlx-tune (training)
cd /Volumes/ExternalSSD/train/mlx-tune
pip install -e ".[all]"

# Install oMLX (inference + rollout)
cd /Volumes/ExternalSSD/serve/omlx
pip install -e ".[dev]"

# Install OpenClaw-RL dependencies
cd /Volumes/ExternalSSD/train/OpenClaw-RL
pip install -r requirements.txt
```

---

## Model Download

### CUDA version
```bash
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
# Then convert HF → Megatron distributed format (required):
python tools/convert_hf_to_torch_dist.py --hf-checkpoint /root/Qwen3-4B --save /root/Qwen3-4B_torch_dist
```

### MLX version
```bash
# Download pre-quantized MLX format — no conversion needed
huggingface-cli download mlx-community/Qwen3-4B-4bit

# Or use the full BF16 model if memory allows
huggingface-cli download Qwen/Qwen3-4B
```

> **Key difference:** Megatron requires a separate conversion step to its own distributed tensor format. oMLX reads HF safetensors directly — no conversion.

---

## Architecture: The Training Loop

### CUDA version
```
Prompt Dataset
      │
      ▼
  SGLang Server ──────────────────────── rollout inference (TP=2)
      │ trajectories
      ▼
  Slime Rollout Buffer ─── reward_func ──► deepscaler reward model
      │ scored batches
      ▼
  Megatron-LM Actor (TP=4) ──────────── GRPO gradient update
      │ updated weights
      ▼
  SGLang (weight sync) ◄───────────────── hot reload
```

### MLX version
```
Prompt Dataset / Live Conversations
      │
      ▼
  oMLX — Policy Slot ─────────────────── rollout inference (continuous batching)
      │ trajectories
      ▼
  Rollout Buffer (asyncio queue) ──────── passes turns to PRM
      │                   │
      │         ┌─────────▼──────────┐
      │         │ oMLX — PRM Slot    │  reward scoring (majority vote × m)
      │         └─────────┬──────────┘
      │                   │ reward signals
      ◄───────────────────┘
      │ scored batches
      ▼
  mlx-tune GRPOTrainer ───────────────── GRPO gradient update (LoRA or full)
      │ updated LoRA weights
      ▼
  oMLX hot-reload ◄───────────────────── no server restart needed
```

---

## Parameter Groups — Side by Side

### MODEL_ARGS

| CUDA | MLX |
|------|-----|
| `source scripts/models/qwen3-4B.sh` — Megatron needs explicit architecture args (hidden size, layers, heads) | Not needed — mlx-tune reads architecture from HF `config.json` automatically |
| Must match `--rotary-base` exactly or training diverges | RoPE config read from model checkpoint, no manual setting |

```python
# MLX equivalent — just load the model
model, tokenizer = FastLanguageModel.from_pretrained(
    "mlx-community/Qwen3-4B-4bit",
    max_seq_length=32768,
    load_in_4bit=True,
)
```

---

### CKPT_ARGS

| Arg | CUDA | MLX |
|-----|------|-----|
| Base model | HF checkpoint → converted to Megatron dist format | HF safetensors loaded directly by mlx-tune |
| Reference model | `--ref-load` separate Megatron checkpoint | Same model in memory; mlx-tune handles KL internally |
| Save format | Megatron sharded tensors (large, multi-file) | LoRA adapters only (~50MB) or merged HF format |
| Save interval | Every 20–100 steps | Every N steps via `GRPOConfig(save_steps=N)` |

```python
# MLX checkpoint config
args = GRPOConfig(
    output_dir="./checkpoints/qwen3-4b-grpo",
    save_steps=20,
)
```

---

### ROLLOUT_ARGS

| Arg | CUDA (Slime) | MLX (oMLX + mlx-tune) |
|-----|-------------|----------------------|
| Prompt dataset | `--prompt-data dapo-math-17k.jsonl` | Pass as `train_dataset` to GRPOTrainer, or use live rollout queue |
| Reward model | `--rm-type deepscaler` (built-in) | Custom `reward_func` passed to GRPOTrainer |
| Rollout function | Built-in Slime rollout | `generate_rollout_openclaw_*` or custom async function |
| Batch size | `--rollout-batch-size 32` | `per_device_train_batch_size` × `gradient_accumulation_steps` |
| Samples per prompt | `--n-samples-per-prompt 8` | `num_generations=8` in GRPOConfig |
| Max response | `--rollout-max-response-len 8192` | `max_completion_length=8192` |
| Temperature | `--rollout-temperature 1.0` | Set on oMLX server: `--generation-temperature 1.0` |

```python
# MLX rollout config in GRPOConfig
args = GRPOConfig(
    num_generations=8,                  # group size (was n-samples-per-prompt)
    max_completion_length=8192,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,      # effective batch = 32
)
```

---

### EVAL_ARGS

| CUDA | MLX |
|------|-----|
| `--eval-interval 5` — evaluates every 5 rollout steps | Run eval loop manually or use `GRPOConfig(eval_steps=5)` |
| `--eval-prompt-data aime-2024.jsonl` | Pass eval dataset to GRPOTrainer |
| `--n-samples-per-eval-prompt 16` | Set `num_generations=16` for eval calls |
| `--eval-max-response-len 16384` | Override `max_completion_length` for eval |

---

### PERF_ARGS (Parallelism)

| CUDA (Megatron) | MLX (mlx-tune) |
|----------------|----------------|
| `--tensor-model-parallel-size 2` — shards each layer across 2 GPUs | Not applicable — single unified memory pool |
| `--sequence-parallel` | Not applicable |
| `--recompute-granularity full` — recomputes activations to save VRAM | `--gradient-checkpointing` in GRPOConfig |
| `--max-tokens-per-gpu 9216` — dynamic batching per GPU | `max_seq_length` controls context window |
| Requires multi-GPU coordination via NCCL | Single process, MLX handles memory management |

```python
# MLX performance config
args = GRPOConfig(
    gradient_checkpointing=True,        # was recompute-granularity full
    max_completion_length=8192,
)
```

---

### GRPO_ARGS

Identical algorithm — same formula, same hyperparameters recommended:

```python
# MLX GRPOConfig
args = GRPOConfig(
    # Advantage: A = (r - mean) / std  — same as CUDA
    num_generations=8,

    # PPO clip — same values
    epsilon=0.2,                        # was --eps-clip 0.2
    # eps-clip-high (0.28) not yet in mlx-tune — standard epsilon used

    # KL and entropy — both disabled, same as CUDA default
    # kl_coef=0.0 (default)
    # entropy_coef=0.0 (default)

    learning_rate=1e-6,                 # same as Slime example
)
```

---

### OPTIMIZER_ARGS

| Arg | CUDA | MLX |
|-----|------|-----|
| Optimizer | Adam | Adam (default in mlx-tune) |
| LR | `1e-6` | `learning_rate=1e-6` |
| LR schedule | constant | `lr_scheduler_type="constant"` |
| Weight decay | `0.1` | `weight_decay=0.1` |
| β1, β2 | `0.9, 0.98` | `adam_beta1=0.9, adam_beta2=0.98` |
| CPU offload | `--optimizer-cpu-offload` (needed on H100 for full FT) | Not needed — unified memory handles this transparently |

---

### SGLANG_ARGS → oMLX Server Args

| CUDA (SGLang) | MLX (oMLX) |
|--------------|-----------|
| `--rollout-num-gpus-per-engine 2` (TP=2) | Not applicable — oMLX uses unified memory, no TP |
| `--sglang-mem-fraction-static 0.7` | `--max-model-memory 24GB` |
| Separate SGLang process on dedicated GPUs | oMLX runs in the same process/memory space |
| Requires weight sync after each training step | `--lora-hot-reload` — oMLX picks up new adapters automatically |

```bash
# MLX oMLX server launch
omlx serve \
  --model-dir ~/.cache/huggingface/hub \
  --max-model-memory 24GB \
  --paged-ssd-cache-dir ~/.omlx/cache \
  --hot-cache-max-size 8GB \
  --port 30000 \
  --api-key "${API_KEY}"
```

---

## Advanced Features — MLX Equivalents

### Dynamic Sampling (DAPO-style)

| CUDA | MLX |
|------|-----|
| `--over-sampling-batch-size 64` | Over-sample in the async rollout queue before filling the training batch |
| `--dynamic-sampling-filter-path check_reward_nonzero_std` | Filter in the custom rollout function: discard groups where `rewards.std() == 0` |
| Slime handles abort + requeue internally | Implement in `generate_rollout_*` — skip prompt if all rewards are identical |

```python
# MLX dynamic sampling filter (in custom rollout function)
def has_nonzero_reward_std(rewards: list[float]) -> bool:
    import statistics
    return len(set(rewards)) > 1  # at least two different reward values
```

---

### Partial Rollout

| CUDA | MLX |
|------|-----|
| `--partial-rollout` saves aborted SGLang requests to buffer | Store partial KV cache entries in oMLX's tiered cache (NVMe cold tier) |
| `--buffer-filter-path pop_first` | Custom buffer management in the async rollout queue |

oMLX's **prefix cache with copy-on-write** naturally handles partial reuse — interrupted sequences keep their KV blocks on NVMe and can be resumed without full re-prefill.

---

### Decoupled Training and Inference

| CUDA | MLX |
|------|-----|
| Remove `--colocate`, set `--rollout-num-gpus 6` | oMLX and mlx-tune already run as separate processes sharing unified memory |
| Ray manages GPU assignment across processes | OS unified memory scheduler handles allocation transparently |
| Risk: SGLang CUDA graph concurrency limit (160) | Not applicable — oMLX uses continuous batching, not CUDA graphs |

---

### Asynchronous Training

| CUDA | MLX |
|------|-----|
| Switch `train.py` → `train_async.py` | oMLX's async rollout queue is always async — training and rollout overlap by default |
| Ray `.remote` / `ray.get` for coordination | Python `asyncio.gather` in rollout buffer |
| GPU idle time between rollout and training steps | Eliminated — mlx-tune training runs while oMLX generates next batch |

---

## Complete MLX Training Script (Qwen3-4B, Math GRPO)

```python
from mlx_tune import FastLanguageModel, GRPOTrainer, GRPOConfig
from datasets import load_dataset

# Load model — no conversion needed
model, tokenizer = FastLanguageModel.from_pretrained(
    "mlx-community/Qwen3-4B-4bit",
    max_seq_length=32768,
    load_in_4bit=True,
)

# Add LoRA adapters (replaces Megatron full FT for low-memory setups)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# Load dataset (same dapo-math-17k)
dataset = load_dataset("zhuzilin/dapo-math-17k")["train"]

# Reward function (replaces --rm-type deepscaler)
def math_reward_func(completions, labels, **kwargs):
    rewards = []
    for completion, label in zip(completions, labels):
        answer = extract_answer(completion)
        rewards.append(1.0 if answer == label else 0.0)
    return rewards

# GRPO config — mirrors Slime's GRPO_ARGS + OPTIMIZER_ARGS
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_funcs=[math_reward_func],
    args=GRPOConfig(
        num_generations=8,              # n-samples-per-prompt
        learning_rate=1e-6,             # same as Slime example
        adam_beta1=0.9,
        adam_beta2=0.98,
        weight_decay=0.1,
        lr_scheduler_type="constant",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,  # effective batch ≈ 32
        max_completion_length=8192,
        gradient_checkpointing=True,    # replaces recompute-granularity full
        output_dir="./checkpoints/qwen3-4b-grpo",
        save_steps=20,
        max_steps=3000,                 # num-rollout equivalent
    ),
    train_dataset=dataset,
)

trainer.train()
```

---

## Hardware Requirements

| Model | CUDA Stack | MLX Stack |
|-------|-----------|-----------|
| Qwen3-4B (4-bit) | 1× H100 80GB (minimal) | **16 GB Mac** (M1/M2/M3/M4 base) |
| Qwen3-4B (BF16 full FT) | 4× H100 (TP=4) | **64 GB Mac** (M2/M3/M4 Max) |
| Qwen3-8B (4-bit) | 1–2× H100 | **24 GB Mac** (M1/M2/M3/M4 Pro) |
| Qwen3-32B (4-bit) | 4× H100 | **64 GB Mac** (M2/M3/M4 Max/Ultra) |
