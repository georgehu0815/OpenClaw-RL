# OpenClaw-RL — System Guide (MLX / Apple Silicon Stack)

> **Train a personalized AI agent simply by talking to it — natively on Apple Silicon.**
>
> This guide replaces the CUDA/Linux stack (Megatron-LM + SGLang) with the MLX-native stack (**mlx-tune** for training, **oMLX** for inference), running fully on macOS with no CUDA dependency.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Repository Layout](#3-repository-layout)
4. [Dependencies & Setup](#4-dependencies--setup)
5. [Training Methods](#5-training-methods)
6. [Track 1 — Personal Agent (OpenClaw)](#6-track-1--personal-agent-openclaw)
7. [Track 2 — General Agentic RL](#7-track-2--general-agentic-rl)
8. [Infrastructure Layer (mlx-tune + oMLX)](#8-infrastructure-layer-mlx-tune--omlx)
9. [Key Configuration Knobs](#9-key-configuration-knobs)
10. [Deployment Options](#10-deployment-options)
11. [Evaluation](#11-evaluation)
12. [Extending the Framework](#12-extending-the-framework)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. System Overview

OpenClaw-RL runs on a **fully MLX-native stack** on Apple Silicon Macs, replacing CUDA dependencies entirely:

| Original (CUDA) | Replacement (MLX) | Purpose |
|-----------------|-------------------|---------|
| Megatron-LM | **mlx-tune** | Distributed LLM training → Mac-native fine-tuning |
| SGLang | **oMLX** | CUDA inference engine → MLX inference server |
| Ray (cluster) | Unified memory | Multi-node coordination → single Mac up to 512GB |

| Track | Goal | Methods |
|-------|------|---------|
| **Track 1** — Personal Agent | Train from live OpenClaw conversations | Binary RL (GRPO), OPD, Combined |
| **Track 2** — General Agentic RL | Train agents for real-world tasks | GRPO + PRM via mlx-tune for Terminal/GUI/SWE/Tool-call |

**Why Apple Silicon?**

- **No GPU/CUDA setup** — runs on any Mac with M1 or later
- **Unified memory** — up to 512GB shared CPU/GPU on Mac Studio Ultra
- **Quantization built-in** — 4-bit/8-bit models reduce memory footprint significantly
- **oMLX continuous batching** — production-grade throughput without a Linux server
- **mlx-tune ↔ Unsloth compatible** — proto on Mac, optionally scale to cloud with Unsloth

---

## 2. Architecture

### The 4-Component Async Loop (MLX Stack)

```
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│   ┌────────────────┐  multi-turn   ┌───────────────────────────┐ │
│   │  User /        │──conversation▶│  oMLX Server              │ │
│   │  OpenBot App  │               │  (FastAPI, OpenAI-compat) │ │
│   └────────────────┘               │  continuous batching      │ │
│                                    │  tiered KV cache          │ │
│                                    └────────────┬──────────────┘ │
│                                                 │                 │
│                                          trajectories             │
│                                                 │                 │
│   ┌────────────────┐                   ┌────────▼─────────────┐  │
│   │  mlx-tune      │◀──gradient update─│  Rollout Buffer      │  │
│   │  GRPOTrainer   │                   │  (async queue)       │  │
│   │  (MLX native)  │                   └────────┬─────────────┘  │
│   └────────────────┘                            │                 │
│          │                               reward signals           │
│          │ weight sync                          │                 │
│          ▼                       ┌──────────────▼─────────────┐  │
│   ┌────────────────┐             │  PRM / Judge Model         │  │
│   │  Updated Model │────────────▶│  (oMLX engine)             │  │
│   │  (LoRA or full)│             │  majority vote × m         │  │
│   └────────────────┘             └────────────────────────────┘  │
│                                                                   │
│                     [All on one Mac — Unified Memory]             │
└───────────────────────────────────────────────────────────────────┘
```

**Component responsibilities:**

| Component | Role | Technology |
|-----------|------|------------|
| **oMLX Server** | OpenAI-compatible endpoint; serves policy model and PRM | oMLX + MLX |
| **Rollout Buffer** | Collects trajectories, waits for rewards, formats training data | Python async queues |
| **PRM / Judge** | Evaluates turns or full episodes; produces reward signal | oMLX (separate model slot) |
| **mlx-tune Trainer** | Runs GRPO/OPD/SFT gradient updates on the policy | mlx-tune GRPOTrainer |

### Memory Layout on Apple Silicon

```
┌────────────────────────────────────────────────────────┐
│                  Unified Memory (e.g. 64GB)            │
│                                                        │
│  ┌──────────────────┐  ┌──────────────────┐           │
│  │  Policy Model    │  │  PRM / Judge     │           │
│  │  (oMLX hot tier) │  │  (oMLX hot tier) │           │
│  │  e.g. Qwen3-8B   │  │  e.g. Qwen3-4B  │           │
│  │  4-bit ≈ 5GB     │  │  4-bit ≈ 3GB    │           │
│  └──────────────────┘  └──────────────────┘           │
│                                                        │
│  ┌──────────────────────────────────────────────┐     │
│  │  mlx-tune training (LoRA adapters + grads)   │     │
│  │  ≈ 8–16GB depending on batch size            │     │
│  └──────────────────────────────────────────────┘     │
│                                                        │
│  ┌──────────────────────────────────────────────┐     │
│  │  oMLX SSD KV cache (cold tier → NVMe)        │     │
│  │  spills inactive KV blocks to disk            │     │
│  └──────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────┘
```

---

## 3. Repository Layout

```
OpenClaw-RL/
│
├── slime/                      # Async rollout framework (THUDM) — used for queue logic
├── openclaw-rl/                # Track 1 — Binary RL (GRPO)
├── openclaw-opd/               # Track 1 — On-Policy Distillation
├── openclaw-combine/           # Track 1 — Combined RL + OPD
├── openclaw-tinker/            # Track 1 — Cloud deployment (LoRA)
├── openclaw-test/              # End-to-end evaluation suite
│
├── toolcall-rl/                # Track 2 — Math/tool-call (ReTool)
├── terminal-rl/                # Track 2 — Terminal/shell agent
├── gui-rl/                     # Track 2 — Desktop GUI agent (Qwen3-VL)
├── swe-rl/                     # Track 2 — SWE-Bench issue fixing
│
├── extensions/                 # OpenClaw plugin for self-hosted deployment
├── requirements.txt            # Python dependencies
└── GUIDE-MLX.md                # This file
│
# External (local paths):
# /Volumes/ExternalSSD/train/mlx-tune     ← training framework
# /Volumes/ExternalSSD/serve/omlx         ← inference server
```

---

## 4. Dependencies & Setup

### Hardware Requirements

| Model Size | Mac RAM | Chip |
|-----------|---------|------|
| 4B (4-bit) | 16 GB | M1/M2/M3/M4 any |
| 8B (4-bit) | 24 GB | M1/M2/M3/M4 Pro or better |
| 32B (4-bit) | 64 GB | M2/M3/M4 Max / Ultra |
| 70B (4-bit) | 128 GB | M2/M3/M4 Ultra |
| 32B (full BF16) | 128 GB | M3/M4 Ultra |

> Quantized models (4-bit) can fit much more on unified memory than CUDA GPU VRAM.
> Use `mlx-community/` HuggingFace models for pre-quantized MLX safetensors.

### Software Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Runtime |
| PyTorch | 2.9.1 | Deep learning |
| Transformers | 4.57.1 | HuggingFace models |
| **oMLX** | custom | Fast LLM inference + rollout |
| **mlx-tune** | dev_rl branch | LLM training (MLX-native) |
| PEFT | ≥0.12.0 | LoRA support |
| FastAPI | 0.131.0 | API server |
| wandb | 0.25.0 | Experiment tracking |
| mlx | latest | Apple MLX framework |
| mlx-lm | latest | MLX language model utilities |

### Step-by-Step Installation

**Step 1 — Clone the repo**

```bash
git clone https://github.com/Gen-Verse/OpenClaw-RL.git
cd OpenClaw-RL
```

**Step 2 — Create a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Step 3 — Install OpenClaw-RL dependencies**

```bash
pip install -r requirements.txt
```

**Step 4 — Install mlx-tune**

```bash
cd /Volumes/ExternalSSD/train/mlx-tune
pip install -e ".[all]"     # includes GRPO, VLM, audio support
cd -
```

**Step 5 — Install oMLX**

```bash
cd /Volumes/ExternalSSD/serve/omlx
pip install -e ".[dev]"
cd -
```

**Step 6 — Verify oMLX server**

```bash
omlx serve --model-dir ~/models --max-model-memory 32GB
# Navigate to http://localhost:8000/admin to confirm
```

**Step 7 — Download models (MLX format)**

```bash
# Download pre-quantized MLX models from mlx-community
huggingface-cli download mlx-community/Qwen3-4B-4bit
huggingface-cli download mlx-community/Qwen3-8B-4bit
# Or point oMLX at your HuggingFace cache:
omlx serve --model-dir ~/.cache/huggingface/hub
```

---

## 5. Training Methods

### Method Comparison

| Method | Reward Signal | Best For | Trainer |
|--------|--------------|----------|---------|
| **Binary RL** | Scalar +1/−1/0 from evaluator | Implicit feedback (thumbs up/down) | `mlx_tune.GRPOTrainer` |
| **OPD** (Token-level) | Token-level directional: log_teacher − log_student | Explicit feedback with hints | custom MLX loss |
| **OPD** (Top-K) | Reverse KL over top-K teacher distribution | Strong teacher signal | `topk_distillation_loss.py` |
| **Combined** | Both RL and OPD signals simultaneously | Maximum training signal density | `combine_loss.py` |

### GRPO with mlx-tune

```python
from mlx_tune import FastLanguageModel, GRPOTrainer, GRPOConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    "mlx-community/Qwen3-4B-4bit",
    max_seq_length=8192,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_funcs=[my_reward_func],
    args=GRPOConfig(
        num_generations=8,          # group size (n-samples-per-prompt)
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_completion_length=8192,
        output_dir="./checkpoints",
    ),
    train_dataset=dataset,
)
trainer.train()
```

### GRPO Advantage Estimation

```
A_i = (r_i − mean(r)) / (std(r) + ε)
```

Where `r` is the reward across `num_generations` rollouts for the same prompt.

### Process Reward Model (PRM)

The PRM runs as a **second model slot in oMLX**. It evaluates each intermediate step:

- `m` independent generations per step → majority vote
- Reward signal passed back to GRPOTrainer after each turn
- Controlled via `--prm-enable`, `--prm-m`, and `--prm-model-path`

---

## 6. Track 1 — Personal Agent (OpenClaw)

### How It Works

1. User sends a message to the OpenClaw app
2. **oMLX** serves the response from the policy model (continuous batching, tiered KV cache)
3. The rollout worker intercepts the conversation trajectory
4. **oMLX** (PRM slot) evaluates each turn
5. **mlx-tune GRPOTrainer** updates the policy in the background
6. Updated LoRA weights are hot-reloaded into oMLX

### 6.1 Binary RL

**Entry point:** [openclaw-rl/openclaw_api_server.py](openclaw-rl/openclaw_api_server.py)

**Setup:**

```bash
cd openclaw-rl

# Set paths
export HF_CKPT=mlx-community/Qwen3-4B-4bit
export SAVE_CKPT=./checkpoints/qwen3-4b-openclaw-rl
export SGLANG_API_KEY=your-api-key     # used by oMLX too
export PORT=30000
```

**Start oMLX (inference):**

```bash
omlx serve \
  --model-dir ~/.cache/huggingface/hub \
  --max-model-memory 24GB \
  --max-process-memory 80% \
  --paged-ssd-cache-dir ~/.omlx/cache \
  --hot-cache-max-size 8GB \
  --port 30000 \
  --api-key "${SGLANG_API_KEY}"
```

**Start training loop:**

```python
from mlx_tune import FastLanguageModel, GRPOTrainer, GRPOConfig
from openclaw_rollout import generate_rollout_openclaw

model, tokenizer = FastLanguageModel.from_pretrained(
    "mlx-community/Qwen3-4B-4bit",
    max_seq_length=32768,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_funcs=[openclaw_reward_func],
    args=GRPOConfig(
        num_generations=8,
        learning_rate=1e-5,
        max_completion_length=8192,
        output_dir=SAVE_CKPT,
        save_steps=100,
    ),
    train_dataset=rollout_dataset,  # live-populated by rollout worker
)
trainer.train()
```

**LoRA mode (less RAM):**

Use `r=8` and `load_in_4bit=True` to fit on 16GB Mac.

**GPU split on a 64GB Mac:**

| Component | Memory |
|-----------|--------|
| oMLX policy model (Qwen3-8B 4-bit) | ~5 GB |
| oMLX PRM model (Qwen3-4B 4-bit) | ~3 GB |
| mlx-tune training (LoRA + grads) | ~12 GB |
| oMLX SSD KV cache (cold tier) | NVMe |
| OS + Python overhead | ~8 GB |

### 6.2 On-Policy Distillation (OPD)

**Entry point:** [openclaw-opd/openclaw_opd_api_server.py](openclaw-opd/openclaw_opd_api_server.py)

The teacher model runs as a **second model in oMLX** (or a separate oMLX instance on port 30001).

**Token-level OPD:**

```bash
# Start teacher oMLX on a different port
omlx serve --model-dir ~/models --port 30001 --api-key "${TEACHER_KEY}"

# Then run OPD
cd openclaw-opd
python openclaw_opd_rollout.py \
  --student-url http://localhost:30000 \
  --teacher-url http://localhost:30001
```

**Top-K logits distillation** uses the custom loss:

```python
# Uses topk_distillation_loss.py — reverse KL over top-K teacher distribution
# Pass to mlx-tune via:
trainer = GRPOTrainer(..., loss_fn=topk_distillation_loss)
```

### 6.3 Combined Method (Recommended)

```bash
cd openclaw-combine
# Both RL and OPD signals per turn, combined:
# A_combined = w_rl × GRPO_advantage + w_opd × teacher_advantage
python openclaw_combine_rollout.py
```

### 6.4 Connecting to OpenClaw App

Once oMLX is running on port 30000:

1. Open the OpenClaw app
2. Set API base URL to `http://localhost:30000/v1`
3. Set API key to match `--api-key` in oMLX
4. Start chatting — mlx-tune trains in the background
5. Monitor at `http://localhost:8000/admin` (oMLX dashboard)

---

## 7. Track 2 — General Agentic RL

### 7.1 Tool-Call / Math Reasoning (`toolcall-rl`)

```
Math problem → oMLX generates THOUGHT + Python code
             → tool_sandbox.py executes code
             → oMLX processes output → generates answer
             → Reward: answer correctness
             → mlx-tune GRPOTrainer updates policy
```

**Setup:**

```bash
cd toolcall-rl

# Set model and data paths
export HF_CKPT=mlx-community/Qwen3-4B-4bit
export PROMPT_DATA=./data/dapo-math-17k.jsonl
export EVAL_DATA=./data/aime-2024.jsonl
export SAVE_CKPT=./checkpoints/qwen3-4b-retool-rl

# Start oMLX
omlx serve \
  --model-dir ~/.cache/huggingface/hub \
  --max-model-memory 24GB \
  --port 8000

# Run training
python generate_with_retool.py \
  --model-url http://localhost:8000/v1 \
  --trainer mlx-tune \
  --save-dir "${SAVE_CKPT}"
```

**Key arguments (mlx-tune GRPOTrainer):**

```python
GRPOConfig(
    num_generations=8,
    learning_rate=1e-6,
    max_completion_length=8192,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    output_dir=SAVE_CKPT,
    save_steps=40,
)
```

### 7.2 Terminal Agent (`terminal-rl`)

Docker-based terminal environments — the training backend switches to mlx-tune, but the environment infrastructure is unchanged.

```bash
cd terminal-rl

# 1. Download dataset
python data_utils/download.py

# 2. Start Docker pool server (on worker nodes or locally)
python remote/pool_server.py --port 5000

# 3. Start oMLX
omlx serve --model-dir ~/models --port 8000 --max-model-memory 32GB

# 4. Launch training
python generate.py \
  --model-url http://localhost:8000/v1 \
  --env-pool http://localhost:5000 \
  --trainer mlx-tune
```

### 7.3 GUI Agent (`gui-rl`)

Qwen3-VL is a VLM — both oMLX (VLM batching engine) and mlx-tune (VLMSFTTrainer / VLM GRPO) support it.

```bash
cd gui-rl

# oMLX auto-detects VLM from config.json
omlx serve \
  --model-dir ~/.cache/huggingface/hub \
  --max-model-memory 40GB \
  --port 8000

# VM pool manager
python env_pool_server.py --port 5000

# Launch RL
python generate_with_gui.py \
  --model-url http://localhost:8000/v1 \
  --env-pool http://localhost:5000 \
  --trainer mlx-tune \
  --vlm                            # enable vision input handling
```

**mlx-tune VLM training:**

```python
from mlx_tune import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "mlx-community/Qwen3-VL-7B-4bit",
    max_seq_length=8192,
)
model = FastVisionModel.get_peft_model(model, r=16)
```

### 7.4 SWE-Bench Agent (`swe-rl`)

```bash
cd swe-rl

# 1. Pull Docker images
bash data/pull_swe_images.sh

# 2. Preprocess dataset
python data/preprocess_swe_dataset.py

# 3. Start oMLX
omlx serve --model-dir ~/models --port 8000 --max-model-memory 48GB

# 4. Start pool servers
python server/swe_env_pool_server.py    # GPU head node :18090
python server/swe_exec_server.py        # ECS node :5000

# 5. Launch training
python generate_with_swe_remote.py \
  --model-url http://localhost:8000/v1 \
  --trainer mlx-tune
```

---

## 8. Infrastructure Layer (mlx-tune + oMLX)

### oMLX — Inference & Serving

**Start command:**

```bash
omlx serve \
  --model-dir /path/to/models \
  --max-model-memory 32GB \
  --max-process-memory 80% \
  --paged-ssd-cache-dir ~/.omlx/cache \
  --hot-cache-max-size 8GB \
  --prefill-batch-size 8 \
  --completion-batch-size 32 \
  --port 8000 \
  --api-key your-secret-key
```

**Key oMLX features used by OpenClaw-RL:**

| Feature | How it's used |
|---------|--------------|
| OpenAI-compatible API `/v1/chat/completions` | Rollout generation for all methods |
| Multi-model serving | Policy model + PRM in separate slots |
| Continuous batching | High-throughput rollout collection |
| Tiered KV cache (hot GPU + cold SSD) | Keeps large context in play during multi-turn rollouts |
| Hot-reload LoRA weights | Updated adapters loaded without restarting server |
| Admin dashboard `:8000/admin` | Monitor throughput, cache hit rate, memory |

**Admin dashboard access:**

```
http://localhost:8000/admin          ← model management
http://localhost:8000/admin/chat     ← built-in chat UI for testing
```

### mlx-tune — Training

**Available trainers (all Unsloth-compatible API):**

| Trainer | Class | Use in OpenClaw-RL |
|---------|-------|-------------------|
| `GRPOTrainer` | `mlx_tune.GRPOTrainer` | Binary RL (primary) |
| `SFTTrainer` | `mlx_tune.SFTTrainer` | Cold-start SFT before RL |
| `DPOTrainer` | `mlx_tune.DPOTrainer` | Preference pairs from conversations |
| `VLMSFTTrainer` | `mlx_tune.VLMSFTTrainer` | GUI agent (Qwen3-VL) |

**GRPOConfig key parameters:**

```python
GRPOConfig(
    # GRPO-specific
    num_generations=8,              # group size (was --n-samples-per-prompt)
    beta=0.001,                     # KL penalty coefficient
    epsilon=0.2,                    # PPO clip lower bound
    epsilon_high=0.28,              # PPO clip upper bound

    # Training
    learning_rate=1e-5,
    lr_scheduler_type="constant",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_completion_length=8192,
    max_prompt_length=32768,

    # Memory
    load_in_4bit=True,
    grad_checkpoint=True,           # saves ~40% memory

    # Output
    output_dir="./checkpoints",
    save_steps=100,
    logging_steps=10,
)
```

**LoRA configuration:**

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                           # rank (8 for less memory, 32 for more capacity)
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    use_gradient_checkpointing=True,
)
```

**Checkpoint export:**

```python
# Save LoRA adapters only
model.save_pretrained("./lora_adapters")

# Save merged model for deployment
model.save_pretrained_merged("./merged_model", tokenizer)

# Export to GGUF (for Ollama)
model.save_pretrained_gguf("./gguf_model", tokenizer)

# Push to HuggingFace Hub
model.push_to_hub("username/my-openclaw-model", tokenizer)
```

---

## 9. Key Configuration Knobs

### oMLX Server

```bash
# Memory
--max-model-memory 32GB          # cap per-model memory usage
--max-process-memory 80%         # total process memory limit

# KV cache
--paged-ssd-cache-dir ~/.omlx/cache  # enable SSD cold tier
--hot-cache-max-size 8GB              # GPU hot tier size
--no-cache                            # disable for debugging

# Batching (tune for throughput vs latency)
--prefill-batch-size 8           # parallel prefill tokens
--completion-batch-size 32       # parallel generation sequences

# Network
--port 30000
--api-key your-secret            # required for non-localhost
```

### mlx-tune GRPOTrainer

```python
GRPOConfig(
    num_generations=8,           # group size (more = better advantage estimate, more memory)
    beta=0.001,                  # KL penalty (0 = unconstrained)
    epsilon=0.2,                 # clip lower (was --eps-clip)
    epsilon_high=0.28,           # clip upper (was --eps-clip-high)
    learning_rate=1e-5,          # 1e-6 for 32B models
    grad_checkpoint=True,        # saves ~40% memory
    load_in_4bit=True,           # quantized base model
    save_steps=100,
    logging_steps=10,
)
```

### Memory Saving Tips

| Technique | Memory saved | How |
|-----------|-------------|-----|
| 4-bit quantization | ~4× | `load_in_4bit=True` |
| LoRA only | saves full weights | `r=8–16` instead of full FT |
| Gradient checkpointing | ~40% activations | `grad_checkpoint=True` |
| Smaller batch | linear | reduce `per_device_train_batch_size` |
| oMLX SSD tier | spills KV to disk | `--paged-ssd-cache-dir` |
| oMLX LRU eviction | auto | `--max-model-memory` limit |

---

## 10. Deployment Options

### Option A — Single Mac (LoRA, recommended)

Works on any Mac with 16GB+ unified memory.

```bash
# Start oMLX serving
omlx serve --model-dir ~/models --max-model-memory 16GB --port 30000

# Train with mlx-tune LoRA
python train.py --model mlx-community/Qwen3-4B-4bit --method rl --lora-r 16
```

### Option B — Single Mac (Full Fine-Tune)

Requires 64GB+ unified memory (Mac Studio/Pro Ultra).

```python
# No LoRA — train all weights
model, tokenizer = FastLanguageModel.from_pretrained(
    "mlx-community/Qwen3-8B-bf16",   # full precision
    load_in_4bit=False,
)
# Skip get_peft_model() — train everything
```

### Option C — Tinker Cloud (No Local GPU)

```bash
cd openclaw-tinker
python run.py --method combine   # Binary RL + OPD
```

Tinker deployment is LoRA-only. mlx-tune produces a LoRA adapter that gets uploaded. The cloud worker handles the actual gradient computation.

### Option D — Mac for Rollout, Cloud for Training

Use oMLX locally for fast rollout data collection, then train on a cloud GPU with Unsloth (mlx-tune is Unsloth-compatible):

```bash
# 1. Collect rollouts locally with oMLX
python collect_rollouts.py --save-path ./rollouts.jsonl

# 2. Upload to cloud and train with Unsloth (same API)
# cloud: pip install unsloth
# cloud: from unsloth import FastLanguageModel, GRPOTrainer  ← identical API
```

---

## 11. Evaluation

### Built-in Eval (during training)

```python
GRPOConfig(
    eval_strategy="steps",
    eval_steps=20,
    eval_dataset=aime_dataset,     # AIME 2024
    metric_for_best_model="reward",
)
```

### OpenClaw-Test

```bash
cd openclaw-test

# Phase 1: Student solves GSM8K via your trained agent
python student_chat.py \
  --model-url http://localhost:30000/v1 \
  --api-key your-key

# Phase 2: Teacher grades solutions
python teacher_chat.py
```

### oMLX Built-in Benchmark

```
http://localhost:8000/admin → Benchmark tab
```

One-click prefill/generation token/sec benchmarks directly from the dashboard.

---

## 12. Extending the Framework

### Custom Reward Function

```python
def openclaw_reward_func(prompts, completions, **kwargs):
    """Return list of scalar rewards, one per completion."""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # your logic here — query next state, evaluate correctness, etc.
        reward = evaluate(prompt, completion)
        rewards.append(float(reward))
    return rewards

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[openclaw_reward_func],   # can pass multiple
    ...
)
```

### Custom Loss Function

```python
import mlx.core as mx

def custom_loss(logprobs, advantages, masks, **kwargs):
    """Custom advantage-weighted loss."""
    clipped = mx.clip(advantages, -2.0, 2.0)
    loss = -(logprobs * clipped * masks).sum() / masks.sum()
    return loss

trainer = GRPOTrainer(model=model, loss_fn=custom_loss, ...)
```

### Custom Rollout (oMLX client)

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:30000/v1",
    api_key=os.environ["SGLANG_API_KEY"],
)

async def generate_rollout(prompt):
    response = await client.chat.completions.create(
        model="Qwen3-4B-4bit",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8192,
        temperature=0.6,
    )
    return response.choices[0].message.content
```

### Adding a New Model

1. Download the MLX-format model:
   ```bash
   huggingface-cli download mlx-community/MyModel-4bit
   ```
2. oMLX auto-discovers it from `--model-dir` — no config needed
3. In mlx-tune, point `from_pretrained()` at the model ID or local path
4. For VLMs, use `FastVisionModel` instead of `FastLanguageModel`

---

## 13. Troubleshooting

### oMLX not starting

```bash
# Check port conflict
lsof -i :8000

# Verify MLX is installed
python -c "import mlx; print(mlx.__version__)"

# Start with verbose logging
omlx serve --model-dir ~/models --log-level debug
```

### Out of memory (Mac)

1. Reduce `--max-model-memory` to force LRU eviction
2. Enable SSD KV cache: `--paged-ssd-cache-dir ~/.omlx/cache`
3. Use 4-bit quantized models (`mlx-community/*-4bit`)
4. Reduce LoRA rank: `r=8` instead of `r=16`
5. Enable gradient checkpointing: `grad_checkpoint=True`

### Model not found in oMLX

```bash
# Check if model exists in your model dir
ls ~/models/

# Reload model list without restart
curl http://localhost:8000/v1/models

# Or trigger via dashboard
http://localhost:8000/admin → Models tab → Reload
```

### Slow training throughput

- Increase `gradient_accumulation_steps` to decouple batch size from memory
- Use `--completion-batch-size 32` in oMLX for higher rollout parallelism
- Enable `--hot-cache-max-size 8GB` to keep KV blocks in GPU memory
- Use smaller context during rollout: `max_completion_length=4096` for warm-up

### LoRA adapter not reflected in oMLX

After mlx-tune saves an adapter, trigger hot-reload:

```bash
# Via API
curl -X POST http://localhost:8000/v1/models/reload \
  -d '{"model": "Qwen3-4B-4bit", "adapter_path": "./checkpoints/lora_adapters"}'

# Or via dashboard
http://localhost:8000/admin → Models → Reload Adapter
```

### mlx-tune GRPO diverging

- Lower learning rate: `1e-6` for 8B+, `1e-5` for 4B
- Increase KL penalty: `beta=0.01`
- Increase group size: `num_generations=16` for better advantage estimation
- Check reward scale — rewards should be in roughly [−1, +1]

---

## Quick Reference — Which Script to Run

| Goal | Command |
|------|---------|
| Start oMLX server | `omlx serve --model-dir ~/models --max-model-memory 32GB` |
| Personal agent, binary RL, 4B | `cd openclaw-rl && python openclaw_rollout.py` |
| Personal agent, OPD token-level | `cd openclaw-opd && python openclaw_opd_rollout.py` |
| Personal agent, combined (recommended) | `cd openclaw-combine && python openclaw_combine_rollout.py` |
| Cloud deployment (no local GPU) | `cd openclaw-tinker && python run.py --method combine` |
| Math / tool-call, 4B | `cd toolcall-rl && python generate_with_retool.py` |
| Terminal agent, 8B | `cd terminal-rl && python generate.py` |
| GUI agent, 8B VLM | `cd gui-rl && python generate_with_gui.py --vlm` |
| SWE agent | `cd swe-rl && python generate_with_swe_remote.py` |
| End-to-end eval | `cd openclaw-test && python student_chat.py` |
| Monitor training | `open http://localhost:8000/admin` |
| Export trained model | `model.save_pretrained_merged("./merged")` |

---

## Platform Comparison

| Aspect | CUDA Stack (original) | MLX Stack (this guide) |
|--------|----------------------|----------------------|
| **Training** | Megatron-LM (TP/PP/EP) | mlx-tune (unified memory) |
| **Inference** | SGLang | oMLX |
| **Hardware** | Linux + NVIDIA GPU clusters | macOS + Apple Silicon |
| **Memory** | VRAM (limited, expensive) | Unified RAM up to 512GB |
| **Parallelism** | TP × PP × EP across nodes | Single Mac, no distribution |
| **Model scale** | Up to 405B+ (multi-node) | Up to ~70B (single Mac Ultra) |
| **Setup complexity** | High (cluster, NCCL, Slurm) | Low (brew install omlx) |
| **API compatibility** | OpenAI-compatible | OpenAI + Anthropic compatible |
| **Portability** | Cloud-scale to Unsloth | Mac-local, cloud-portable |
