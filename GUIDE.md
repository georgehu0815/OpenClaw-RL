# OpenClaw-RL — System Guide

> **Train a personalized AI agent simply by talking to it. Scale RL to real-world agentic settings.**

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Repository Layout](#3-repository-layout)
4. [Dependencies & Setup](#4-dependencies--setup)
5. [Training Methods](#5-training-methods)
6. [Track 1 — Personal Agent (OpenClaw)](#6-track-1--personal-agent-openclaw)
7. [Track 2 — General Agentic RL](#7-track-2--general-agentic-rl)
8. [Infrastructure Layer (SLIME + Megatron)](#8-infrastructure-layer-slime--megatron)
9. [Key Configuration Knobs](#9-key-configuration-knobs)
10. [Deployment Options](#10-deployment-options)
11. [Evaluation](#11-evaluation)
12. [Extending the Framework](#12-extending-the-framework)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. System Overview

OpenClaw-RL is a **fully asynchronous reinforcement learning framework** with two tracks:

| Track | Goal | Methods |
|-------|------|---------|
| **Track 1** — Personal Agent | Train from live OpenClaw conversations | Binary RL, OPD, Combined |
| **Track 2** — General Agentic RL | Train agents for real-world tasks | GRPO + PRM for Terminal/GUI/SWE/Tool-call |

**Key design principles:**

- **Fully async** — 4 components run independently, never blocking each other
- **Self-hosted & private** — everything runs on your own infrastructure
- **Zero manual labeling** — reward signals come from environment feedback or natural conversation
- **LoRA or full fine-tuning** — works with 1 GPU (LoRA) or large multi-node clusters

---

## 2. Architecture

### The 4-Component Async Loop

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   ┌──────────────┐    multi-turn     ┌──────────────────┐  │
│   │   User /     │ ──conversations─▶ │  API Server      │  │
│   │   Environment│                   │  (FastAPI proxy) │  │
│   └──────────────┘                   └────────┬─────────┘  │
│                                               │             │
│                                        trajectories         │
│                                               │             │
│   ┌──────────────┐                   ┌────────▼─────────┐  │
│   │  Policy      │ ◀─── gradient ─── │  Rollout Buffer  │  │
│   │  Trainer     │      updates       │  (SLIME)         │  │
│   │  (Megatron)  │                   └────────┬─────────┘  │
│   └──────────────┘                            │             │
│                                          reward signals      │
│                                               │             │
│                                   ┌───────────▼──────────┐ │
│                                   │  PRM / Judge Model   │ │
│                                   │  (SGLang engine)     │ │
│                                   └──────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Component responsibilities:**

| Component | Role | Technology |
|-----------|------|------------|
| **API Server** | OpenAI-compatible endpoint; intercepts user ↔ agent conversations | FastAPI + SGLang |
| **Rollout Buffer** | Collects trajectories, waits for rewards, formats training data | SLIME (`ray`) |
| **PRM / Judge** | Evaluates each turn or full trajectory; produces reward signal | SGLang + custom model |
| **Policy Trainer** | Runs GRPO/OPD gradient updates on the policy | Megatron-LM + Ray |

All four components communicate via **async queues** — the model keeps serving requests while training runs in the background.

---

## 3. Repository Layout

```
OpenClaw-RL/
│
├── slime/                      # Core RL training framework (THUDM)
│   ├── slime/                  # Library: Ray actors, rollout, backends
│   ├── slime_plugins/          # Megatron bridge, SGLang models, buffers
│   ├── scripts/models/         # Model configs (Qwen3, Qwen2.5, GLM4, ...)
│   └── examples/               # On-policy distillation, VLM, etc.
│
├── Megatron-LM/                # NVIDIA distributed training framework
│
├── openclaw-rl/                # Track 1 — Binary RL (GRPO)
├── openclaw-opd/               # Track 1 — On-Policy Distillation
├── openclaw-combine/           # Track 1 — Combined RL + OPD
├── openclaw-tinker/            # Track 1 — Cloud (Tinker) deployment
├── openclaw-test/              # End-to-end evaluation suite
│
├── toolcall-rl/                # Track 2 — Math/tool-call (ReTool)
├── terminal-rl/                # Track 2 — Terminal/shell agent
├── gui-rl/                     # Track 2 — Desktop GUI agent (Qwen3-VL)
├── swe-rl/                     # Track 2 — SWE-Bench issue fixing
│
├── extensions/                 # OpenClaw plugin for self-hosted deployment
├── instructions/               # Setup instructions
├── requirements.txt            # Python dependencies
└── GUIDE.md                    # This file
```

---

## 4. Dependencies & Setup

### Hardware Requirements

| Mode | Minimum | Recommended |
|------|---------|-------------|
| LoRA (4B model) | 1× A100 80GB | 2× A100 |
| Full fine-tune (4B) | 4× A100 80GB | 8× A100 |
| Full fine-tune (32B) | 4 nodes × 8× A100 | 8 nodes × 8× H100 |

### Software Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Runtime |
| PyTorch | 2.9.1 | Deep learning |
| Transformers | 4.57.1 | HuggingFace models |
| Ray | 2.54.0 | Distributed actor system |
| SGLang | custom | Fast LLM inference + rollout |
| Megatron-LM | dev_rl branch | Distributed training |
| PEFT | ≥0.12.0 | LoRA support |
| FastAPI | 0.131.0 | API server |
| wandb | 0.25.0 | Experiment tracking |


### New Software Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Runtime |
| PyTorch | 2.11.0 | Deep learning |
| Transformers | 4.57.1 | HuggingFace models |

| oMLX | custom | Fast LLM inference + rollout |
| mlx-tune | dev_rl branch | LLM training |
| mlx-tune | "0.4.7" | LoRA support |
| FastAPI | 0.131.0 | API server |
| wandb | 0.25.0 | Experiment tracking |

mlx-tune /Volumes/ExternalSSD/train/mlx-tune
oMLX /Volumes/ExternalSSD/serve/omlx
    "torch==2.11.0",
    "torchvision==0.26.0"
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

**Step 3 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4 — Install Megatron-LM**

```bash
cd Megatron-LM
pip install -e .
cd ..
```

**Step 5 — Install SLIME**

```bash
cd slime
pip install -e .
cd ..
```

**Step 6 — Set PYTHONPATH** (or let the launch scripts handle it)

```bash
export PYTHONPATH="/path/to/OpenClaw-RL/Megatron-LM:$PYTHONPATH"
```

---

## 5. Training Methods

### Method Comparison

| Method | Reward Signal | Best For | Loss |
|--------|--------------|----------|------|
| **Binary RL** | Scalar +1/-1/0 from evaluator | Implicit feedback (thumbs up/down) | GRPO clipped surrogate |
| **OPD** (Token-level) | Token-level directional: log_teacher - log_student | Explicit feedback with hints | Directional advantage |
| **OPD** (Top-K) | Reverse KL over top-K teacher distribution | Strong teacher signal (SDFT/SDPO style) | KL divergence |
| **Combined** | Both RL and OPD signals simultaneously | Maximum training signal density | w_rl × GRPO + w_opd × teacher |

### GRPO Advantage Estimation

For Binary RL and Combined methods, advantage is computed per group:

```
A_i = (r_i - mean(r)) / (std(r) + ε)
```

where `r` is the reward across `n-samples-per-prompt` rollouts for the same prompt.

### Process Reward Model (PRM)

Optional per-step reward instead of terminal reward:

- `m` independent evaluations per step → majority vote
- Produces intermediate reward at each agent action
- Controlled via `--prm-enable --prm-m <m> --prm-num-gpus <n>`

---

## 6. Track 1 — Personal Agent (OpenClaw)

This track trains your personal OpenClaw agent from real conversations.

### How It Works

1. User sends a message to the OpenClaw app
2. The API server (FastAPI proxy) intercepts the conversation
3. The rollout worker collects multi-turn trajectories
4. PRM/judge evaluates each turn based on next-state feedback
5. SLIME queues the scored trajectory for GRPO training
6. Policy updates happen asynchronously — the model keeps serving requests

### 6.1 Binary RL

**Entry point:** [openclaw-rl/openclaw_api_server.py](openclaw-rl/openclaw_api_server.py)

**Architecture:**
```
User → OpenClaw app → API proxy (openclaw_api_server.py)
                          ↓
                    Rollout worker (openclaw_rollout.py)
                          ↓
                    SLIME buffer → Megatron training
```

**Launch (single node, 8 GPUs):**

```bash
cd openclaw-rl

# Set your paths
export HF_CKPT=/path/to/Qwen3-4B
export SAVE_CKPT=/path/to/save/checkpoints
export WANDB_KEY=your_wandb_key          # optional

bash run_qwen3_4b_openclaw_rl.sh
```

**GPU split (default for 8 GPUs):**

| Component | GPUs |
|-----------|------|
| Actor (Megatron trainer) | 4 |
| Rollout (SGLang engine) | 2 |
| PRM (judge model) | 2 |

**LoRA mode (fewer GPUs):**

```bash
bash run_qwen3_4b_openclaw_rl_lora.sh
```

**Key arguments:**

```bash
--rollout-function-path openclaw_rollout.generate_rollout_openclaw
--custom-generate-function-path openclaw_api_server.generate
--custom-rm-path openclaw_api_server.reward_func
--prm-enable                          # enable process reward model
--prm-m 3                             # majority vote across 3 evaluations
--rollout-temperature 0.6
--rollout-max-response-len 8192
```

### 6.2 On-Policy Distillation (OPD)

**Entry point:** [openclaw-opd/openclaw_opd_api_server.py](openclaw-opd/openclaw_opd_api_server.py)

The key difference: a **hint-judge** extracts hindsight hints from the next turn's feedback, then a **teacher model** is queried with those hints to get directional log-probabilities.

**Token-level OPD:**

```bash
cd openclaw-opd
bash run_qwen3_4b_openclaw_opd.sh
```

**Top-K logits distillation (SDFT/SDPO style):**

```bash
bash run_qwen3_4b_openclaw_opd_topk.sh
```

The top-K variant uses a custom loss from [openclaw-opd/topk_distillation_loss.py](openclaw-opd/topk_distillation_loss.py), injected via:

```bash
--custom-loss-function-path topk_distillation_loss.loss_func
```

### 6.3 Combined Method (Recommended)

**Entry point:** [openclaw-combine/openclaw_combine_api_server.py](openclaw-combine/openclaw_combine_api_server.py)

Combines both RL and OPD signals per turn. The combined advantage:

```
A_combined = w_rl × A_grpo + w_opd × A_teacher
```

```bash
cd openclaw-combine
bash run_qwen3_4b_openclaw_combine.sh
```

The combined loss is in [openclaw-combine/combine_loss.py](openclaw-combine/combine_loss.py), controlled by `w_rl` and `w_opd` weights.

### 6.4 Connecting to OpenClaw App

Once your API server is running (default `0.0.0.0:30000`):

1. Open the OpenClaw app
2. Point it to `http://your-server:30000/v1`
3. Set the API key matching `SGLANG_API_KEY`
4. Start chatting — training runs automatically in the background

---

## 7. Track 2 — General Agentic RL

### 7.1 Tool-Call / Math Reasoning (`toolcall-rl`)

Trains models to solve math problems by generating and executing Python code.

**Dataset:** ReTool (DeepSeek-R1-Distill-Qwen-32B-SFT)
**Eval:** AIME 2024

**Rollout loop:**

```
Prompt → Model generates THOUGHT + Python code
       → Sandbox executes code (tool_sandbox.py)
       → Model sees output → generates answer
       → Reward: answer correctness
```

**Single-node (4B):**

```bash
cd toolcall-rl
export HF_CKPT=/path/to/Qwen3-4B
bash retool_qwen3_4b_rl.sh
```

**Multi-node (32B, 4 nodes × 8 GPUs):**

```bash
# On head node (MLP_ROLE_INDEX=0):
export HF_CKPT=/path/to/ReTool-Qwen-32B-SFT
export MLP_ROLE_INDEX=0
export MASTER_ADDR=<head-node-ip>
bash retool_qwen25_32b_4nodes_rl.sh

# On each worker node (MLP_ROLE_INDEX=1,2,3):
export MLP_ROLE_INDEX=1   # 2, 3 on subsequent nodes
export MASTER_ADDR=<head-node-ip>
bash retool_qwen25_32b_4nodes_rl.sh
```

**Key arguments:**

```bash
--custom-generate-function-path generate_with_retool.generate
--custom-rm-path generate_with_retool.reward_func
--n-samples-per-prompt 8              # group size for GRPO
--rollout-batch-size 32
--rollout-max-response-len 8192
--eval-interval 20                    # evaluate on AIME every 20 steps
```

### 7.2 Terminal Agent (`terminal-rl`)

Trains agents to execute shell commands inside Docker environments.

**Infrastructure:**

```
Training node (Ray + SLIME)
       ↓  HTTP
Remote Docker workers (pool_server.py)
       ↓
Docker containers (bash tasks from SETA dataset)
```

**Setup:**

```bash
# 1. Download SETA dataset
cd terminal-rl
python data_utils/download.py
python data_utils/load_tasks.py
python data_utils/convert_task_to_dataset.py

# 2. Start pool server on remote worker nodes
python remote/pool_server.py --port 5000

# 3. Launch training
bash terminal_qwen3_8b_rl.sh
```

**With PRM (2 nodes):**

```bash
bash terminal_qwen3_8b_prm_rl_2nodes.sh
```

### 7.3 GUI Agent (`gui-rl`)

Trains vision-language models (Qwen3-VL) to control desktop GUIs.

**Infrastructure:**

```
Training node
       ↓  HTTP
VM pool server (env_pool_server.py)
       ↓
Cloud VMs running OSWorld environments
```

**Agent loop:**

```
Screenshot → Qwen3-VL generates pyautogui action
           → Action executes on VM
           → New screenshot → repeat
           → OSWorld evaluator scores success/failure
```

**Setup:**

```bash
cd gui-rl

# Start VM pool manager (on VM management node)
python env_pool_server.py --port 5000

# Launch RL training
bash gui_qwen3vl_8b_rl.sh

# Or with PRM
bash gui_qwen3vl_8b_prm_rl.sh

# Evaluation only
bash gui_qwen3vl_4b_eval.sh
```

### 7.4 SWE-Bench Agent (`swe-rl`)

Trains agents to fix GitHub issues by interacting with Docker-based repo environments.

**Infrastructure:**

```
GPU head node (training + env pool server :18090)
       ↓
ECS Docker nodes (swe_exec_server.py :5000)
       ↓
Docker containers (one per GitHub repo + issue)
```

**Setup:**

```bash
cd swe-rl

# 1. Pull Docker images for SWE-Bench
bash data/pull_swe_images.sh

# 2. Preprocess dataset
python data/preprocess_swe_dataset.py

# 3. Start ECS pool server on GPU head node
python server/swe_env_pool_server.py

# 4. Start execution server on each ECS node
python server/swe_exec_server.py

# 5. Launch training
bash run_swe_rl_8b_remote_2nodes.sh    # 8B model, 2 nodes
bash run_swe_rl_32b_remote_4nodes.sh   # 32B model, 4 nodes
bash run_swe_rl_32b_remote_8nodes.sh   # 32B model, 8 nodes
```

**Agent action format:**

```
THOUGHT: <reasoning>
<bash>
command here
</bash>
```

The agent iterates: think → execute → observe → repeat → submit patch → test suite evaluates.

---

## 8. Infrastructure Layer (SLIME + Megatron)

### SLIME Training Entry Point

All methods ultimately call `train_async.py` from SLIME:

```bash
python3 train_async.py \
  --actor-num-nodes N \
  --actor-num-gpus-per-node M \
  --rollout-num-gpus K \
  [model args] [ckpt args] [rollout args] [grpo args] [optimizer args]
```

### Model Configs

Pre-built model configs live in [slime/scripts/models/](slime/scripts/models/). Each script exports `MODEL_ARGS`:

```bash
source slime/scripts/models/qwen3-4B.sh     # Qwen3 4B
source slime/scripts/models/qwen3-8B.sh     # Qwen3 8B
source slime/scripts/models/qwen3-32B.sh    # Qwen3 32B
source slime/scripts/models/qwen2.5-32B.sh  # Qwen2.5 32B
source slime/scripts/models/deepseek-r1.sh  # DeepSeek R1
```

### Checkpoint Management

| Argument | Description |
|----------|-------------|
| `--hf-checkpoint` | HuggingFace checkpoint path (initial weights) |
| `--ref-load` | Reference model path (for KL regularization) |
| `--save` | Output checkpoint directory |
| `--save-interval` | Save every N training steps |
| `--megatron-to-hf-mode bridge` | Convert Megatron ↔ HF format |

Checkpoints are saved in **distributed format** (sharded by TP/PP ranks). Use the conversion tools in `slime/tools/` to get a standard HuggingFace checkpoint.

### Parallelism Strategy

| Argument | Purpose |
|----------|---------|
| `--tensor-model-parallel-size` | TP: split weight matrices across GPUs |
| `--pipeline-model-parallel-size` | PP: split layers across GPU groups |
| `--expert-model-parallel-size` | EP: for MoE models (e.g. DeepSeek) |
| `--sequence-parallel` | Split sequence dimension (saves activation memory) |
| `--recompute-granularity full` | Recompute all activations during backward (saves memory) |

**Typical setups:**

| Model | Nodes | GPUs/node | TP | PP |
|-------|-------|-----------|----|----|
| 4B | 1 | 8 | 4 | 1 |
| 8B | 1 | 8 | 4 | 1 |
| 32B | 4 | 8 | 8 | 1 |
| 70B | 8 | 8 | 8 | 2 |

---

## 9. Key Configuration Knobs

### Rollout Arguments

```bash
--rollout-batch-size 16          # prompts processed per rollout step
--n-samples-per-prompt 8         # rollouts per prompt (GRPO group size)
--rollout-temperature 0.6        # sampling temperature
--rollout-max-response-len 8192  # max tokens generated per turn
--rollout-max-context-len 32768  # max context window
--num-rollout 100000000          # total rollout steps (effectively infinite)
--num-steps-per-rollout 1        # training steps per rollout batch
```

### GRPO Arguments

```bash
--advantage-estimator grpo
--eps-clip 0.2                   # PPO clip lower bound
--eps-clip-high 0.28             # PPO clip upper bound
--use-kl-loss
--kl-loss-coef 0.001             # KL penalty strength (0 = disabled)
--kl-loss-type low_var_kl
--entropy-coef 0.0               # entropy bonus
--disable-rewards-normalization  # skip reward whitening (for sparse rewards)
```

### Optimizer Arguments

```bash
--optimizer adam
--lr 1e-5                        # learning rate (1e-6 for large models)
--lr-decay-style constant        # keep LR constant throughout
--weight-decay 0.1
--adam-beta1 0.9
--adam-beta2 0.98
--optimizer-cpu-offload          # offload optimizer states to CPU (saves GPU memory)
--overlap-cpu-optimizer-d2h-h2d  # overlap CPU↔GPU transfers
--use-precision-aware-optimizer  # mixed precision optimizer states
```

### Memory Saving

```bash
--recompute-granularity full    # recompute activations (saves ~40% memory)
--optimizer-cpu-offload         # CPU optimizer states
--use-dynamic-batch-size        # pack sequences by token count
--max-tokens-per-gpu 32768      # token budget per GPU per step
```

---

## 10. Deployment Options

### Option A — Local GPUs (Full Fine-Tune)

Standard path for 8+ GPU nodes. All launch scripts in each method folder handle this.

```bash
# Example: Binary RL, single node
cd openclaw-rl
bash run_qwen3_4b_openclaw_rl.sh
```

### Option B — Local GPUs (LoRA)

Reduced memory with LoRA adapters. Works on as few as 1-2 GPUs.

```bash
cd openclaw-rl
bash run_qwen3_4b_openclaw_rl_lora.sh
```

### Option C — Tinker Cloud (No Local GPUs)

For users without local GPU infrastructure. Uses Thinking Machines AI's Tinker platform.

```bash
cd openclaw-tinker
python run.py --method rl      # Binary RL
python run.py --method opd     # OPD
python run.py --method combine # Combined (recommended)
```

Tinker deployment is LoRA-only. See [openclaw-tinker/README.md](openclaw-tinker/README.md) for cloud credentials setup.

### Option D — Multi-Node Cluster

For large models (32B+). Uses MLP environment variables for node discovery:

```bash
# Head node
export MLP_ROLE_INDEX=0
export MASTER_ADDR=<head-ip>
bash retool_qwen25_32b_4nodes_rl.sh

# Worker nodes
export MLP_ROLE_INDEX=1   # increment per node
export MASTER_ADDR=<head-ip>
bash retool_qwen25_32b_4nodes_rl.sh
```

---

## 11. Evaluation

### Built-in Eval (during training)

Configure inline eval for math benchmarks:

```bash
--eval-interval 20                     # evaluate every 20 steps
--eval-prompt-data aime /path/to/aime.jsonl
--n-samples-per-eval-prompt 16         # maj@16
--eval-max-response-len 16384
--eval-reward-key acc
```

### OpenClaw-Test (end-to-end)

Tests trained models on GSM8K math problems via a 2-phase simulation:

**Phase 1 — Student interaction:**

```bash
cd openclaw-test
python student_chat.py --model-path /path/to/your-model
```

The student LLM interacts with the trained agent across 36 GSM8K problems.

**Phase 2 — Teacher grading:**

```bash
python teacher_chat.py
```

The teacher LLM grades student submissions and provides feedback scores.

This tests: instruction following, math reasoning, file I/O, and style adaptation.

---

## 12. Extending the Framework

### Custom Rollout Function

Create a Python file with a `generate_rollout_*` function:

```python
# my_rollout.py
async def generate_rollout_custom(prompts, model, tokenizer, **kwargs):
    # collect trajectories from your environment
    # return list of (prompt, response, reward) tuples
    ...
```

Then pass:

```bash
--rollout-function-path my_rollout.generate_rollout_custom
```

### Custom Generate Function

Override how the model generates responses:

```python
# my_generate.py
async def generate(prompts, sampling_params, **kwargs):
    # call your API server, environment, etc.
    # return completions
    ...
```

```bash
--custom-generate-function-path my_generate.generate
```

### Custom Reward Model

Plug in your own reward function:

```python
# my_rm.py
def reward_func(prompts, completions, **kwargs):
    # return list of scalar rewards
    ...
```

```bash
--custom-rm-path my_rm.reward_func
```

### Custom Loss Function

Replace the GRPO loss entirely:

```python
# my_loss.py
def loss_func(logprobs, advantages, masks, **kwargs):
    # return scalar loss
    ...
```

```bash
--custom-loss-function-path my_loss.loss_func
```

### Adding a New Model

1. Create a model config in `slime/scripts/models/my-model.sh` exporting `MODEL_ARGS`
2. Copy a launch script from an existing method folder
3. Change `source slime/scripts/models/...sh` to point to your new config
4. Set `HF_CKPT` to your model checkpoint path

---

## 13. Troubleshooting

### Ray cluster issues

```bash
# Kill all stale processes before restarting
pkill -9 sglang && ray stop --force && pkill -9 ray && pkill -9 python
sleep 5
# Then re-run your launch script
```

### Out of memory (OOM)

1. Enable activation recompute: `--recompute-granularity full`
2. CPU offload optimizer: `--optimizer-cpu-offload`
3. Reduce batch: `--rollout-batch-size 8` and `--max-tokens-per-gpu 16384`
4. Use LoRA instead of full fine-tune

### SGLang engine not starting

- Check `--sglang-mem-fraction-static` — default 0.85, lower if OOM
- Check `--sglang-context-length` matches your model's max context
- Ensure `SGLANG_API_KEY` env var is set and matches what your OpenClaw app uses

### Workers not joining Ray cluster

```bash
# Increase health check timeouts
export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60
```

Also ensure `no_proxy` includes the head node IP:

```bash
export no_proxy="127.0.0.1,${MASTER_ADDR}"
```

### Slow training throughput

- Enable `--use-dynamic-batch-size` to pack sequences
- Enable overlapped communication: `--overlap-cpu-optimizer-d2h-h2d`
- For NVLink clusters: set `NCCL_NVLS_ENABLE=1`
- Increase `--max-tokens-per-gpu` to better utilize GPU memory

### Checkpoint format mismatch

Use `--megatron-to-hf-mode bridge` for Megatron ↔ HuggingFace conversion.
The `--auto-detect-ckpt-format` flag helps with mixed checkpoint formats.

---

## Quick Reference — Which Script to Run

| Goal | Script |
|------|--------|
| Personal agent, binary RL, 4B, 1 node | `openclaw-rl/run_qwen3_4b_openclaw_rl.sh` |
| Personal agent, binary RL, 4B, LoRA | `openclaw-rl/run_qwen3_4b_openclaw_rl_lora.sh` |
| Personal agent, OPD token-level, 4B | `openclaw-opd/run_qwen3_4b_openclaw_opd.sh` |
| Personal agent, OPD top-K, 4B | `openclaw-opd/run_qwen3_4b_openclaw_opd_topk.sh` |
| Personal agent, combined (recommended) | `openclaw-combine/run_qwen3_4b_openclaw_combine.sh` |
| Cloud deployment (no local GPU) | `openclaw-tinker/run.py --method combine` |
| Math / tool-call, 4B | `toolcall-rl/retool_qwen3_4b_rl.sh` |
| Math / tool-call, 32B, 4 nodes | `toolcall-rl/retool_qwen25_32b_4nodes_rl.sh` |
| Terminal agent, 8B | `terminal-rl/terminal_qwen3_8b_rl.sh` |
| GUI agent, 8B | `gui-rl/gui_qwen3vl_8b_rl.sh` |
| SWE agent, 8B, 2 nodes | `swe-rl/run_swe_rl_8b_remote_2nodes.sh` |
| SWE agent, 32B, 4 nodes | `swe-rl/run_swe_rl_32b_remote_4nodes.sh` |
| End-to-end eval | `openclaw-test/student_chat.py` |
