# OpenClaw-RL — System Architecture

![OpenClaw-RL Architecture](openclaw-rl-architecture.png)

---

## Architecture Overview

OpenClaw-RL is a **fully asynchronous reinforcement learning framework** organized around two tracks and a shared infrastructure core.

---

## System Diagram (Mermaid Source)

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#e7f5ff',
    'primaryTextColor': '#1a1a2e',
    'primaryBorderColor': '#1971c2',
    'lineColor': '#495057',
    'secondaryColor': '#d3f9d8',
    'tertiaryColor': '#f8f9fa',
    'background': '#ffffff',
    'fontFamily': 'Inter, Segoe UI, sans-serif',
    'fontSize': '15px'
  }
}}%%
graph TB

  subgraph USERS["⬛ Entry Points"]
    direction LR
    U1["👤 User (OpenClaw App)"]
    U2["🔧 Terminal Environment"]
    U3["🖥️ GUI / Desktop Environment"]
    U4["💻 SWE-Bench GitHub Repo"]
    U5["🧮 Math / Tool-Call Python Sandbox"]
  end

  subgraph T1["🟦 TRACK 1 — Personal Agent (OpenClaw)"]
    direction TB
    subgraph T1_API["API Layer"]
      API["🌐 API Server (FastAPI + SGLang)"]
      SGLANG_SERVE["⚡ SGLang Engine (Policy Serving)"]
    end
    subgraph T1_METHODS["Training Methods"]
      M1["📊 Binary RL (GRPO)"]
      M2["📚 On-Policy Distillation (OPD)"]
      M3["🔀 Combined (w_rl × GRPO + w_opd × teacher)"]
    end
    subgraph T1_DEPLOY["Deployment"]
      D1["🖥️ Local GPU Full Fine-Tune"]
      D2["🔬 Local GPU LoRA"]
      D3["☁️ Tinker Cloud (LoRA only)"]
    end
  end

  subgraph T2["🟩 TRACK 2 — General Agentic RL"]
    direction TB
    subgraph T2_TERMINAL["Terminal Agent"]
      TERM_POOL["🐳 Docker Pool Server"]
      TERM_ENV["Container SETA Tasks"]
    end
    subgraph T2_GUI["GUI Agent"]
      GUI_POOL["☁️ VM Pool Server"]
      GUI_ENV["Cloud VM OSWorld"]
      GUI_VLM["🖼️ Qwen3-VL"]
    end
    subgraph T2_SWE["SWE-Bench Agent"]
      SWE_POOL["🏗️ Env Pool Server"]
      SWE_ENV["Docker GitHub Repo"]
    end
    subgraph T2_TOOL["Tool-Call / Math"]
      TOOL_SBX["🐍 Python Sandbox"]
      TOOL_DATA["📐 ReTool / DAPO-Math"]
    end
  end

  subgraph CORE["⬛ Shared Infrastructure"]
    direction TB
    subgraph SLIME_LAYER["🟣 SLIME Framework"]
      ROLLOUT_BUF["📦 Rollout Buffer (Ray)"]
      REWARD_GATE["🎯 Reward Aggregator"]
    end
    subgraph PRM_LAYER["🟠 PRM / Judge Layer"]
      PRM["🧠 Process Reward Model (SGLang)"]
      JUDGE["⚖️ Outcome Judge"]
    end
    subgraph TRAINER["🔴 Policy Trainer — Megatron-LM"]
      MEGA["⚙️ Megatron-LM (TP/PP/EP)"]
      GRPO_LOSS["📉 GRPO Loss"]
      OPT["🔧 Adam Optimizer (CPU offload)"]
    end
    subgraph CKPT["💾 Checkpoint Management"]
      HF_CKPT["🤗 HuggingFace checkpoint"]
      MEGA_CKPT["🗄️ Megatron distributed ckpt"]
      BRIDGE["🔄 HF ↔ Megatron bridge"]
    end
  end

  subgraph MODELS["🟡 Supported Models"]
    direction LR
    MOD1["Qwen3 4B/8B/32B"]
    MOD2["Qwen3-VL 4B/8B"]
    MOD3["Qwen2.5 32B"]
    MOD4["DeepSeek-R1"]
    MOD5["GLM4 9B/30B/355B"]
  end

  subgraph OBS["📡 Observability"]
    WB["📊 Weights & Biases"]
    RAY_DASH["🖥️ Ray Dashboard :8265"]
  end

  U1 -->|"conversation"| API
  API --> SGLANG_SERVE
  API --> ROLLOUT_BUF
  SGLANG_SERVE -.-> MEGA
  M1 --> ROLLOUT_BUF
  M2 --> ROLLOUT_BUF
  M3 --> ROLLOUT_BUF

  U2 --> TERM_POOL --> TERM_ENV --> ROLLOUT_BUF
  U3 --> GUI_POOL --> GUI_ENV --> GUI_VLM --> ROLLOUT_BUF
  U4 --> SWE_POOL --> SWE_ENV --> ROLLOUT_BUF
  U5 --> TOOL_SBX --> ROLLOUT_BUF
  TOOL_DATA --> REWARD_GATE

  ROLLOUT_BUF --> REWARD_GATE
  REWARD_GATE --> PRM
  REWARD_GATE --> JUDGE
  PRM --> GRPO_LOSS
  JUDGE --> GRPO_LOSS
  GRPO_LOSS --> OPT --> MEGA
  MEGA -->|"weight sync"| SGLANG_SERVE
  HF_CKPT --> BRIDGE --> MEGA_CKPT -.-> MEGA
  MEGA --> WB
  MEGA --> RAY_DASH
```

---

## Component Reference

### Track 1 — Personal Agent

| Component | File | Role |
|-----------|------|------|
| API Server | `openclaw-rl/openclaw_api_server.py` | OpenAI-compatible proxy; intercepts conversations |
| Rollout Worker | `openclaw-rl/openclaw_rollout.py` | Bridges API server to SLIME training buffer |
| OPD Server | `openclaw-opd/openclaw_opd_api_server.py` | Adds hindsight hint extraction + teacher query |
| Combined Server | `openclaw-combine/openclaw_combine_api_server.py` | Unified RL + OPD signal collection |
| Combined Loss | `openclaw-combine/combine_loss.py` | `w_rl × GRPO + w_opd × teacher` advantage |
| Top-K Loss | `openclaw-opd/topk_distillation_loss.py` | Reverse KL over teacher's top-K distribution |
| Cloud Entry | `openclaw-tinker/run.py` | `--method {rl,opd,combine}` for Tinker cloud |

### Track 2 — General Agentic RL

| Domain | Entry Point | Environment |
|--------|-------------|-------------|
| Tool-Call / Math | `toolcall-rl/generate_with_retool.py` | Python sandbox (`tool_sandbox.py`) |
| Terminal | `terminal-rl/generate.py` | Docker containers via `pool_server.py` |
| GUI | `gui-rl/generate_with_gui.py` | Cloud VMs via `env_pool_server.py` |
| SWE-Bench | `swe-rl/generate_with_swe_remote.py` | Docker repos via `swe_env_pool_server.py` |

### Shared Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Rollout Buffer | SLIME + Ray | Async trajectory collection & queuing |
| PRM | SGLang engine | Per-step majority-vote reward (`--prm-m`) |
| Policy Trainer | Megatron-LM | TP/PP/EP distributed GRPO training |
| Inference Engine | SGLang | Fast token generation during rollout |
| Experiment Tracking | Weights & Biases | Loss, reward, eval metrics |
| Cluster Mgmt | Ray 2.54 | Distributed actor system |

---

## Data & Control Flow

```
User / Environment
        │
        ▼
  API Server / Env Pool
  (FastAPI + SGLang)
        │ trajectories
        ▼
  Rollout Buffer (SLIME/Ray)
        │
        ├──▶ PRM (step-wise reward, majority vote)
        │
        └──▶ Outcome Judge (episode reward)
                │
                ▼
          GRPO Advantage Estimator
          A_i = (r_i − mean(r)) / (std(r) + ε)
                │
                ▼
          Megatron-LM Trainer
          PPO clip [0.2, 0.28] + KL reg
                │
                ▼
          Updated Policy Weights
          ──▶ synced back to SGLang engine
```

---

## Parallelism Strategy

| Model Size | Nodes | GPUs/node | TP | PP | EP |
|-----------|-------|-----------|----|----|-----|
| 4B | 1 | 8 | 4 | 1 | 1 |
| 8B | 1 | 8 | 4 | 1 | 1 |
| 32B | 4 | 8 | 8 | 1 | 1 |
| 70B | 8 | 8 | 8 | 2 | 1 |
| MoE (DeepSeek) | 8+ | 8 | 1 | 8 | 32 |

---

*Diagram files:*
- *PNG (high-res, 1.7 MB):* `openclaw-rl-architecture.png`
- *PDF (print quality, 3.0 MB):* `openclaw-rl-architecture.pdf`
- *Mermaid source:* `openclaw-rl-architecture.mmd`
