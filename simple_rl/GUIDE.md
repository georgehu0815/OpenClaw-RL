# Simple-RL — Online Agent RL: Serving & Training Guide

> **Apple Silicon only.**  No Ray, no CAMEL, no remote workers — just
> `asyncio` + Docker + an OpenAI-compatible model server (oMLX / mlx-lm).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Repository Layout](#3-repository-layout)
4. [Step 1 — Install Dependencies](#step-1--install-dependencies)
5. [Step 2 — Prepare the Task Dataset](#step-2--prepare-the-task-dataset)
6. [Step 3 — Start the Policy Server (oMLX)](#step-3--start-the-policy-server-omlx)
7. [Step 4 — (Optional) Start the PRM Server](#step-4--optional-start-the-prm-server)
8. [Step 5 — Verify the Environment](#step-5--verify-the-environment)
9. [Step 6 — Run Unit Tests](#step-6--run-unit-tests)
10. [Step 7 — Run Integration Tests](#step-7--run-integration-tests)
11. [Step 8 — Dry-Run Training](#step-8--dry-run-training)
12. [Step 9 — Full Training Run](#step-9--full-training-run)
13. [Step 10 — Wire mlx-tune for Weight Updates](#step-10--wire-mlx-tune-for-weight-updates)
14. [Configuration Reference](#configuration-reference)
15. [Data Flow — One Episode](#data-flow--one-episode)
16. [GRPO Advantage Computation](#grpo-advantage-computation)
17. [Adding Custom Tasks](#adding-custom-tasks)
18. [Memory Layout (64 GB Mac)](#memory-layout-64-gb-mac)
19. [Troubleshooting](#troubleshooting)

---

## 1. Architecture Overview

```
Architecture
────────────
  dataset (JSONL)
      │
      ▼
  Async RL Loop   ── dispatches tasks ──▶  AgentLoop × MAX_CONCURRENT
      │                                        │
      │                               LocalEnvPool (Docker)
      │                                        │
      ◀── Trajectory (messages + score) ───────┘
      │
      ▼
  RolloutBuffer  (asyncio.Queue)
      │   accumulate N_SAMPLES_PER_PROMPT per task
      │   compute GRPO advantages
      ▼
  GRPOBatch  ──▶  _submit_to_mlx_tune()  [stub — wire to mlx-tune]

┌─────────────────────────────────────────────────────────┐
│                      train_async.py                     │
│   asyncio event loop  —  no Ray, no remote workers      │
│                                                         │
│   task ──▶ run_episode() ──▶ Trajectory                 │
│             │    ▲                │                     │
│             │    │ obs            ▼                     │
│         agent_loop.py       rollout_buffer.py           │
│             │                    │                     │
│   POST /v1/chat/completions  GRPO advantages            │
│             │                    │                     │
│     ┌───────┴──────┐     ┌───────▼──────┐              │
│     │  oMLX Policy │     │ mlx-tune     │              │
│     │  :8080       │◀────│ GRPOTrainer  │              │
│     └──────────────┘     └──────────────┘              │
│                                                         │
│   local_env_pool.py                                     │
│     └─▶ TerminalEnv (Docker container per episode)      │
└─────────────────────────────────────────────────────────┘

train_async.py
└─ AsyncIO Event Loop
   (single process, no Ray / no remote workers)

   Task
   └─▶ run_episode()
        └─▶ Trajectory
             ├─ Observations
             │    ▲
             │    │
             │  agent_loop.py
             │    │
             │    └─ POST /v1/chat/completions
             │
             └─ rollout_buffer.py
                  └─ GRPO Advantages
                        │
                        ▼
                 mlx-tune
                 └─ GRPOTrainer
                        │
                        ▼
                 oMLX Policy Server
                 (localhost:8080)

Environment Layer
└─ local_env_pool.py
    └─ TerminalEnv
        (Docker container per episode)


```
flowchart TD

%% =======================
%% TRAINING ORCHESTRATOR
%% =======================

A[train_async.py<br>AsyncIO Event Loop] -->|schedule tasks| B(run_episode)

%% =======================
%% ENVIRONMENT LAYER
%% =======================

subgraph ENV_POOL[local_env_pool.py]
    C1[TerminalEnv<br>Docker Container]
end

B --> C1

%% =======================
%% AGENT INTERACTION LOOP
%% =======================

subgraph AGENT_LOOP[agent_loop.py]
    D1[Observations]
    D2[Action Request]
    D3[POST /v1/chat/completions]
end

C1 --> D1
D1 --> D2
D2 --> D3

%% =======================
%% POLICY SERVER (oMLX)
%% =======================

subgraph POLICY[oMLX Policy Server]
    E1[mlx Policy]
    E2[localhost:8080]
end

D3 --> E2
E2 --> E1
E1 -->|Response| D3

%% =======================
%% TRAJECTORY STORAGE
%% =======================

subgraph ROLLOUT[rollout_buffer.py]
    F1[Trajectory]
    F2[Rewards]
    F3[Advantages]
end

D3 --> F1
C1 --> F2

F1 --> F3
F2 --> F3

%% =======================
%% TRAINING LOOP
%% =======================

subgraph TRAINER[mlx-tune GRPOTrainer]
    G1[Compute Advantages]
    G2[Policy Update]
end

F3 --> G1
G1 --> G2
G2 --> E1

%% =======================
%% EPISODE LOOPBACK
%% =======================

G2 -->|updated policy| A
```
**Key design decisions:**

| Old terminal-rl | simple-rl replacement |
|---|---|
| CAMEL Agent | plain `asyncio` + `openai` client |
| Router Server | `LocalEnvPool` (in-process) |
| Pool Server | `asyncio.Semaphore` slot limiter |
| Remote Docker workers | local `docker run` subprocess |
| Ray object store | `asyncio.Queue` |
| SGLang-specific adapter | any OpenAI-compat server |

---

## 2. Prerequisites

| Requirement | Check |
|---|---|
| macOS with Apple Silicon (M1/M2/M3/M4) | `uname -m` → `arm64` |
| Python 3.12+ | `python3 --version` |
| Docker Desktop | `docker info` (must show daemon running) |
| `openai` Python package ≥ 1.0 | `pip show openai` |
| `mlx-lm` or `oMLX` for model serving | `mlx_lm.server --help` |
| `pytest` + `pytest-asyncio` | `pytest --version` |

Install missing Python packages:

```bash
pip install openai wandb pytest pytest-asyncio
```

---

## 3. Repository Layout

```
simple_rl/
├── config.py               # All knobs — override via env vars
├── terminal_env.py         # Single Docker container wrapper
├── local_env_pool.py       # Bounded in-process container pool
├── agent_loop.py           # Asyncio multi-turn agent (replaces CAMEL)
├── rollout_buffer.py       # asyncio.Queue + GRPO advantage
├── prm_client.py           # Optional PRM step-scoring client
├── train_async.py          # Top-level RL training entry point
├── run.sh                  # Startup / convenience script
├── data/
│   └── sample_tasks.jsonl  # 8 ready-to-use bash tasks
└── tests/
    ├── conftest.py
    ├── test_terminal_env.py
    ├── test_local_env_pool.py
    ├── test_agent_loop.py
    └── test_rollout_buffer.py
```

---

## Step 1 — Install Dependencies

```bash
# From the OpenClaw-RL repo root
pip install openai wandb pytest pytest-asyncio

# Verify
python3 -c "import openai; print('openai', openai.__version__)"
pytest --version
docker --version
```

Expected output:
```
openai 2.6.1
pytest 8.x.x
Docker version 28.x.x
```

---

## Step 2 — Prepare the Task Dataset

Tasks are JSONL files where each line is a JSON object with:

| Field | Required | Description |
|---|---|---|
| `task_name` | yes | Unique identifier |
| `instruction` | yes | Natural-language prompt sent to the agent |
| `grader` | yes | Bash script run inside the container; must `echo` a float 0.0–1.0 |
| `docker_image` | no | Override per-task (default: `ubuntu:22.04`) |

### Using the built-in sample dataset

```bash
# 8 tasks already provided:
cat simple_rl/data/sample_tasks.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    t = json.loads(line)
    print(f\"  {t['task_name']:20s}  {t['instruction'][:60]}\")"
```

Example output:
```
  hello_file           Create a file at /tmp/hello.txt containing exactly ...
  count_words          Count the number of words in the sentence 'The quic...
  sorted_numbers       Create /tmp/numbers.txt containing the integers 1 t...
  find_python          Find the path to the python3 binary and write it to ...
  ...
```

### Writing your own tasks

```jsonl
{"task_name": "my_task",
 "instruction": "Create /tmp/result.txt containing the MD5 hash of the string 'hello'",
 "grader": "[ -f /tmp/result.txt ] && grep -q '5d41402abc4b2a76b9719d911017c592' /tmp/result.txt && echo 1.0 || echo 0.0",
 "docker_image": "ubuntu:22.04"}
```

**Grader protocol:** The grader is a bash script executed *inside* the same Docker container the agent used. The last non-empty line it prints must be a float in `[0.0, 1.0]`.

```bash
# Test your grader locally:
docker run --rm ubuntu:22.04 bash -c "
  echo 'Hello World' > /tmp/hello.txt
  [ -f /tmp/hello.txt ] && grep -qF 'Hello World' /tmp/hello.txt && echo 1.0 || echo 0.0
"
# Expected: 1.0
```

### Converting Terminal-Bench tasks

```bash
# Download seta_env dataset
python3 terminal-rl/data_utils/download.py seta_env

DATASET_DIR=./terminal-rl/dataset \
python3 terminal-rl/data_utils/convert_task_to_dataset.py \
  --tasks_dir ./terminal-rl/dataset/seta_env \
  --output_dir ./simple_rl/data/seta_env_converted

# Convert to JSONL
DATASET_DIR=./terminal-rl/dataset \
python3 terminal-rl/data_utils/convert_task_to_dataset.py \
  --tasks_dir ./terminal-rl/dataset/seta_env \
  --output_dir ./simple_rl/data/seta_env_converted
```

---

## Step 3 — Start the Policy Server (oMLX)

The agent calls `POST /v1/chat/completions` — any OpenAI-compatible server works.

### Option A — mlx_lm (recommended for development)

```bash
# Install
pip install mlx-lm

# Serve Qwen3-8B quantized (≈5 GB unified memory)
mlx_lm.server \
  --model mlx-community/Qwen3-8B-4bit \
  --port 8080 \
  --host 0.0.0.0

# Verify
curl -s http://localhost:8080/v1/models  | python3 -m json.tool
curl -s http://localhost:8080/v1/models -H "Authorization: Bearer 1111" | python3 -m json.tool
```

### Option B — oMLX (for training with weight sync)

```bash
# Start oMLX policy slot (follows oMLX documentation)
omlx serve \
  --model mlx-community/Qwen3-8B-4bit \
  --port 8080 \
  --slot policy
```

### Option C — Any OpenAI-compatible server

```bash
# Point POLICY_URL at any running server
export POLICY_URL=http://localhost:8080/v1
export POLICY_MODEL=<model-name-as-reported-by-/v1/models>
```

**Quick smoke test:**

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer 1111" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-0.8B-8bit",
    "messages": [{"role":"user","content":"Say hello"}],
    "max_tokens": 20
  }' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

---

## Step 4 — (Optional) Start the PRM Server

The PRM (Process Reward Model) scores each bash step, not just the final outcome. Enable it for richer reward signal on long-horizon tasks.

```bash
# Serve a smaller judge model on port 8081
mlx_lm.server \
  --model mlx-community/Qwen3-4B-4bit \
  --port 8081

# Enable in training:
export PRM_ENABLE=1
export PRM_URL=http://localhost:8081/v1
export PRM_MODEL=Qwen3-4B-4bit
export PRM_M=3     # majority-vote samples per step
```

**Memory budget with both slots (64 GB Mac):**

| Component | Memory |
|---|---|
| Policy slot (Qwen3-8B 4-bit) | ~5 GB |
| PRM slot (Qwen3-4B 4-bit) | ~3 GB |
| mlx-tune LoRA adapters + gradients | ~10 GB |
| KV cache (hot tier) | ~8 GB |
| Docker containers (4 × ~0.5 GB) | ~2 GB |
| **Total** | **~28 GB** |

---

## Step 5 — Verify the Environment

```bash
cd /Volumes/ExternalSSD/train/OpenClaw-RL

# 1. Docker daemon
docker info | grep -E "Server Version|Operating System"

# 2. Pull the base image (one-time, ~30 MB)
docker pull ubuntu:22.04

# 3. Smoke-test a container
docker run --rm ubuntu:22.04 bash -c "echo 'Docker OK'"

# 4. Policy server reachable
curl -s http://localhost:8080/v1/models -H "Authorization: Bearer 1111" | grep -o '"id":"[^"]*"'

# 5. Python imports
python3 -c "from simple_rl.agent_loop import run_episode; print('imports OK')"
```

---

## Step 6 — Run Unit Tests

Unit tests mock all external dependencies (Docker, OpenAI). They must pass before any real run.

```bash
cd /Volumes/ExternalSSD/train/OpenClaw-RL

pytest simple_rl/tests/ -m "not integration" -v
```

**Expected:** `42 passed` in < 5 seconds.

```
PASSED test_agent_loop.py::TestParseBashCmd::test_xml_tags
PASSED test_agent_loop.py::TestParseBashCmd::test_markdown_fence
PASSED test_agent_loop.py::TestTrimMessages::test_trims_old_turns
PASSED test_agent_loop.py::TestRunEpisode::test_task_complete_signal_ends_loop
PASSED test_agent_loop.py::TestRunEpisode::test_max_turns_respected
PASSED test_agent_loop.py::TestRunEpisode::test_close_called_on_exception
PASSED test_local_env_pool.py::TestLocalEnvPoolUnit::test_max_concurrent_blocks
PASSED test_rollout_buffer.py::TestRolloutBuffer::test_grpo_advantage_formula
PASSED test_rollout_buffer.py::TestRolloutBuffer::test_independent_tasks_dont_mix
... (42 total)
```
==================================================================================== test session starts ====================================================================================
platform darwin -- Python 3.12.10, pytest-8.3.4, pluggy-1.5.0 -- /usr/local/bin/python3
cachedir: .pytest_cache
rootdir: /Volumes/ExternalSSD/train/OpenClaw-RL/simple_rl
configfile: pytest.ini
plugins: devtools-0.12.2, timeout-2.4.0, logfire-3.9.0, asyncio-1.3.0, anyio-4.7.0, docker-3.1.1
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collected 47 items / 5 deselected / 42 selected                                                                                                                                             

simple_rl/tests/test_agent_loop.py::TestParseBashCmd::test_xml_tags <- ../simple-rl/tests/test_agent_loop.py PASSED                                                                   [  2%]
simple_rl/tests/test_agent_loop.py::TestParseBashCmd::test_markdown_fence <- ../simple-rl/tests/test_agent_loop.py PASSED                                                             [  4%]
simple_rl/tests/test_agent_loop.py::TestParseBashCmd::test_sh_fence <- ../simple-rl/tests/test_agent_loop.py PASSED                                                                   [  7%]
simple_rl/tests/test_agent_loop.py::TestParseBashCmd::test_no_command <- ../simple-rl/tests/test_agent_loop.py PASSED                                                                 [  9%]
simple_rl/tests/test_agent_loop.py::TestParseBashCmd::test_case_insensitive_tag <- ../simple-rl/tests/test_agent_loop.py PASSED                                                       [ 11%]
simple_rl/tests/test_agent_loop.py::TestParseBashCmd::test_multiline_cmd <- ../simple-rl/tests/test_agent_loop.py PASSED                                                              [ 14%]
simple_rl/tests/test_agent_loop.py::TestTrimMessages::test_within_budget <- ../simple-rl/tests/test_agent_loop.py PASSED                                                              [ 16%]
simple_rl/tests/test_agent_loop.py::TestTrimMessages::test_trims_old_turns <- ../simple-rl/tests/test_agent_loop.py PASSED                                                            [ 19%]
simple_rl/tests/test_agent_loop.py::TestTrimMessages::test_short_messages_unchanged <- ../simple-rl/tests/test_agent_loop.py PASSED                                                   [ 21%]
simple_rl/tests/test_agent_loop.py::TestRunEpisode::test_task_complete_signal_ends_loop <- ../simple-rl/tests/test_agent_loop.py PASSED                                               [ 23%]
simple_rl/tests/test_agent_loop.py::TestRunEpisode::test_max_turns_respected <- ../simple-rl/tests/test_agent_loop.py PASSED                                                          [ 26%]
simple_rl/tests/test_agent_loop.py::TestRunEpisode::test_no_bash_cmd_returns_hint <- ../simple-rl/tests/test_agent_loop.py PASSED                                                     [ 28%]
simple_rl/tests/test_agent_loop.py::TestRunEpisode::test_policy_error_records_and_returns <- ../simple-rl/tests/test_agent_loop.py PASSED                                             [ 30%]
simple_rl/tests/test_agent_loop.py::TestRunEpisode::test_token_tracking <- ../simple-rl/tests/test_agent_loop.py PASSED                                                               [ 33%]
simple_rl/tests/test_agent_loop.py::TestRunEpisode::test_close_called_on_exception <- ../simple-rl/tests/test_agent_loop.py PASSED                                                    [ 35%]
simple_rl/tests/test_local_env_pool.py::TestLocalEnvPoolUnit::test_allocate_and_close <- ../simple-rl/tests/test_local_env_pool.py PASSED                                             [ 38%]
simple_rl/tests/test_local_env_pool.py::TestLocalEnvPoolUnit::test_exec_delegates_to_env <- ../simple-rl/tests/test_local_env_pool.py PASSED                                          [ 40%]
simple_rl/tests/test_local_env_pool.py::TestLocalEnvPoolUnit::test_evaluate_uses_task_grader <- ../simple-rl/tests/test_local_env_pool.py PASSED                                      [ 42%]
simple_rl/tests/test_local_env_pool.py::TestLocalEnvPoolUnit::test_max_concurrent_blocks <- ../simple-rl/tests/test_local_env_pool.py PASSED                                          [ 45%]
simple_rl/tests/test_local_env_pool.py::TestLocalEnvPoolUnit::test_unknown_lease_raises <- ../simple-rl/tests/test_local_env_pool.py PASSED                                           [ 47%]
simple_rl/tests/test_local_env_pool.py::TestLocalEnvPoolUnit::test_stop_closes_all_containers <- ../simple-rl/tests/test_local_env_pool.py PASSED                                     [ 50%]
simple_rl/tests/test_rollout_buffer.py::TestGRPOBatch::test_mean_score <- ../simple-rl/tests/test_rollout_buffer.py PASSED                                                            [ 52%]
simple_rl/tests/test_rollout_buffer.py::TestGRPOBatch::test_mean_advantage <- ../simple-rl/tests/test_rollout_buffer.py PASSED                                                        [ 54%]
simple_rl/tests/test_rollout_buffer.py::TestGRPOBatch::test_empty_batch <- ../simple-rl/tests/test_rollout_buffer.py PASSED                                                           [ 57%]
simple_rl/tests/test_rollout_buffer.py::TestRolloutBuffer::test_returns_none_before_full <- ../simple-rl/tests/test_rollout_buffer.py PASSED                                          [ 59%]
simple_rl/tests/test_rollout_buffer.py::TestRolloutBuffer::test_returns_batch_when_full <- ../simple-rl/tests/test_rollout_buffer.py PASSED                                           [ 61%]
simple_rl/tests/test_rollout_buffer.py::TestRolloutBuffer::test_grpo_advantage_formula <- ../simple-rl/tests/test_rollout_buffer.py PASSED                                            [ 64%]
simple_rl/tests/test_rollout_buffer.py::TestRolloutBuffer::test_grpo_advantage_identical_scores <- ../simple-rl/tests/test_rollout_buffer.py PASSED                                   [ 66%]
simple_rl/tests/test_rollout_buffer.py::TestRolloutBuffer::test_independent_tasks_dont_mix <- ../simple-rl/tests/test_rollout_buffer.py PASSED                                        [ 69%]
simple_rl/tests/test_rollout_buffer.py::TestRolloutBuffer::test_put_and_get_batch <- ../simple-rl/tests/test_rollout_buffer.py PASSED                                                 [ 71%]
simple_rl/tests/test_rollout_buffer.py::TestRolloutBuffer::test_prm_scoring_called <- ../simple-rl/tests/test_rollout_buffer.py PASSED                                                [ 73%]
simple_rl/tests/test_rollout_buffer.py::TestRolloutBuffer::test_queue_depth <- ../simple-rl/tests/test_rollout_buffer.py PASSED                                                       [ 76%]
simple_rl/tests/test_terminal_env.py::TestTerminalEnvUnit::test_start_success <- ../simple-rl/tests/test_terminal_env.py PASSED                                                       [ 78%]
simple_rl/tests/test_terminal_env.py::TestTerminalEnvUnit::test_start_failure_raises <- ../simple-rl/tests/test_terminal_env.py PASSED                                                [ 80%]
simple_rl/tests/test_terminal_env.py::TestTerminalEnvUnit::test_exec_not_running <- ../simple-rl/tests/test_terminal_env.py PASSED                                                    [ 83%]
simple_rl/tests/test_terminal_env.py::TestTerminalEnvUnit::test_exec_returns_output <- ../simple-rl/tests/test_terminal_env.py PASSED                                                 [ 85%]
simple_rl/tests/test_terminal_env.py::TestTerminalEnvUnit::test_exec_timeout_returns_message <- ../simple-rl/tests/test_terminal_env.py PASSED                                        [ 88%]
simple_rl/tests/test_terminal_env.py::TestTerminalEnvUnit::test_evaluate_parses_float <- ../simple-rl/tests/test_terminal_env.py PASSED                                               [ 90%]
simple_rl/tests/test_terminal_env.py::TestTerminalEnvUnit::test_evaluate_clamps_to_01 <- ../simple-rl/tests/test_terminal_env.py PASSED                                               [ 92%]
simple_rl/tests/test_terminal_env.py::TestTerminalEnvUnit::test_evaluate_no_float_returns_zero <- ../simple-rl/tests/test_terminal_env.py PASSED                                      [ 95%]
simple_rl/tests/test_terminal_env.py::TestTerminalEnvUnit::test_close_not_running <- ../simple-rl/tests/test_terminal_env.py PASSED                                                   [ 97%]
simple_rl/tests/test_terminal_env.py::TestTerminalEnvUnit::test_close_running <- ../simple-rl/tests/test_terminal_env.py PASSED                                                       [100%]

=============================================================

---

## Step 7 — Run Integration Tests

Integration tests spin real Docker containers. Requires Docker Desktop running.

```bash
# Pull base image first (one-time)
docker pull ubuntu:22.04

# Run all integration tests
pytest simple_rl/tests/ -m "integration" -v
```

**Expected:** `5 passed` in ~30–60 seconds.

```
PASSED test_terminal_env.py::TestTerminalEnvIntegration::test_full_lifecycle
PASSED test_terminal_env.py::TestTerminalEnvIntegration::test_grader_fail_when_file_missing
PASSED test_terminal_env.py::TestTerminalEnvIntegration::test_output_cap
PASSED test_local_env_pool.py::TestLocalEnvPoolIntegration::test_real_exec_and_evaluate
PASSED test_local_env_pool.py::TestLocalEnvPoolIntegration::test_parallel_allocations
```

What each integration test verifies:

| Test | Verifies |
|---|---|
| `test_full_lifecycle` | Container starts, agent command executes, grader scores 1.0, container removed |
| `test_grader_fail_when_file_missing` | Grader returns 0.0 when agent did nothing |
| `test_output_cap` | `exec()` caps output at 4096 bytes (no context overflow) |
| `test_real_exec_and_evaluate` | Full pool: allocate → exec → evaluate → close |
| `test_parallel_allocations` | Two isolated containers run in parallel without conflict |

---

## Step 8 — Dry-Run Training

A dry-run uses the real code path but with minimal settings to verify end-to-end connectivity quickly (no long waits).

```bash
cd /Volumes/ExternalSSD/train/OpenClaw-RL

POLICY_URL=http://localhost:8080/v1 \
bash simple_rl/run.sh --dry-run
```

Dry-run settings: `max_rounds=2`, `n_samples=2`, `rollout_batch_size=2`, `max_turns=3`.

**Expected log output:**

```
=== simple-rl startup ===
[✓] Docker daemon reachable
[✓] Policy server reachable at http://localhost:8080/v1

Configuration:
  POLICY_URL       = http://localhost:8080/v1
  POLICY_MODEL     = Qwen3-8B-4bit
  MAX_CONCURRENT   = 4
  N_SAMPLES        = 2
  ROLLOUT_BATCH    = 2
  MAX_TURNS        = 3
  MODE             = DRY-RUN (max_rounds=2)

2026-03-27 14:00:01 INFO  LocalEnvPool started (max_concurrent=4)
2026-03-27 14:00:03 INFO  Started container simple-rl-a1b2c3d4e5
2026-03-27 14:00:08 INFO  Episode done: task=hello_file score=1.000 turns=2 done=True elapsed=5.1s
2026-03-27 14:00:09 INFO  Batch ready: task=hello_file n=2 mean_score=0.750
2026-03-27 14:00:09 INFO  === Training round 1 | n_samples=4 mean_score=0.750 mean_adv=0.000 ===
2026-03-27 14:00:09 INFO  Batch written to logs/grpo_batch_r0001.jsonl
2026-03-27 14:00:12 INFO  === Training round 2 | n_samples=4 mean_score=0.625 mean_adv=0.000 ===
2026-03-27 14:00:12 INFO  Training finished. rounds=2 steps=8
```

**Inspect the GRPO batch output:**

```bash
python3 -c "
import json
with open('simple_rl/logs/grpo_batch_r0001.jsonl') as f:
    for line in f:
        s = json.loads(line)
        print(f\"task={s['task_id']:20s}  score={s['score']:.3f}  adv={s['advantage']:+.3f}  turns={s['n_turns']}\")
"
```
task=hello_file            score=1.000  adv=+1.208  turns=20
task=hello_file            score=1.000  adv=+1.208  turns=20
task=hello_file            score=1.000  adv=+1.208  turns=20
task=hello_file            score=0.000  adv=-0.725  turns=0
task=hello_file            score=0.000  adv=-0.725  turns=0
task=hello_file            score=0.000  adv=-0.725  turns=0
task=hello_file            score=0.000  adv=-0.725  turns=0
task=hello_file            score=0.000  adv=-0.725  turns=0
task=count_words           score=1.000  adv=+1.208  turns=20
task=count_words           score=1.000  adv=+1.208  turns=20
task=count_words           score=1.000  adv=+1.208  turns=20
task=count_words           score=0.000  adv=-0.725  turns=0
task=count_words           score=0.000  adv=-0.725  turns=0
task=count_words           score=0.000  adv=-0.725  turns=0
task=count_words           score=0.000  adv=-0.725  turns=0
task=count_words           score=0.000  adv=-0.725  turns=0
task=sorted_numbers        score=1.000  adv=+1.620  turns=20
task=sorted_numbers        score=1.000  adv=+1.620  turns=20
task=sorted_numbers        score=0.000  adv=-0.540  turns=20
task=sorted_numbers        score=0.000  adv=-0.540  turns=0
task=sorted_numbers        score=0.000  adv=-0.540  turns=0
task=sorted_numbers        score=0.000  adv=-0.540  turns=0
task=sorted_numbers        score=0.000  adv=-0.540  turns=0
task=sorted_numbers        score=0.000  adv=-0.540  turns=0
task=find_python           score=0.000  adv=+0.000  turns=20
task=find_python           score=0.000  adv=+0.000  turns=20
task=find_python           score=0.000  adv=+0.000  turns=16
task=find_python           score=0.000  adv=+0.000  turns=0
task=find_python           score=0.000  adv=+0.000  turns=0
task=find_python           score=0.000  adv=+0.000  turns=0
task=find_python           score=0.000  adv=+0.000  turns=0
task=find_python           score=0.000  adv=+0.000  turns=0
---

## Step 9 — Full Training Run

### Basic run

```bash
cd /Volumes/ExternalSSD/train/OpenClaw-RL

POLICY_URL=http://localhost:8080/v1 \
bash simple_rl/run.sh
```

### All options via environment variables

```bash
POLICY_URL=http://localhost:8080/v1   \  # oMLX / mlx_lm server
POLICY_MODEL=Qwen3-8B-4bit            \  # model name for /v1/chat/completions
MAX_CONCURRENT=4                      \  # parallel Docker containers
N_SAMPLES=8                           \  # trajectories per task for GRPO
ROLLOUT_BATCH=4                       \  # tasks per gradient update
MAX_TURNS=20                          \  # max bash turns per episode
DOCKER_IMAGE=ubuntu:22.04             \  # container image
DATASET=simple_rl/data/sample_tasks.jsonl \
LOG_DIR=simple_rl/logs                \
WANDB_PROJECT=terminal-rl-simple      \  # leave empty to disable W&B
bash simple_rl/run.sh
```

Configuration:
  POLICY_URL       = http://localhost:8080/v1
  POLICY_MODEL     = Qwen3.5-0.8B-8bit
  MAX_CONCURRENT   = 4
  N_SAMPLES        = 8
  ROLLOUT_BATCH    = 4
  MAX_TURNS        = 20
  DATASET          = /Volumes/ExternalSSD/train/OpenClaw-RL/simple_rl/data/sample_tasks.jsonl
  LOG_DIR          = /Volumes/ExternalSSD/train/OpenClaw-RL/simple_rl/logs

### Run for a fixed number of rounds

```bash
POLICY_URL=http://localhost:8080/v1 \
bash simple_rl/run.sh --rounds 50
```

### Run with PRM step scoring

```bash
# Terminal 1: policy server
mlx_lm.server --model mlx-community/Qwen3-8B-4bit --port 8080

# Terminal 2: PRM server
mlx_lm.server --model mlx-community/Qwen3-4B-4bit --port 8081

# Terminal 3: training
POLICY_URL=http://localhost:8080/v1 \
PRM_URL=http://localhost:8081/v1    \
PRM_ENABLE=1                        \
bash simple_rl/run.sh --prm
```

### Monitor training

```bash
# Live trajectory log
tail -f simple_rl/logs/trajectories.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    t = json.loads(line.strip())
    bar = '█' * int(t['score'] * 20) + '░' * (20 - int(t['score'] * 20))
    print(f\"step={t['step']:4d}  {t['task_name']:20s}  [{bar}]  {t['score']:.3f}  turns={t['n_turns']}\")
"

# GRPO batch statistics
for f in simple_rl/logs/grpo_batch_r*.jsonl; do
    python3 -c "
import json, statistics
scores = [json.loads(l)['score'] for l in open('$f')]
advs   = [json.loads(l)['advantage'] for l in open('$f')]
print(f'$f  n={len(scores)}  mean_score={statistics.mean(scores):.3f}  std={statistics.stdev(scores):.3f}')
"
done
```

---

## Step 10 — Wire mlx-tune for Weight Updates

The current `_submit_to_mlx_tune()` in [train_async.py](train_async.py) writes batches to disk and logs stats. To close the RL loop, replace it with real gradient updates:

### Option A — Subprocess call to mlx-tune script

```python
# In train_async.py, replace _submit_to_mlx_tune():

import subprocess

def _submit_to_mlx_tune(batches, round_num, log_dir):
    batch_path = Path(log_dir) / f"grpo_batch_r{round_num:04d}.jsonl"
    _write_batch_jsonl(batches, batch_path)   # keep the existing write logic

    result = subprocess.run(
        [
            "python3", "-m", "mlx_tune.grpo_train",
            "--batch",     str(batch_path),
            "--model",     os.getenv("POLICY_MODEL", "Qwen3-8B-4bit"),
            "--lora_rank", os.getenv("LORA_RANK", "16"),
            "--kl_coef",   os.getenv("KL_LOSS_COEF", "0.01"),
            "--output",    str(Path(log_dir) / "adapters"),
        ],
        check=True,
    )
    logger.info("mlx-tune round %d complete", round_num)
```

### Option B — In-process mlx-tune Python API

```python
# Assuming mlx-tune exposes a Python API:
from mlx_tune import GRPOTrainer

_trainer = None   # module-level singleton

def _get_trainer():
    global _trainer
    if _trainer is None:
        _trainer = GRPOTrainer(
            model_name=config.POLICY_MODEL,
            lora_rank=config.LORA_RANK,
            kl_loss_coef=config.KL_LOSS_COEF,
        )
    return _trainer

def _submit_to_mlx_tune(batches, round_num, log_dir):
    trainer = _get_trainer()
    grpo_data = [
        {"messages": s.trajectory.messages, "advantage": s.advantage}
        for b in batches
        for s in b.samples
    ]
    trainer.step(grpo_data)   # gradient update + in-place weight sync to oMLX
    logger.info("mlx-tune round %d complete", round_num)
```

### GRPO batch format (what mlx-tune receives)

Each line in `logs/grpo_batch_r*.jsonl`:

```json
{
  "task_id":    "hello_file",
  "advantage":  1.4142,
  "score":      1.0,
  "n_turns":    2,
  "turn_scores": [0.8, 0.9],
  "messages": [
    {"role": "system",    "content": "You are a terminal agent..."},
    {"role": "user",      "content": "Create /tmp/hello.txt..."},
    {"role": "assistant", "content": "<bash>\necho 'Hello World' > /tmp/hello.txt\n</bash>"},
    {"role": "user",      "content": "<observation>\n\n</observation>"},
    {"role": "assistant", "content": "TASK_COMPLETE"}
  ]
}
```

---

## Configuration Reference

All values are read from environment variables at startup.

| Variable | Default | Description |
|---|---|---|
| `POLICY_URL` | `http://localhost:8080/v1` | OpenAI-compat policy server |
| `POLICY_MODEL` | `Qwen3-8B-4bit` | Model name sent to `/v1/chat/completions` |
| `PRM_URL` | `http://localhost:8081/v1` | PRM server (optional) |
| `PRM_MODEL` | `Qwen3-4B-4bit` | PRM model name |
| `PRM_ENABLE` | `0` | `1` to enable step scoring |
| `PRM_M` | `3` | Majority-vote samples per step |
| `DOCKER_IMAGE` | `ubuntu:22.04` | Base image for episode containers |
| `MAX_CONCURRENT` | `4` | Parallel Docker containers (limited by RAM) |
| `IDLE_TIMEOUT` | `600` | Seconds before idle container is reaped |
| `EXEC_TIMEOUT` | `30.0` | Per-command timeout in seconds |
| `EVAL_TIMEOUT` | `60.0` | Grader script timeout in seconds |
| `MAX_TURNS` | `20` | Max bash turns per episode |
| `CONTEXT_LEN` | `16384` | Approx character budget before trimming |
| `N_SAMPLES_PER_PROMPT` | `8` | Group size for GRPO advantage normalisation |
| `ROLLOUT_BATCH_SIZE` | `4` | Groups per gradient update |
| `KL_LOSS_COEF` | `0.01` | KL penalty vs reference model |
| `LORA_RANK` | `16` | LoRA rank (`0` = full fine-tune) |
| `DATASET_PATH` | `data/sample_tasks.jsonl` | Input task file |
| `LOG_DIR` | `logs` | Output directory for trajectories and batches |
| `WANDB_PROJECT` | `` | W&B project name (empty = disabled) |

---

## Data Flow — One Episode

```
1.  train_async picks task from dataset (round-robin)
2.  run_episode(task, env_pool) starts
    │
    ├─ env_pool.allocate(task)
    │     └─▶ docker run --rm -d ubuntu:22.04 sleep infinity
    │
    ├─ Turn N  (repeated up to MAX_TURNS):
    │     a. Build messages [system + history]  ← trimmed to CONTEXT_LEN
    │     b. POST /v1/chat/completions          → oMLX Policy Slot
    │     c. Parse <bash>…</bash> from response
    │     d. docker exec <container> bash -c "<cmd>"  → stdout/stderr
    │     e. Append <observation> to messages
    │     f. If "TASK_COMPLETE" in response → exit loop
    │
    ├─ env_pool.evaluate(lease)
    │     └─▶ docker exec <container> bash -c "<grader script>"
    │              └─ prints float 0.0–1.0 on last line
    │
    └─ env_pool.close(lease)
          └─▶ docker rm -f <container>

3.  Trajectory (messages + score) → RolloutBuffer.add(task_id, traj)
4.  When N_SAMPLES_PER_PROMPT collected for a task:
      a. (optional) PRMClient scores each (cmd, obs) turn
      b. GRPO advantage: A_i = (r_i − mean(r)) / (std(r) + ε)
      c. GRPOBatch → training queue
5.  When ROLLOUT_BATCH_SIZE groups ready:
      └─ _submit_to_mlx_tune(batches)  →  gradient update  →  weight sync
```

---

## GRPO Advantage Computation

Within each group of `N` trajectories from the same task prompt:

```
scores  = [r_1, r_2, ..., r_N]          # episode scores 0.0–1.0
mean_r  = mean(scores)
std_r   = stdev(scores)

A_i     = (r_i − mean_r) / (std_r + ε)  # ε = 1e-8
```

This normalises rewards *within the group*, giving:
- Positive advantage → trajectory was better than the group average
- Negative advantage → trajectory was worse than average
- All-same scores → all advantages ≈ 0 (no learning signal)

**Example** — 4 trajectories on task `hello_file`:

| Sample | Score | Advantage |
|---|---|---|
| 1 | 1.0 | +1.34 |
| 2 | 1.0 | +1.34 |
| 3 | 0.0 | -1.34 |
| 4 | 0.5 | -0.45 |

The two successful trajectories get positive gradient reinforcement; the failures get negative.

---

## Adding Custom Tasks

### Minimal task (bash grader)

```json
{"task_name": "reverse_file",
 "instruction": "Create /tmp/rev.txt containing the reverse of 'abcdef' (i.e. 'fedcba').",
 "grader": "[ -f /tmp/rev.txt ] && [ \"$(cat /tmp/rev.txt | tr -d '[:space:]')\" = 'fedcba' ] && echo 1.0 || echo 0.0"}
```

### Partial-credit grader (0.0 / 0.5 / 1.0)

```json
{"task_name": "install_and_check",
 "instruction": "Install the 'tree' utility with apt-get, then run 'tree /usr/bin' and save the output to /tmp/tree_out.txt.",
 "grader": "HAS_TREE=$(which tree 2>/dev/null && echo 1 || echo 0); HAS_FILE=$([ -f /tmp/tree_out.txt ] && echo 1 || echo 0); echo \"$(python3 -c \"print(($HAS_TREE + $HAS_FILE) / 2.0)\")\""}
```

### Python grader (for richer checking)

```json
{"task_name": "csv_stats",
 "instruction": "Download the file at http://example.com/data.csv ... (no network in sandbox, use a generated file instead)",
 "grader": "python3 /tmp/grade.py"}
```

> **Tip:** Write the grader script to `/tmp/grade.py` as part of the `docker run` startup, or embed a heredoc in the grader field.

---

## Memory Layout (64 GB Mac)

| Component | Memory |
|---|---|
| oMLX Policy Slot (Qwen3-8B 4-bit) | ~5 GB |
| oMLX PRM Slot (Qwen3-4B 4-bit, optional) | ~3 GB |
| mlx-tune LoRA adapters + gradients | ~10 GB |
| oMLX KV cache (hot tier) | ~8 GB |
| Docker containers (4 × ~0.5 GB) | ~2 GB |
| macOS + other processes | ~5 GB |
| **Total** | **~33 GB** |

For a 32 GB Mac, reduce `MAX_CONCURRENT=2` and use the 4-bit model.

---

## Troubleshooting

### `Cannot connect to Docker daemon`

```bash
# Start Docker Desktop, then:
docker info
# If using Docker Engine directly:
sudo systemctl start docker
```

### `Policy API error: Connection refused`

```bash
# Check server is running on the right port
curl http://localhost:8080/v1/models
# Restart mlx_lm server if needed
mlx_lm.server --model mlx-community/Qwen3-8B-4bit --port 8080
```

### Agent never says `TASK_COMPLETE`

The model may need prompting guidance. The system prompt in [agent_loop.py](agent_loop.py) already instructs it. If it still loops:
- Reduce `MAX_TURNS` to force faster termination
- Check the model is following the `<bash>…</bash>` protocol
- Try a stronger model (Qwen3-32B, etc.)

### Score always 0.0

Test your grader script directly inside a container:

```bash
# Run agent commands manually
docker run -it --rm ubuntu:22.04 bash

# Then test grader:
bash -c "[ -f /tmp/hello.txt ] && echo 1.0 || echo 0.0"
```

### Container not removed after error

```bash
# List lingering containers
docker ps -a | grep simple-rl

# Force remove all
docker ps -a --format '{{.Names}}' | grep simple-rl | xargs docker rm -f
```

### GRPO advantage is NaN / Inf

Occurs when all trajectories in a group have the same score and `stdev=0`. The buffer adds `ε=1e-8` to prevent exact division by zero — the result is very small, not NaN. If you still see NaN, check that scores are actual floats (not Python `None`):

```bash
grep '"score": null' simple_rl/logs/trajectories.jsonl | head -5
```

### W&B not logging

```bash
wandb login
# or disable entirely:
unset WANDB_PROJECT
```
