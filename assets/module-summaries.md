# OpenClaw-RL — Module Summaries 

## oMLX Server
**"The AI Brain"**
- OpenAI-compatible inference endpoint
- Serves policy model + PRM simultaneously
- Continuous batching, tiered KV cache (RAM → NVMe SSD)
- Hot-reloads LoRA weights without restart

---

## Rollout Buffer
**"The Memory"**
- Intercepts every conversation turn as a trajectory
- Async queue — waits for reward signal before releasing
- Formats data into training batches for mlx-tune
- Bridges inference ↔ training in real-time

---

## PRM Judge
**"The Critic"**
- Process Reward Model running as a second model slot in oMLX
- Evaluates quality of each turn (not just final output)
- Runs `m` independent evaluations → majority vote for robustness
- Produces scalar reward signal (+1 / −1 / 0)

---

## mlx-tune Trainer
**"The Learner"**
- Runs GRPO gradient updates on the policy model
- LoRA fine-tuning (r=16) — updates only ~1% of weights
- Advantage estimation: `A = (r − mean) / std` across a group of rollouts
- Exports checkpoints as LoRA-only, merged, or GGUF

---

## OpenClaw App (Track 1)
**"Learn from Conversations"**
- Normal chat UI — model trains from your actual usage
- No labels needed: thumbs up/down = reward signal
- Supports Binary RL, OPD (teacher distillation), or Combined

---

## Agentic Tasks (Track 2)
**"Learn by Doing"**
- **Tool-call / Math** — generates + executes Python code, graded on answer correctness
- **Terminal** — shell agent running real bash commands in sandboxed Docker
- **GUI** — desktop automation with Qwen3-VL visual model
- **SWE-Bench** — fixes real GitHub issues, graded on test pass rate

---

## Apple Silicon — Unified Memory
**"The Platform Advantage"**
- Single Mac replaces a multi-GPU Linux cluster
- CPU + GPU share the same memory pool (up to 512 GB)
- 4-bit quantization: run 32B models on 64 GB, 70B on 128 GB
- No CUDA, no Ray cluster, no cloud required
