# OpenClaw-RL — Process Reward Model (PRM) Guide

> How the PRM evaluates, scores, and gates training signal turn-by-turn.

---

## What the PRM Is

The PRM is **not a fine-tuned classifier**. It is a regular language model prompted to act as a judge. It reads what the agent said at turn N and what happened next (turn N+1), reasons step-by-step, and outputs one of three scores inside `\boxed{}`:

| Score | Meaning | When |
|-------|---------|------|
| `\boxed{1}` | Good | Task progressed — user moves on, says thanks, tool returned success |
| `\boxed{-1}` | Bad | Failure signal — user asks to redo/retry, correction request, environment error |
| `\boxed{0}` | Neutral | Ambiguous — unrelated follow-up, can't tell if the turn succeeded |

---

## The 5-Stage Flow Per Turn

```
Turn N completes
      │
      │  agent response saved → _pending_turn_data[session_id][turn_num]
      │
      ↓
Turn N+1 request arrives  ← this is the "next state"
      │
      │  _flush_pending_record() extracts the first message of the
      │  new request as evidence of what happened after turn N
      │
      ↓
_fire_prm_scoring()
      │
      │  builds judge prompt:
      │    [SYSTEM: scoring rubric]
      │    [USER: "## Assistant output\n..." + "## Next state [role: ...]\n..."]
      │
      ↓
m concurrent oMLX/SGLang calls  (default m=3, asyncio.gather)
      │
      │  each call returns a reasoning chain ending in \boxed{1}, \boxed{-1}, or \boxed{0}
      │  wall time = single inference time (all m calls run in parallel)
      │
      ↓
_majority_vote(scores)
      │
      │  majority wins → final score
      │  tie           → 0.0  (abstain)
      │
      ↓
loss_mask decision  (openclaw_api_server.py:615–629)
      │
      score ≠ 0 and has_next_state  →  loss_mask = [1, 1, ...]  ← train on this turn
      score = 0                      →  loss_mask = [0, 0, ...]  ← masked out, gradient = 0
      no next_state (last turn)      →  loss_mask = [0, ...]     ← no evidence, excluded
```

---

## The Judge Prompt

Source: [`openclaw_api_server.py:75–117`](openclaw_api_server.py#L75-L117)

```
SYSTEM:
  You are a process reward model (PRM) evaluating an AI assistant.
  You will see the assistant's output and the subsequent next state.
  Your task: decide whether the assistant's output successfully fulfilled
  the user's intent at that step, using the next state as evidence.

  ## Understanding the next state's role
  - role='user': A reply from the user.
  - role='tool': The return value of a tool the assistant invoked.
    This content was NOT available before the assistant's action.
    A successful, non-error tool output means the assistant's action
    worked correctly and should be scored positively.

  ## Scoring rules
  - \boxed{1} (good): next state shows task progressed as expected
  - \boxed{-1} (bad): redo/retry request, correction, environment error
  - \boxed{0} (neutral): ambiguous, insufficient information

  ## Important
  A change request IS negative feedback — treat it as \boxed{-1},
  NOT as a neutral new instruction.

  Think step-by-step, then give your final score inside \boxed{}.

USER:
  ## Assistant output
  <agent's response at turn N>

  ## Next state [role: user | tool]
  <first message of turn N+1>

  First, classify the next state: is it (a) positive progression,
  (b) a correction / redo / change request, or (c) ambiguous?
  Then assign \boxed{1}, \boxed{-1}, or \boxed{0}.
```

The `role` field matters: a `role=tool` next state means the tool result **exists because** the agent called it, so a clean non-error result is positive evidence.

---

## Majority Vote

Source: [`openclaw_api_server.py:130–138`](openclaw_api_server.py#L130-L138)

```python
def _majority_vote(scores: list[int | None]) -> float:
    valid = [s for s in scores if s is not None]
    if not valid:
        return 0.0
    counter = Counter(valid)
    top = counter.most_common(1)[0]
    if list(counter.values()).count(top[1]) > 1:  # tie
        return 0.0
    return float(top[0])
```

- `None` scores (inference failures) are discarded before voting
- A tie between two values resolves to `0.0` (abstain, not forced)
- With `m=3`: need at least 2 votes for the same value to win

---

## How It Gates Training

Source: [`openclaw_api_server.py:615–629`](openclaw_api_server.py#L615-L629)

```python
exclude = not has_next_state or score == 0.0

# At-least-one guarantee: if this session has produced zero
# effective samples and this turn was PRM-evaluated (just scored 0),
# promote it so the session is never entirely wasted.
if exclude and has_next_state and self._session_effective.get(session_id, 0) == 0:
    exclude = False

sample.loss_mask = [0] * len(response_ids) if exclude else [1] * len(response_ids)
sample.reward = {"score": score}
```

The `loss_mask` controls whether GRPO sees this turn:

```
score = +1  →  loss_mask = [1, 1, ...]  positive advantage  →  reinforce
score = -1  →  loss_mask = [1, 1, ...]  negative advantage  →  suppress
score =  0  →  loss_mask = [0, 0, ...]  masked out          →  no gradient
last turn   →  loss_mask = [0, 0, ...]  no next state       →  excluded
```

The `reward = {"score": score}` value is passed to the GRPO advantage estimator, which computes per-group normalized advantages:

```
A_i = (r_i − mean(r)) / (std(r) + ε)
```

---

## The Timing Dependency

The PRM **cannot fire until the next turn arrives** — it needs the next state as evidence. This creates a natural pipeline:

```
Turn 1 response ready  →  buffered, waiting
Turn 2 request arrives →  Turn 1's PRM fires  (Turn 2 = next_state)
Turn 2 response ready  →  buffered, waiting
Turn 3 request arrives →  Turn 2's PRM fires  (Turn 3 = next_state)
...
Session ends           →  last turn excluded  (no next_state available)
```

The `_pending_turn_data` dict holds each turn's tokens, log-probs, and response text until its PRM task completes. Only then does the sample enter the output queue for the trainer.

New requests are **blocked** during a weight update (`submission_enabled` event is cleared). This prevents stale log-probs entering the buffer while the policy is being updated.

---

## Concurrency Model

```
asyncio event loop (FastAPI / uvicorn thread)
       │
       ├── _handle_request()        ← incoming turn, runs in event loop
       │       └── _fire_prm_scoring()
       │               └── asyncio.create_task(_prm_evaluate())
       │                       └── asyncio.gather(
       │                               _query_prm_once(0),   ─┐
       │                               _query_prm_once(1),    ├── m concurrent HTTP calls
       │                               _query_prm_once(2),   ─┘  to PRM server
       │                           )
       │
       └── _maybe_submit_ready_samples()
               └── checks: is PRM task done?
                       yes → _submit_turn_sample() → output_queue.put()
                       no  → wait for task.add_done_callback()
```

All `m` PRM calls are fired with `asyncio.gather` — wall time equals one inference call, not `m` calls.

---

## Configuration

Controlled via environment variables and CLI args:

| Variable / Arg | Default | Meaning |
|---|---|---|
| `--prm-enable` | `False` | Enable PRM scoring |
| `PRM_M` / `--prm-m` | `3` | Number of parallel votes per turn |
| `--prm-model-path` | same as policy | Path to PRM model |
| `--prm-temperature` | `0.6` | Sampling temperature for judge |
| `--prm-max-new-tokens` | `4096` | Max tokens for reasoning chain |
| `--prm-num-gpus` | — | GPUs allocated to PRM engine |
| `--prm-num-gpus-per-engine` | `2` | GPUs per PRM SGLang/oMLX instance |

**PRM server endpoint** is resolved from `--prm-router-ip` and `--prm-router-port` at startup:

```python
self._prm_url = f"http://{prm_ip}:{prm_port}/generate"
```

In the **MLX/oMLX stack**, this changes to:

```python
self._prm_url = "http://localhost:8000/v1/chat/completions"
# model slot: "Qwen3-4B-4bit" loaded as a second model in oMLX
```

The oMLX **prefix cache** is especially effective here: all `m` calls in one majority vote share the same long system prompt and assistant output prefix, so only the first call incurs full prefill cost.

---

## Observability

PRM scores are logged to two JSONL files when `OPENCLAW_RECORD_ENABLED=1`:

| File | Content |
|------|---------|
| `*_record.jsonl` | Per-turn: session, turn, messages, prompt/response text, next_state |
| `*_prm.jsonl` | Per-turn: session, turn, `score`, `votes` list, `representative_eval` text |

`representative_eval` is the full reasoning chain of one PRM call that agreed with the final majority score — useful for debugging why a turn was rated bad or good.

The average PRM score across a rollout batch is tracked as:

```python
extra_metrics = {"rollout/prm_eval_score": mean(eval_scores)}
```

and logged to W&B automatically.

---

## Key Source Locations

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `_build_prm_judge_prompt` | `openclaw_api_server.py` | 75 | Constructs the system+user judge prompt |
| `_parse_prm_score` | `openclaw_api_server.py` | 120 | Extracts `\boxed{N}` from model output |
| `_majority_vote` | `openclaw_api_server.py` | 130 | Resolves m votes to final score |
| `_query_prm_once` | `openclaw_api_server.py` | 376 | Single HTTP call to PRM server |
| `_prm_evaluate` | `openclaw_api_server.py` | 405 | Fires m calls, gathers, votes |
| `_fire_prm_scoring` | `openclaw_api_server.py` | 438 | Launches PRM as asyncio task |
| `_maybe_submit_ready_samples` | `openclaw_api_server.py` | 573 | Submits when PRM task is done |
| `_submit_turn_sample` | `openclaw_api_server.py` | 601 | Sets loss_mask, puts in queue |
