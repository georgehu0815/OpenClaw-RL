"""
agent_loop.py — Asyncio multi-turn terminal agent.

Replaces CAMEL / CamelAgent from the old terminal-rl stack.
Uses the standard OpenAI client talking to an oMLX-served policy
at POLICY_URL (OpenAI-compatible /v1/chat/completions endpoint).

Tool protocol
─────────────
The model signals a bash command by wrapping it in <bash>…</bash> tags.
The observation (stdout+stderr) is fed back as:

    <observation>…</observation>

The model signals task completion with the literal word TASK_COMPLETE
anywhere in its response.

Context budget
──────────────
If the total message text exceeds CONTEXT_LEN characters the oldest
tool-turns (assistant + observation pairs) are dropped to stay within
budget.  The system prompt and first user turn are always kept.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import openai

from . import config

logger = logging.getLogger(__name__)

# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a terminal agent running inside a Docker container (Linux).
Solve the given task by executing bash commands.

To run a command, wrap it in XML tags:
<bash>
your command here
</bash>

Rules:
- Run one command at a time; wait for its output before continuing.
- Inspect the observation before deciding the next step.
- When the task is fully complete, write exactly: TASK_COMPLETE
- If a command hangs, send a signal (e.g. kill PID) or use a shorter timeout.
- Use absolute paths; never assume the current directory.
"""

DONE_SIGNAL = "TASK_COMPLETE"
_BASH_RE = re.compile(r"<bash>(.*?)</bash>", re.DOTALL | re.IGNORECASE)
_CODE_RE = re.compile(r"```(?:bash|sh)?\n(.*?)```", re.DOTALL)


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class Trajectory:
    """Complete record of one RL episode."""

    task: dict
    messages: List[Dict[str, Any]] = field(default_factory=list)
    # Per-turn step scores set by PRMClient (optional)
    turn_scores: List[float] = field(default_factory=list)
    score: float = 0.0
    n_turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_s: float = 0.0
    done: bool = False          # True if model signalled TASK_COMPLETE
    error: Optional[str] = None

    def to_log_dict(self) -> dict:
        return {
            "task_name": self.task.get("task_name", "?"),
            "score": self.score,
            "n_turns": self.n_turns,
            "done": self.done,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "elapsed_s": round(self.elapsed_s, 2),
            "error": self.error,
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_bash_cmd(text: str) -> Optional[str]:
    """Extract the first bash command from <bash>…</bash> or ```bash…``` blocks."""
    m = _BASH_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _CODE_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def _trim_messages(
    messages: List[Dict[str, Any]], budget: int = config.CONTEXT_LEN
) -> List[Dict[str, Any]]:
    """
    Drop the oldest assistant+observation pairs until the total character
    count of all message content is within *budget*.

    Always preserves:
    - messages[0]: system prompt
    - messages[1]: initial user task
    """
    if len(messages) <= 2:
        return messages

    def _total_chars(msgs: List[Dict[str, Any]]) -> int:
        return sum(len(m.get("content") or "") for m in msgs)

    # Find the first mutable index (everything after system + first user)
    mutable_start = 2
    result = messages.copy()

    while _total_chars(result) > budget and len(result) > mutable_start + 1:
        # Drop oldest pair (assistant turn + following observation)
        result.pop(mutable_start)
        if len(result) > mutable_start:
            result.pop(mutable_start)

    return result


# ── Main episode coroutine ─────────────────────────────────────────────────────

async def run_episode(
    task: dict,
    env_pool,                              # LocalEnvPool instance
    policy_url: str = config.POLICY_URL,
    policy_model: str = config.POLICY_MODEL,
    max_turns: int = config.MAX_TURNS,
    context_len: int = config.CONTEXT_LEN,
    temperature: float = 0.7,
) -> Trajectory:
    """
    Run one RL episode: allocate env → multi-turn agent loop → evaluate → close.

    Returns a Trajectory with messages, score, and token stats.
    """
    traj = Trajectory(task=task)
    t0 = time.monotonic()

    client = openai.AsyncOpenAI(base_url=policy_url, api_key="none")

    instruction = task.get("instruction") or task.get("prompt") or str(task)
    messages: List[Dict[str, Any]] = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": instruction},
    ]

    lease_id: Optional[str] = None
    try:
        lease_id = await env_pool.allocate(task)

        for turn in range(max_turns):
            # ── Policy call ──────────────────────────────────────────────────
            trimmed = _trim_messages(messages, budget=context_len)
            try:
                resp = await client.chat.completions.create(
                    model=policy_model,
                    messages=trimmed,
                    max_tokens=1024,
                    temperature=temperature,
                )
            except openai.OpenAIError as exc:
                traj.error = f"Policy API error on turn {turn}: {exc}"
                logger.error("Policy error: %s", exc)
                break

            choice = resp.choices[0]
            text = choice.message.content or ""

            # Track token usage
            if resp.usage:
                traj.prompt_tokens     += resp.usage.prompt_tokens or 0
                traj.completion_tokens += resp.usage.completion_tokens or 0

            messages.append({"role": "assistant", "content": text})

            # ── Termination check ────────────────────────────────────────────
            if DONE_SIGNAL in text:
                traj.done = True
                traj.n_turns = turn + 1
                break

            # ── Parse & execute bash command ─────────────────────────────────
            bash_cmd = _parse_bash_cmd(text)
            if bash_cmd is None:
                obs = (
                    "[AGENT_HINT] No <bash>…</bash> block found.  "
                    "Use <bash>your command</bash> or write TASK_COMPLETE."
                )
                logger.debug("Turn %d: no bash command parsed", turn)
            else:
                obs = await env_pool.exec(lease_id, bash_cmd)
                logger.debug("Turn %d: cmd=%r obs_len=%d", turn, bash_cmd[:60], len(obs))

            messages.append({"role": "user", "content": f"<observation>\n{obs}\n</observation>"})
            traj.n_turns = turn + 1

        # ── Evaluate ─────────────────────────────────────────────────────────
        traj.score = await env_pool.evaluate(lease_id)

    except Exception as exc:
        traj.error = str(exc)
        logger.exception("Episode failed for task=%s", task.get("task_name", "?"))

    finally:
        traj.messages = messages
        traj.elapsed_s = time.monotonic() - t0
        if lease_id is not None:
            try:
                await env_pool.close(lease_id)
            except Exception as exc:
                logger.warning("Error closing lease %s: %s", lease_id, exc)

    logger.info(
        "Episode done: task=%s score=%.3f turns=%d done=%s elapsed=%.1fs",
        task.get("task_name", "?"),
        traj.score,
        traj.n_turns,
        traj.done,
        traj.elapsed_s,
    )
    return traj
