"""
prm_client.py — Optional Process Reward Model client.

Connects to the oMLX PRM Slot (a second model served on a different port)
to score each (command, observation) step of a trajectory.

Enable with:  PRM_ENABLE=1 python train_async.py …

Majority-vote scoring (PRM_M=3 by default) reduces noise from a single
model call by averaging over multiple samples.

PRM JSONL output (written when log_dir is set)
───────────────────────────────────────────────
File: {log_dir}/prm_steps.jsonl
Each line (one per agent turn):
  {
    "session":            str,   # unique ID for this training run
    "task":               str,   # task_name
    "turn":               int,   # 0-based turn index within the trajectory
    "score":              float, # majority-vote average (0.0–1.0)
    "votes":              [float, ...],  # individual sample scores
    "representative_eval": str   # the eval prompt sent to the PRM
  }
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import openai

from . import config
from .agent_loop import Trajectory

# Serialize all PRM requests globally — the omlx server crashes when it tries
# to batch concurrent requests with different sequence lengths (broadcast_shapes
# error).  One in-flight request at a time keeps it happy.
_PRM_SEM: asyncio.Semaphore | None = None


def _get_sem() -> asyncio.Semaphore:
    global _PRM_SEM
    if _PRM_SEM is None:
        _PRM_SEM = asyncio.Semaphore(1)
    return _PRM_SEM

logger = logging.getLogger(__name__)

_SCORE_PROMPT = """\
You are a step evaluator for a terminal agent.
Given the task description, a bash command the agent ran, and the resulting
output, rate how well that step progresses toward solving the task.

Task: {task}

Bash command:
{command}

Output:
{output}

Reply with a single float between 0.0 (useless/wrong) and 1.0 (perfect step).
No explanation, just the number."""


@dataclass
class PRMStepResult:
    """Structured result for one (command, observation) step."""
    score: float               # majority-vote average
    votes: List[float]         # individual sample scores
    representative_eval: str   # the prompt text sent to the PRM


class PRMClient:
    """
    Scores individual (command, observation) steps via the PRM model slot.

    Args:
        prm_url:   Base URL of the oMLX PRM Slot  (default: config.PRM_URL)
        prm_model: Model name to request           (default: config.PRM_MODEL)
        m:         Number of votes per step        (default: config.PRM_M)
        log_dir:   Directory to write prm_steps.jsonl (None = no file output)
        session:   Unique training-run ID (auto-generated if not provided)
    """

    def __init__(
        self,
        prm_url: str = config.PRM_URL,
        prm_model: str = config.PRM_MODEL,
        prm_api_key: str = config.PRM_API_KEY,
        m: int = config.PRM_M,
        log_dir: Optional[str] = None,
        session: Optional[str] = None,
    ) -> None:
        self._model = prm_model
        self._m = m
        self._client = openai.AsyncOpenAI(base_url=prm_url, api_key=prm_api_key)
        self._session = session or uuid.uuid4().hex[:12]

        self._log_path: Optional[Path] = None
        if log_dir:
            self._log_path = Path(log_dir) / "prm_steps.jsonl"
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("PRM step log: %s (session=%s)", self._log_path, self._session)
        else:
            logger.info("PRM step logging disabled (no log_dir set).")
            print("\n\n==PRM step logging disabled. To enable, set log_dir in config.py and pass log_dir to PRMClient\n\n - prm_client.py:112")

    # ── Public API ─────────────────────────────────────────────────────────────

    async def score_step(
        self,
        task_desc: str,
        command: str,
        output: str,
    ) -> PRMStepResult:
        """
        Score one (command, output) step via majority vote over *m* samples.

        Returns a PRMStepResult with score, individual votes, and the eval
        prompt.  score defaults to 0.5 on total failure.
        """
        prompt = _SCORE_PROMPT.format(
            task=task_desc[:400],
            command=command[:300],
            output=output[:600],
        )
        votes: List[float] = []
        # Single request with n=m avoids flooding the server with M separate
        # calls per step.  The semaphore ensures only one PRM request is
        # in-flight at a time, preventing the KV-cache broadcast_shapes crash
        # that occurs when the omlx server batches requests with different
        # sequence lengths.
        async with _get_sem():
            try:
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8,
                    temperature=0.3,
                    n=self._m,
                )
                for choice in resp.choices:
                    text = (choice.message.content or "").strip()
                    for tok in text.split():
                        try:
                            votes.append(max(0.0, min(1.0, float(tok))))
                            break
                        except ValueError:
                            continue
            except Exception as exc:
                logger.warning("PRM sample failed: %s", exc)

        score = sum(votes) / len(votes) if votes else 0.5
        return PRMStepResult(score=score, votes=votes, representative_eval=prompt)

    async def score_trajectory(self, traj: Trajectory) -> List[float]:
        """
        Score every assistant→observation pair in a trajectory.

        Populates traj.turn_scores (List[float]) and, when log_dir was set,
        appends one JSON line per turn to prm_steps.jsonl.

        Returns a list of floats (one per turn) for backward compatibility.
        """
        task_name = traj.task.get("task_name", "unknown")
        task_desc = traj.task.get("instruction") or traj.task.get("prompt", "")
        step_results: List[PRMStepResult] = []
        msgs = traj.messages

        turn_idx = 0
        for i, msg in enumerate(msgs):
            if msg.get("role") != "assistant":
                continue
            command = msg.get("content") or ""
            obs = msgs[i + 1].get("content") if i + 1 < len(msgs) else ""
            result = await self.score_step(task_desc, command, obs or "")
            step_results.append(result)

            if self._log_path is not None:
                record = {
                    "session":            self._session,
                    "task":               task_name,
                    "turn":               turn_idx,
                    "score":              result.score,
                    "votes":              result.votes,
                    "representative_eval": result.representative_eval,
                }
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

            turn_idx += 1

        return [r.score for r in step_results]
