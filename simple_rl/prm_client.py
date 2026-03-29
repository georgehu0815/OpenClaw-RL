"""
prm_client.py — Optional Process Reward Model client.

Connects to the oMLX PRM Slot (a second model served on a different port)
to score each (command, observation) step of a trajectory.

Enable with:  PRM_ENABLE=1 python train_async.py …

Majority-vote scoring (PRM_M=3 by default) reduces noise from a single
model call by averaging over multiple samples.
"""
from __future__ import annotations

import logging
from typing import List

import openai

from . import config
from .agent_loop import Trajectory

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


class PRMClient:
    """
    Scores individual (command, observation) steps via the PRM model slot.

    Args:
        prm_url:   Base URL of the oMLX PRM Slot  (default: config.PRM_URL)
        prm_model: Model name to request           (default: config.PRM_MODEL)
        m:         Number of votes per step        (default: config.PRM_M)
    """

    def __init__(
        self,
        prm_url: str = config.PRM_URL,
        prm_model: str = config.PRM_MODEL,
        prm_api_key: str = config.PRM_API_KEY,
        m: int = config.PRM_M,
    ) -> None:
        self._model = prm_model
        self._m = m
        self._client = openai.AsyncOpenAI(base_url=prm_url, api_key=prm_api_key)

    # ── Public API ─────────────────────────────────────────────────────────────

    async def score_step(
        self,
        task_desc: str,
        command: str,
        output: str,
    ) -> float:
        """
        Score one (command, output) step via majority vote over *m* samples.

        Returns a float in [0.0, 1.0]; defaults to 0.5 on failure.
        """
        prompt = _SCORE_PROMPT.format(
            task=task_desc[:400],
            command=command[:300],
            output=output[:600],
        )
        scores: List[float] = []
        for _ in range(self._m):
            try:
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8,
                    temperature=0.3,
                )
                text = (resp.choices[0].message.content or "").strip()
                # Parse first token that looks like a float
                for tok in text.split():
                    try:
                        scores.append(max(0.0, min(1.0, float(tok))))
                        break
                    except ValueError:
                        continue
            except Exception as exc:
                logger.warning("PRM sample failed: %s", exc)

        if not scores:
            return 0.5
        return sum(scores) / len(scores)

    async def score_trajectory(self, traj: Trajectory) -> List[float]:
        """
        Score every assistant→observation pair in a trajectory.

        Returns a list of floats (one per turn), in message order.
        """
        task_desc = traj.task.get("instruction") or traj.task.get("prompt", "")
        step_scores: List[float] = []
        msgs = traj.messages

        for i, msg in enumerate(msgs):
            if msg.get("role") != "assistant":
                continue
            command = msg.get("content") or ""
            # Next message should be the observation
            if i + 1 < len(msgs):
                obs = msgs[i + 1].get("content") or ""
            else:
                obs = ""
            score = await self.score_step(task_desc, command, obs)
            step_scores.append(score)

        return step_scores
