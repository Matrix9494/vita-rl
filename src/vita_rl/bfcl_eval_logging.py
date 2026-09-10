"""Persistent per-task logging for BFCL held-out GRPO evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _reward(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("reward", "rewards", "score"):
            if key in value:
                return _reward(value[key])
    return None


def log_eval_rollout_data(
    rollout_id: int, args: Any, data: list[Any], extra_metrics: dict[str, Any]
) -> bool:
    """Persist one record per held-out BFCL evaluation pass."""
    del args
    destination = os.environ.get("BFCL_EVAL_METRICS_PATH")
    if not destination:
        return False

    datasets = data if isinstance(data, dict) else {}
    heldout = datasets.get("bfcl-heldout", {})
    raw_rewards = heldout.get("rewards", []) if isinstance(heldout, dict) else []
    rewards = [_reward(item) for item in raw_rewards]
    samples = heldout.get("samples", []) if isinstance(heldout, dict) else []
    task_outcomes = []
    for index, reward in enumerate(rewards):
        sample = samples[index] if index < len(samples) else None
        metadata = getattr(sample, "metadata", {}) if sample is not None else {}
        task_outcomes.append(
            {
                "task_id": metadata.get("task_id") if isinstance(metadata, dict) else None,
                "terminal_reward": reward,
                "termination_reason": metadata.get("environment_termination_reason") if isinstance(metadata, dict) else None,
                "num_agent_turns": metadata.get("environment_num_agent_turns") if isinstance(metadata, dict) else None,
                "simulation": metadata.get("environment_simulation") if isinstance(metadata, dict) else None,
                "final_response": getattr(sample, "response", None) if sample is not None else None,
            }
        )
    numeric_rewards = [reward for reward in rewards if reward is not None]
    record = {
        "rollout_id": int(rollout_id),
        "num_tasks": len(rewards),
        "mean_terminal_reward": (
            sum(numeric_rewards) / len(numeric_rewards) if numeric_rewards else 0.0
        ),
        "task_outcomes": task_outcomes,
        "extra_metrics": extra_metrics,
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return False
