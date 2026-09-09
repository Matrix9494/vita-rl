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

    rewards = [_reward(item) for item in data]
    numeric_rewards = [reward for reward in rewards if reward is not None]
    record = {
        "rollout_id": int(rollout_id),
        "num_tasks": len(data),
        "mean_terminal_reward": (
            sum(numeric_rewards) / len(numeric_rewards) if numeric_rewards else 0.0
        ),
        "terminal_rewards": rewards,
        "extra_metrics": extra_metrics,
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return False
