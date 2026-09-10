"""Raw, logging-only diagnostics for BFCL GRPO training groups.

This hook deliberately delegates to Dressage's normal rollout logger before it
writes its own JSONL sidecar.  It neither mutates samples nor participates in
reward/advantage computation.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _trajectory_id(sample: Any, position: int) -> str:
    metadata = _metadata(sample)
    value = (
        metadata.get("parent_traj_id")
        or metadata.get("session_id")
        or getattr(sample, "session_id", None)
        or getattr(sample, "index", None)
    )
    return str(value) if value is not None else f"sample-{position}"


def _segment(sample: Any, position: int) -> dict[str, Any]:
    metadata = _metadata(sample)
    simulation = metadata.get("environment_simulation")
    simulation = simulation if isinstance(simulation, dict) else {}
    loss_mask = getattr(sample, "loss_mask", None) or []
    response_length = getattr(sample, "response_length", None)
    if response_length is None:
        response_length = len(loss_mask)
    status = str(getattr(sample, "status", ""))
    return {
        "trajectory_id": _trajectory_id(sample, position),
        "group_index": str(getattr(sample, "group_index", metadata.get("group_index", "unknown"))),
        "task_id": metadata.get("environment_task_id", metadata.get("task_id")),
        "segment_index": int(metadata.get("segment_index", 0)),
        "terminal_reward": _as_float(metadata.get("environment_reward", getattr(sample, "reward", 0.0))),
        "response_tokens": int(response_length or 0),
        "trainable_tokens": int(sum(1 for item in loss_mask if item)),
        "truncated": bool(metadata.get("truncated", False) or status.endswith("TRUNCATED")),
        "fully_excluded": bool(getattr(sample, "remove_sample", False) or int(response_length or 0) == 0),
        "termination_reason": metadata.get("environment_termination_reason"),
        "num_agent_turns": metadata.get("environment_num_agent_turns"),
        "num_tool_calls": simulation.get("num_tool_calls"),
        "num_tool_errors": simulation.get("num_tool_errors"),
    }


def _write_groups(rollout_id: int, samples: list[Any]) -> None:
    destination = os.environ.get("BFCL_TRAIN_GROUPS_PATH")
    if not destination:
        return
    trajectories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for position, sample in enumerate(samples):
        segment = _segment(sample, position)
        trajectories[(segment["group_index"], segment["trajectory_id"])].append(segment)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (group_index, trajectory_id), segments in trajectories.items():
        anchor = max(segments, key=lambda item: item["segment_index"])
        grouped[group_index].append(
            {
                "trajectory_id": trajectory_id,
                "task_id": anchor["task_id"],
                "terminal_reward": anchor["terminal_reward"],
                "segment_count": len(segments),
                "response_tokens": sum(item["response_tokens"] for item in segments),
                "trainable_tokens": sum(item["trainable_tokens"] for item in segments),
                "truncated": any(item["truncated"] for item in segments),
                "fully_excluded_segments": sum(item["fully_excluded"] for item in segments),
                "termination_reason": anchor["termination_reason"],
                "num_agent_turns": anchor["num_agent_turns"],
                "num_tool_calls": anchor["num_tool_calls"],
                "num_tool_errors": anchor["num_tool_errors"],
            }
        )

    groups = []
    for group_index, members in sorted(grouped.items()):
        rewards = [member["terminal_reward"] for member in members]
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        groups.append(
            {
                "group_index": group_index,
                "num_trajectories": len(members),
                "mean_terminal_reward": mean_reward,
                "reward_variance": (
                    sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
                    if rewards
                    else 0.0
                ),
                "members": members,
            }
        )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"rollout_id": int(rollout_id), "groups": groups}, sort_keys=True, default=str) + "\n")


def log_rollout_data(
    rollout_id: int, args: Any, samples: list[Any], extra_metrics: dict[str, Any], rollout_time: float
) -> bool:
    """Run the stock logger, then persist exact GRPO group composition."""
    from dressage.rollout.log_rollout import log_rollout_data as stock_log_rollout_data

    stock_log_rollout_data(rollout_id, args, samples, extra_metrics, rollout_time)
    _write_groups(rollout_id, samples)
    return False
