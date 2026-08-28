"""Run one heuristic-vs-heuristic duel through the PettingZoo API."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from advancewars import raw_env
from advancewars.engine.actions import Action
from advancewars.engine.coordinates import Coord
from advancewars.engine.state import Event
from advancewars.policies import HeuristicPolicy


def run_rollout(
    *,
    output_path: str | Path = "runs/heuristic_duel_rollout.json",
    map_name: str = "duel",
    config: str | None = None,
    max_steps: int = 100,
    max_turns: int = 100,
    seed: int = 0,
    reward_mode: str = "dense_basic",
) -> dict[str, Any]:
    """Run a capped duel rollout and save a JSON trajectory."""
    environment = raw_env(
        render_mode="ansi",
        map_name=map_name,
        config=config,
        max_turns=max_turns,
        reward_mode=reward_mode,
    )
    policy = HeuristicPolicy()
    environment.reset(seed=seed)

    trajectory: dict[str, Any] = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "map_name": map_name,
            "config": config,
            "max_steps": max_steps,
            "max_turns": max_turns,
            "seed": seed,
            "reward_mode": reward_mode,
            "policy": "HeuristicPolicy",
            "api": "PettingZoo AEC raw_env",
        },
        "initial": {
            "game": environment.game.to_json(),
            "render": environment.render(),
        },
        "steps": [],
    }

    steps_taken = 0
    for step, agent in enumerate(environment.agent_iter(max_iter=max_steps)):
        observation, reward, termination, truncation, info = environment.last()
        record: dict[str, Any] = {
            "step": step,
            "agent": agent,
            "reward_before": reward,
            "termination_before": termination,
            "truncation_before": truncation,
            "observation": _jsonable(observation),
            "info": _jsonable_info(info),
            "state_before": environment.game.to_json(),
            "render_before": environment.render(),
        }
        if termination or truncation:
            record["selected_action_index"] = None
            record["selected_action"] = None
            trajectory["steps"].append(record)
            break

        action_index = policy.choose_action(environment, agent, observation, info)
        record["selected_action_index"] = action_index
        record["selected_action"] = _jsonable(_selected_action(action_index, info))

        environment.step(action_index)
        steps_taken += 1
        record["events"] = _jsonable(environment.game.last_events)
        record["rewards_after"] = _jsonable(environment.rewards)
        record["terminations_after"] = _jsonable(environment.terminations)
        record["truncations_after"] = _jsonable(environment.truncations)
        record["state_after"] = environment.game.to_json()
        record["render_after"] = environment.render()
        trajectory["steps"].append(record)

        state = environment.game.state
        if state is not None and (state.done or state.truncated):
            break

    state = environment.game.state
    rollout_hit_step_limit = bool(
        state is not None
        and not state.done
        and not state.truncated
        and steps_taken >= max_steps
    )
    trajectory["final"] = {
        "steps_taken": steps_taken,
        "rollout_hit_step_limit": rollout_hit_step_limit,
        "game": environment.game.to_json(),
        "render": environment.render(),
        "rewards": _jsonable(environment.rewards),
        "terminations": _jsonable(environment.terminations),
        "truncations": _jsonable(environment.truncations),
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trajectory, indent=2, sort_keys=True))
    return trajectory


def _selected_action(action_index: int, info: dict[str, Any]) -> Action | None:
    legal_indices = [int(index) for index in info.get("legal_action_indices", [])]
    legal_actions = info.get("legal_actions", [])
    if action_index not in legal_indices:
        return None
    return legal_actions[legal_indices.index(action_index)]


def _jsonable_info(info: dict[str, Any]) -> dict[str, Any]:
    payload = dict(info)
    if "legal_actions" in payload:
        payload["legal_actions"] = [
            _jsonable(action) for action in payload["legal_actions"]
        ]
    return _jsonable(payload)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Action):
        return {
            "type": value.type.value,
            "unit_id": value.unit_id,
            "path": [_jsonable(coord) for coord in value.path],
            "target": _jsonable(value.target),
            "build_unit": value.build_unit,
            "metadata": _jsonable(value.metadata),
        }
    if isinstance(value, Event):
        return {"type": value.type, "payload": _jsonable(value.payload)}
    if isinstance(value, Coord):
        return {"x": value.x, "y": value.y}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/heuristic_duel_rollout.json")
    parser.add_argument("--map", default="duel", dest="map_name")
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-mode", default="dense_basic")
    args = parser.parse_args()

    trajectory = run_rollout(
        output_path=args.output,
        map_name=args.map_name,
        config=args.config,
        max_steps=args.max_steps,
        max_turns=args.max_turns,
        seed=args.seed,
        reward_mode=args.reward_mode,
    )
    final = trajectory["final"]
    game = final["game"]
    state = game["state"]
    print(f"saved={args.output}")
    print(f"steps_taken={final['steps_taken']}")
    print(
        f"done={state['done']} truncated={state['truncated']} "
        f"winner={state['winner']}"
    )
    print(f"rollout_hit_step_limit={final['rollout_hit_step_limit']}")


if __name__ == "__main__":
    main()
