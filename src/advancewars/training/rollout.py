"""Rollout helpers for policy-vs-policy map batches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from advancewars import raw_env
from advancewars.training.analysis import state_metrics
from advancewars.training.policy_config import InfantryPolicyConfig
from advancewars.utils.awbw_map import convert_awbw_file


def run_match(
    *,
    map_entry: dict[str, Any],
    raw_map_dir: str | Path,
    converted_map_dir: str | Path,
    policy_a: InfantryPolicyConfig,
    policy_b: InfantryPolicyConfig,
    seed: int,
    max_steps: int,
    max_turns: int,
    a_player_id: int,
) -> dict[str, Any]:
    map_id = int(map_entry["map_id"])
    raw_path = Path(raw_map_dir) / f"{map_id}.json"
    converted_path = Path(converted_map_dir) / f"{map_id}.map"
    convert_awbw_file(raw_path, converted_path)

    policies = {
        a_player_id: policy_a.build_policy(),
        1 - a_player_id: policy_b.build_policy(),
    }
    labels = {a_player_id: "A", 1 - a_player_id: "B"}

    env = raw_env(
        action_mode="structured",
        render_mode="ansi",
        map_name=str(converted_path),
        config="infantry_only",
        max_turns=max_turns,
        reward_mode="dense_basic",
    )
    env.reset(seed=seed)

    steps = 0
    action_counts = {"A": {}, "B": {}}
    for agent in env.agent_iter(max_iter=max_steps):
        observation, _reward, termination, truncation, info = env.last()
        if termination or truncation:
            break
        player_id = int(agent.split("_", 1)[1])
        action = policies[player_id].choose_action(env, agent, observation, info)
        semantic = info["legal_semantic_actions"][
            info["legal_structured_actions"].index(action)
        ]
        label = labels[player_id]
        kind = semantic["kind"]
        action_counts[label][kind] = action_counts[label].get(kind, 0) + 1
        env.step(action)
        steps += 1
        state = env.game.state
        if state is not None and (state.done or state.truncated):
            break

    state = env.game.state
    assert state is not None
    winner_label = "draw" if state.winner is None else labels.get(state.winner, "draw")
    return {
        "map": map_entry,
        "seed": seed,
        "steps": steps,
        "turn": state.turn,
        "done": state.done,
        "truncated": state.truncated,
        "winner": state.winner,
        "winner_label": winner_label,
        "a_player_id": a_player_id,
        "action_counts": action_counts,
        "final_metrics": {
            "A": state_metrics(state, a_player_id),
            "B": state_metrics(state, 1 - a_player_id),
        },
        "final_render": env.render(),
    }
