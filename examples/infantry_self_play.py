"""Run infantry-only decision-tree self-play matches."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from advancewars import raw_env
from advancewars.policies import InfantryDecisionTreePolicy


def run_self_play(
    *,
    rounds: int = 10,
    output_path: str | Path = "runs/infantry_self_play_10.json",
    map_name: str = "duel",
    max_steps: int = 250,
    max_turns: int = 40,
    seed: int = 0,
    reward_mode: str = "dense_basic",
) -> dict[str, Any]:
    policy = InfantryDecisionTreePolicy()
    results: list[dict[str, Any]] = []
    wins = {"player_0": 0, "player_1": 0, "draw_or_truncated": 0}

    for round_index in range(rounds):
        env = raw_env(
            action_mode="structured",
            render_mode="ansi",
            map_name=map_name,
            config="infantry_only",
            max_turns=max_turns,
            reward_mode=reward_mode,
        )
        env.reset(seed=seed + round_index)
        steps = 0
        action_counts: dict[str, int] = {}
        for agent in env.agent_iter(max_iter=max_steps):
            observation, _reward, termination, truncation, info = env.last()
            if termination or truncation:
                break
            action = policy.choose_action(env, agent, observation, info)
            semantic = info["legal_semantic_actions"][
                info["legal_structured_actions"].index(action)
            ]
            action_kind = semantic["kind"]
            action_counts[action_kind] = action_counts.get(action_kind, 0) + 1
            env.step(action)
            steps += 1
            state = env.game.state
            if state is not None and (state.done or state.truncated):
                break

        state = env.game.state
        assert state is not None
        if state.winner == 0:
            wins["player_0"] += 1
        elif state.winner == 1:
            wins["player_1"] += 1
        else:
            wins["draw_or_truncated"] += 1
        results.append(
            {
                "round": round_index,
                "seed": seed + round_index,
                "steps": steps,
                "turn": state.turn,
                "done": state.done,
                "truncated": state.truncated,
                "winner": state.winner,
                "action_counts": action_counts,
                "final_render": env.render(),
                "final_state": env.game.to_json()["state"],
            }
        )

    summary = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rounds": rounds,
            "map_name": map_name,
            "config": "infantry_only",
            "max_steps": max_steps,
            "max_turns": max_turns,
            "seed": seed,
            "reward_mode": reward_mode,
            "policy": "InfantryDecisionTreePolicy",
            "action_mode": "structured",
        },
        "wins": wins,
        "rounds": results,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output", default="runs/infantry_self_play_10.json")
    parser.add_argument("--map", default="duel", dest="map_name")
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-mode", default="dense_basic")
    args = parser.parse_args()

    summary = run_self_play(
        rounds=args.rounds,
        output_path=args.output,
        map_name=args.map_name,
        max_steps=args.max_steps,
        max_turns=args.max_turns,
        seed=args.seed,
        reward_mode=args.reward_mode,
    )
    print(f"saved={args.output}")
    print(f"wins={summary['wins']}")
    for result in summary["rounds"]:
        print(
            f"round={result['round']} steps={result['steps']} "
            f"done={result['done']} truncated={result['truncated']} "
            f"winner={result['winner']}"
        )


if __name__ == "__main__":
    main()
