"""Alternating scripted-policy self-play iteration loop."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from advancewars.training.analysis import summarize_games
from advancewars.training.improver import write_codex_improvement_request
from advancewars.training.policy_config import InfantryPolicyConfig
from advancewars.training.rollout import run_match


def run_iteration(
    *,
    iteration: int,
    dataset_dir: str | Path = "datasets/awbw_maps",
    output_root: str | Path = "runs/scripted_selfplay",
    game_count: int = 200,
    max_steps: int = 250,
    max_turns: int = 40,
    seed: int = 0,
    policy_a: InfantryPolicyConfig | None = None,
    policy_b: InfantryPolicyConfig | None = None,
) -> dict[str, Any]:
    dataset = Path(dataset_dir)
    output = Path(output_root) / f"iter_{iteration:04d}"
    output.mkdir(parents=True, exist_ok=True)
    games_dir = output / "games"
    games_dir.mkdir(exist_ok=True)

    policy_a = policy_a or _load_previous_policy(output_root, iteration, "A")
    policy_b = policy_b or _load_previous_policy(output_root, iteration, "B")
    policy_a.save(output / "policy_A.yaml")
    policy_b.save(output / "policy_B.yaml")
    policy_a.save(output / "next_policy_A.yaml")
    policy_b.save(output / "next_policy_B.yaml")

    train_maps = json.loads((dataset / "splits" / "train_200.json").read_text())
    selected_maps = train_maps[:game_count]
    games: list[dict[str, Any]] = []
    for index, map_entry in enumerate(selected_maps):
        a_player_id = (iteration + index) % 2
        game = run_match(
            map_entry=map_entry,
            raw_map_dir=dataset / "raw",
            converted_map_dir=dataset / "converted",
            policy_a=policy_a,
            policy_b=policy_b,
            seed=seed + iteration * 10000 + index,
            max_steps=max_steps,
            max_turns=max_turns,
            a_player_id=a_player_id,
        )
        games.append(game)
        (games_dir / f"{index:03d}_{map_entry['map_id']}.json").write_text(
            json.dumps(game, indent=2, sort_keys=True)
        )
        print(
            f"[{index + 1:03}/{len(selected_maps)}] map={map_entry['map_id']} "
            f"winner={game['winner_label']} steps={game['steps']}",
            flush=True,
        )

    diagnostics = summarize_games(games)
    learner = "A" if iteration % 2 == 0 else "B"
    (output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True)
    )
    request_path = write_codex_improvement_request(
        iteration_dir=output,
        iteration=iteration,
        learner_label=learner,
        diagnostics=diagnostics,
        games=games,
    )

    summary = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "iteration": iteration,
            "game_count": len(games),
            "max_steps": max_steps,
            "max_turns": max_turns,
            "seed": seed,
            "learner": learner,
            "codex_request": str(request_path),
        },
        "diagnostics": diagnostics,
        "next_policy_files": {
            "A": str(output / "next_policy_A.yaml"),
            "B": str(output / "next_policy_B.yaml"),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _load_previous_policy(
    output_root: str | Path,
    iteration: int,
    label: str,
) -> InfantryPolicyConfig:
    if iteration <= 0:
        return InfantryPolicyConfig()
    previous_policy = Path(output_root) / f"iter_{iteration - 1:04d}" / f"next_policy_{label}.yaml"
    if not previous_policy.exists():
        return InfantryPolicyConfig()
    return InfantryPolicyConfig.load(previous_policy)
