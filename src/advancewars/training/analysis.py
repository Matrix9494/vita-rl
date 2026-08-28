"""Aggregate diagnostics from scripted self-play rollouts."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from advancewars.engine.state import GameState


def state_metrics(state: GameState, player_id: int) -> dict[str, float | int]:
    units = state.living_units(player_id)
    owned_tiles = [
        tile for row in state.map.tiles for tile in row if tile.owner == player_id
    ]
    profitable = [tile for tile in owned_tiles if tile.definition.profitable]
    return {
        "funds": state.players[player_id].funds,
        "unit_count": len(units),
        "unit_value": sum(unit.definition.cost * unit.hp / 100 for unit in units),
        "property_count": len(owned_tiles),
        "income": len(profitable) * 1000,
    }


def summarize_games(games: list[dict[str, Any]]) -> dict[str, Any]:
    wins = Counter(game["winner_label"] for game in games)
    action_counts: dict[str, Counter] = defaultdict(Counter)
    totals: dict[str, Counter] = defaultdict(Counter)
    metric_diffs: dict[str, list[float]] = defaultdict(list)

    for game in games:
        for label in ("A", "B"):
            action_counts[label].update(game["action_counts"].get(label, {}))
        final_metrics = game["final_metrics"]
        for metric in ("unit_value", "income", "property_count", "unit_count"):
            metric_diffs[metric].append(final_metrics["A"][metric] - final_metrics["B"][metric])
        for label in ("A", "B"):
            totals[label]["steps"] += game["steps"]
            totals[label]["wins"] += int(game["winner_label"] == label)
            totals[label]["draws"] += int(game["winner_label"] == "draw")

    return {
        "games": len(games),
        "wins": dict(wins),
        "action_counts": {
            label: dict(counter) for label, counter in sorted(action_counts.items())
        },
        "averages": {
            metric: sum(values) / len(values) if values else 0
            for metric, values in sorted(metric_diffs.items())
        },
        "totals": {label: dict(counter) for label, counter in sorted(totals.items())},
        "failure_modes": _failure_modes(games),
    }


def _failure_modes(games: list[dict[str, Any]]) -> dict[str, Any]:
    modes: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    for label in ("A", "B"):
        losses = [game for game in games if game["winner_label"] not in {label, "draw"}]
        for game in losses[:20]:
            mine = game["final_metrics"][label]
            other = game["final_metrics"]["B" if label == "A" else "A"]
            modes[label].append(
                {
                    "map_id": game["map"]["map_id"],
                    "map_name": game["map"]["name"],
                    "unit_value_diff": mine["unit_value"] - other["unit_value"],
                    "income_diff": mine["income"] - other["income"],
                    "property_diff": mine["property_count"] - other["property_count"],
                }
            )
    return modes
