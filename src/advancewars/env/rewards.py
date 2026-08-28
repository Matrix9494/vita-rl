"""Reward helpers."""

from __future__ import annotations

from advancewars.engine.coordinates import Coord
from advancewars.engine.state import Event, GameState


CAPTURE_REWARD = 0.2
DAMAGE_REWARD_SCALE = 1000.0
UNIT_VALUE_REWARD_SCALE = 100000.0
PROPERTY_INCOME_REWARD = 0.1
HQ_PRESSURE_REWARD_SCALE = 100.0


def rewards_for(
    state: GameState,
    mode: str,
    events: list[Event] | None = None,
) -> dict[str, float]:
    agents = [f"player_{player_id}" for player_id in sorted(state.players)]
    if mode == "none":
        return {agent: 0.0 for agent in agents}
    if mode not in {"win_loss", "dense_basic"}:
        raise ValueError(f"Unknown reward mode: {mode}")
    rewards = {agent: 0.0 for agent in agents}
    if state.done and state.winner is not None:
        for player_id in state.players:
            rewards[f"player_{player_id}"] = 1.0 if player_id == state.winner else -1.0
    if mode == "dense_basic":
        for event in events or []:
            if event.type in {"attack", "counterattack"}:
                attacker = f"player_{event.payload['attacker_owner']}"
                defender = f"player_{event.payload['defender_owner']}"
                damage_reward = event.payload["damage"] / DAMAGE_REWARD_SCALE
                rewards[attacker] += damage_reward
                rewards[defender] -= damage_reward
                if event.payload.get("defender_hp_after") == 0:
                    value_reward = (
                        event.payload.get("defender_unit_cost", 0)
                        / UNIT_VALUE_REWARD_SCALE
                    )
                    rewards[attacker] += value_reward
                    rewards[defender] -= value_reward
            elif event.type == "capture_progress":
                owner = f"player_{event.payload['owner']}"
                coord = event.payload["coord"]
                tile = state.map.tile_at(Coord(coord[0], coord[1]))
                if tile.terrain == "hq":
                    rewards[owner] += (
                        event.payload["progress"] / HQ_PRESSURE_REWARD_SCALE
                    )
            elif event.type == "capture":
                new_owner = f"player_{event.payload['new_owner']}"
                rewards[new_owner] += CAPTURE_REWARD
                old_owner = event.payload["old_owner"]
                if old_owner is not None:
                    rewards[f"player_{old_owner}"] -= CAPTURE_REWARD
                coord = event.payload["coord"]
                tile = state.map.tile_at(Coord(coord[0], coord[1]))
                if tile.definition.profitable:
                    rewards[new_owner] += PROPERTY_INCOME_REWARD
                    if old_owner is not None:
                        rewards[f"player_{old_owner}"] -= PROPERTY_INCOME_REWARD
    return rewards
