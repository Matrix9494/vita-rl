"""JSON-friendly serialization for engine state."""

from __future__ import annotations

from typing import Any

from advancewars.engine.coordinates import Coord
from advancewars.engine.state import (
    GameState,
    MapState,
    PlayerState,
    TileState,
    UnitState,
)


def coord_to_dict(coord: Coord) -> dict[str, int]:
    return {"x": coord.x, "y": coord.y}


def coord_from_dict(payload: dict[str, Any]) -> Coord:
    return Coord(int(payload["x"]), int(payload["y"]))


def game_state_to_dict(state: GameState) -> dict[str, Any]:
    return {
        "map": {
            "width": state.map.width,
            "height": state.map.height,
            "tiles": [
                [
                    {"terrain": tile.terrain, "owner": tile.owner}
                    for tile in row
                ]
                for row in state.map.tiles
            ],
        },
        "players": {
            str(player_id): {
                "id": player.id,
                "funds": player.funds,
                "defeated": player.defeated,
                "commander": player.commander,
                "commanders": list(player.commanders),
                "active_commander_index": player.active_commander_index,
                "swapped_this_turn": player.swapped_this_turn,
                "power_charge": player.power_charge,
                "active_power_turns": player.active_power_turns,
                "active_power_kind": player.active_power_kind,
            }
            for player_id, player in state.players.items()
        },
        "units": {
            str(unit_id): {
                "id": unit.id,
                "owner": unit.owner,
                "unit_type": unit.unit_type,
                "coord": coord_to_dict(unit.coord),
                "hp": unit.hp,
                "fuel": unit.fuel,
                "ammo": dict(unit.ammo),
                "can_act": unit.can_act,
                "capture_progress": unit.capture_progress,
                "cargo": list(unit.cargo),
                "carried_by": unit.carried_by,
                "stunned_turns": unit.stunned_turns,
            }
            for unit_id, unit in state.units.items()
        },
        "current_player": state.current_player,
        "turn": state.turn,
        "weather": state.weather,
        "weather_base": state.weather_base,
        "weather_turns_remaining": state.weather_turns_remaining,
        "done": state.done,
        "truncated": state.truncated,
        "winner": state.winner,
        "fog_visible_coords": {
            str(player_id): [coord_to_dict(coord) for coord in sorted(coords)]
            for player_id, coords in state.fog_visible_coords.items()
        },
        "fog_visible_unit_ids": {
            str(player_id): sorted(unit_ids)
            for player_id, unit_ids in state.fog_visible_unit_ids.items()
        },
    }


def player_state_from_dict(payload: dict[str, Any]) -> PlayerState:
    commander = payload.get("commander", "andy")
    commanders = list(payload.get("commanders", [commander]))
    active_index = payload.get("active_commander_index")
    if active_index is None:
        active_index = commanders.index(commander) if commander in commanders else 0
    return PlayerState(
        id=int(payload["id"]),
        funds=int(payload["funds"]),
        defeated=bool(payload["defeated"]),
        commander=commander,
        commanders=commanders,
        active_commander_index=int(active_index),
        swapped_this_turn=bool(payload.get("swapped_this_turn", False)),
        power_charge=int(payload.get("power_charge", 0)),
        active_power_turns=int(payload.get("active_power_turns", 0)),
        active_power_kind=str(payload.get("active_power_kind", "power")),
    )


def game_state_from_dict(payload: dict[str, Any]) -> GameState:
    map_payload = payload["map"]
    tiles = [
        [
            TileState(terrain=tile["terrain"], owner=tile["owner"])
            for tile in row
        ]
        for row in map_payload["tiles"]
    ]
    players = {
        int(player_id): player_state_from_dict(player)
        for player_id, player in payload["players"].items()
    }
    units = {
        int(unit_id): UnitState(
            id=int(unit["id"]),
            owner=int(unit["owner"]),
            unit_type=unit["unit_type"],
            coord=coord_from_dict(unit["coord"]),
            hp=int(unit["hp"]),
            fuel=None if unit["fuel"] is None else int(unit["fuel"]),
            ammo=dict(unit["ammo"]),
            can_act=bool(unit["can_act"]),
            capture_progress=int(unit["capture_progress"]),
            cargo=[int(cargo_id) for cargo_id in unit["cargo"]],
            carried_by=(
                None if unit["carried_by"] is None else int(unit["carried_by"])
            ),
            stunned_turns=int(unit.get("stunned_turns", 0)),
        )
        for unit_id, unit in payload["units"].items()
    }
    return GameState(
        map=MapState(
            width=int(map_payload["width"]),
            height=int(map_payload["height"]),
            tiles=tiles,
        ),
        players=players,
        units=units,
        current_player=int(payload["current_player"]),
        turn=int(payload["turn"]),
        weather=payload.get("weather", "clear"),
        weather_base=payload.get("weather_base", payload.get("weather", "clear")),
        weather_turns_remaining=int(payload.get("weather_turns_remaining", 0)),
        done=bool(payload["done"]),
        truncated=bool(payload.get("truncated", False)),
        winner=None if payload["winner"] is None else int(payload["winner"]),
        fog_visible_coords={
            int(player_id): {coord_from_dict(coord) for coord in coords}
            for player_id, coords in payload.get("fog_visible_coords", {}).items()
        },
        fog_visible_unit_ids={
            int(player_id): {int(unit_id) for unit_id in unit_ids}
            for player_id, unit_ids in payload.get("fog_visible_unit_ids", {}).items()
        },
    )
