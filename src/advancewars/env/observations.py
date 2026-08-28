"""Observation builders."""

from __future__ import annotations

import numpy as np

from advancewars.engine.commanders import commander_for
from advancewars.engine.data import TERRAIN_BY_KEY, UNITS
from advancewars.engine.coordinates import Coord
from advancewars.engine.state import GameState


TERRAIN_KEYS = sorted(TERRAIN_BY_KEY)
UNIT_KEYS = sorted(UNITS)
WEATHER_CODES = {"clear": 0.0, "rain": 1.0, "snow": 2.0}


def board_planes(
    state: GameState,
    player_id: int,
    visible_coords: set[Coord] | None = None,
    visible_unit_ids: set[int] | None = None,
) -> np.ndarray:
    channels = len(TERRAIN_KEYS) + len(UNIT_KEYS) + 9
    obs = np.zeros((channels, state.map.height, state.map.width), dtype=np.float32)
    terrain_offset = 0
    unit_offset = len(TERRAIN_KEYS)
    owner_offset = unit_offset + len(UNIT_KEYS)
    if visible_coords is None:
        visible_coords = {
            Coord(x, y)
            for y in range(state.map.height)
            for x in range(state.map.width)
        }
    if visible_unit_ids is None:
        visible_unit_ids = {unit.id for unit in state.map_units()}
    player = state.players[player_id]
    commander = commander_for(player.commander)
    power_charge = player.power_charge / max(1, commander.power_cost)
    power_active = 1.0 if player.active_power_turns > 0 else 0.0

    for y, row in enumerate(state.map.tiles):
        for x, tile in enumerate(row):
            coord = Coord(x, y)
            obs[terrain_offset + TERRAIN_KEYS.index(tile.terrain), y, x] = 1.0
            obs[owner_offset, y, x] = -1.0 if tile.owner is None else tile.owner
            obs[owner_offset + 1, y, x] = tile.definition.defense / 4.0
            obs[owner_offset + 4, y, x] = float(player_id)
            obs[owner_offset + 5, y, x] = 1.0 if coord in visible_coords else 0.0
            obs[owner_offset + 6, y, x] = WEATHER_CODES[state.weather]
            obs[owner_offset + 7, y, x] = power_charge
            obs[owner_offset + 8, y, x] = power_active

    for unit in state.map_units():
        if unit.id not in visible_unit_ids:
            continue
        x, y = unit.coord.x, unit.coord.y
        obs[unit_offset + UNIT_KEYS.index(unit.unit_type), y, x] = 1.0
        obs[owner_offset + 2, y, x] = unit.owner
        obs[owner_offset + 3, y, x] = unit.hp / 100.0

    return obs
