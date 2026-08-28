"""Serializable engine state containers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from advancewars.engine.coordinates import Coord
from advancewars.engine.data import TERRAIN_BY_KEY, UNITS, TerrainDef, UnitDef


@dataclass
class TileState:
    terrain: str
    owner: int | None = None

    @property
    def definition(self) -> TerrainDef:
        return TERRAIN_BY_KEY[self.terrain]


@dataclass
class MapState:
    width: int
    height: int
    tiles: list[list[TileState]]

    def in_bounds(self, coord: Coord) -> bool:
        return 0 <= coord.x < self.width and 0 <= coord.y < self.height

    def tile_at(self, coord: Coord) -> TileState:
        if not self.in_bounds(coord):
            raise ValueError(f"Coordinate out of bounds: {coord}")
        return self.tiles[coord.y][coord.x]


@dataclass
class PlayerState:
    id: int
    funds: int = 0
    defeated: bool = False
    commander: str = "andy"
    commanders: list[str] = field(default_factory=lambda: ["andy"])
    active_commander_index: int = 0
    swapped_this_turn: bool = False
    power_charge: int = 0
    active_power_turns: int = 0
    active_power_kind: str = "power"

    def __post_init__(self) -> None:
        if not self.commanders:
            self.commanders = [self.commander]
        if self.commanders == ["andy"] and self.commander != "andy":
            self.commanders = [self.commander]
        if self.active_commander_index < 0:
            raise ValueError("active_commander_index cannot be negative.")
        self.active_commander_index %= len(self.commanders)
        self.commander = self.commanders[self.active_commander_index]

    def set_active_commander_index(self, index: int) -> None:
        if not 0 <= index < len(self.commanders):
            raise ValueError(f"Invalid active commander index: {index}")
        self.active_commander_index = index
        self.commander = self.commanders[index]


@dataclass
class UnitState:
    id: int
    owner: int
    unit_type: str
    coord: Coord
    hp: int = 100
    fuel: int | None = None
    ammo: dict[str, int] = field(default_factory=dict)
    can_act: bool = True
    capture_progress: int = 0
    cargo: list[int] = field(default_factory=list)
    carried_by: int | None = None
    stunned_turns: int = 0

    @property
    def definition(self) -> UnitDef:
        return UNITS[self.unit_type]

    def ready_copy(self) -> UnitState:
        return replace(self, can_act=True, capture_progress=self.capture_progress)


@dataclass
class Event:
    type: str
    payload: dict


@dataclass
class GameState:
    map: MapState
    players: dict[int, PlayerState]
    units: dict[int, UnitState]
    current_player: int = 0
    turn: int = 1
    weather: str = "clear"
    weather_base: str = "clear"
    weather_turns_remaining: int = 0
    done: bool = False
    truncated: bool = False
    winner: int | None = None
    fog_visible_coords: dict[int, set[Coord]] = field(default_factory=dict)
    fog_visible_unit_ids: dict[int, set[int]] = field(default_factory=dict)

    def unit_at(self, coord: Coord) -> UnitState | None:
        for unit in self.units.values():
            if unit.coord == coord and unit.hp > 0 and unit.carried_by is None:
                return unit
        return None

    def living_units(self, owner: int | None = None) -> list[UnitState]:
        units = [unit for unit in self.units.values() if unit.hp > 0]
        if owner is None:
            return units
        return [unit for unit in units if unit.owner == owner]

    def map_units(self, owner: int | None = None) -> list[UnitState]:
        return [unit for unit in self.living_units(owner) if unit.carried_by is None]
