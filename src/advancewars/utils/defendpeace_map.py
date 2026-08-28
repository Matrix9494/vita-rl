"""Parser for DefendPeace fixed-width `.map` files."""

from __future__ import annotations

from dataclasses import dataclass

from advancewars.engine.coordinates import Coord
from advancewars.engine.data import TERRAIN_BY_CODE, UNIT_ALIASES
from advancewars.engine.state import MapState, TileState, UnitState


@dataclass(frozen=True)
class ParsedMap:
    map_state: MapState
    units: dict[int, UnitState]
    player_ids: tuple[int, ...]


def _normalize_unit_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def parse_defendpeace_map(text: str) -> ParsedMap:
    lines = [line.rstrip("\n") for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    map_lines: list[str] = []
    unit_lines: list[str] = []
    in_units = False
    width: int | None = None

    for line in lines:
        if not line.strip():
            in_units = True
            continue
        if in_units or "," in line:
            in_units = True
            unit_lines.append(line)
            continue
        if width is None:
            width = len(line) // 4
        if width and len(line) >= width * 4:
            map_lines.append(line)
        else:
            in_units = True
            unit_lines.append(line)

    if not map_lines:
        raise ValueError("No map rows found.")

    width = len(map_lines[0]) // 4
    height = len(map_lines)
    tiles: list[list[TileState]] = []
    player_ids: set[int] = set()

    for y, line in enumerate(map_lines):
        if len(line) < width * 4:
            raise ValueError(f"Map row {y} is shorter than expected.")
        row: list[TileState] = []
        for x in range(width):
            cell = line[x * 4 : x * 4 + 4]
            owner_text = cell[:2].strip()
            terrain_code = cell[2:4].strip()
            owner = int(owner_text) if owner_text else None
            if owner is not None:
                player_ids.add(owner)
            if terrain_code not in TERRAIN_BY_CODE:
                raise ValueError(f"Unknown terrain code {terrain_code!r} at {(x, y)}")
            row.append(TileState(TERRAIN_BY_CODE[terrain_code].key, owner))
        tiles.append(row)

    units: dict[int, UnitState] = {}
    next_unit_id = 1
    for line in unit_lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            owner = int(parts[0])
        except ValueError:
            continue
        unit_type = UNIT_ALIASES[_normalize_unit_name(parts[1])]
        coord = Coord(int(parts[2]), int(parts[3]))
        player_ids.add(owner)
        unit_def = UNIT_ALIASES[unit_type]
        units[next_unit_id] = UnitState(
            id=next_unit_id,
            owner=owner,
            unit_type=unit_def,
            coord=coord,
        )
        next_unit_id += 1

    return ParsedMap(MapState(width, height, tiles), units, tuple(sorted(player_ids)))
