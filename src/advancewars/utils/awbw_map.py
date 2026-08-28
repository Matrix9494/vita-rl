"""Convert AWBW map JSON payloads into the local DefendPeace-style map text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TERRAIN_ID_TO_CODE: dict[int, str] = {
    1: "GR",
    2: "MT",
    3: "FR",
    **{terrain_id: "RV" for terrain_id in range(4, 15)},
    **{terrain_id: "RD" for terrain_id in range(15, 26)},
    26: "BR",
    27: "BR",
    28: "SE",
    29: "SH",
    30: "SH",
    31: "SH",
    32: "SH",
    33: "RF",
    **{terrain_id: "PI" for terrain_id in range(101, 111)},
    111: "SR",
    112: "BK",
    113: "PI",
    114: "PI",
    115: "GR",
    116: "GR",
    195: "GR",
}

BUILDING_TYPE_TO_CODE = {
    "City": "CT",
    "Base": "FC",
    "Airport": "AP",
    "Port": "SP",
    "HQ": "HQ",
    "ComTower": "TW",
    "Lab": "LB",
    "Missile": "SR",
    "PipeSeam": "PI",
}

BUILDING_ID_TO_TYPE_AND_COUNTRY: dict[int, tuple[str, int | None]] = {
    34: ("City", None),
    35: ("Base", None),
    36: ("Airport", None),
    37: ("Port", None),
    38: ("City", 1),
    39: ("Base", 1),
    40: ("Airport", 1),
    41: ("Port", 1),
    42: ("HQ", 1),
    43: ("City", 2),
    44: ("Base", 2),
    45: ("Airport", 2),
    46: ("Port", 2),
    47: ("HQ", 2),
    48: ("City", 3),
    49: ("Base", 3),
    50: ("Airport", 3),
    51: ("Port", 3),
    52: ("HQ", 3),
    53: ("City", 4),
    54: ("Base", 4),
    55: ("Airport", 4),
    56: ("Port", 4),
    57: ("HQ", 4),
    81: ("City", 6),
    82: ("Base", 6),
    83: ("Airport", 6),
    84: ("Port", 6),
    85: ("HQ", 6),
    86: ("City", 7),
    87: ("Base", 7),
    88: ("Airport", 7),
    89: ("Port", 7),
    90: ("HQ", 7),
    91: ("City", 5),
    92: ("Base", 5),
    93: ("Airport", 5),
    94: ("Port", 5),
    95: ("HQ", 5),
    96: ("City", 8),
    97: ("Base", 8),
    98: ("Airport", 8),
    99: ("Port", 8),
    100: ("HQ", 8),
    111: ("Missile", None),
    113: ("PipeSeam", None),
    114: ("PipeSeam", None),
    117: ("Airport", 9),
    118: ("Base", 9),
    119: ("City", 9),
    120: ("HQ", 9),
    121: ("Port", 9),
    122: ("Airport", 10),
    123: ("Base", 10),
    124: ("City", 10),
    125: ("HQ", 10),
    126: ("Port", 10),
    127: ("ComTower", 9),
    128: ("ComTower", 5),
    129: ("ComTower", 2),
    130: ("ComTower", 8),
    131: ("ComTower", 3),
    132: ("ComTower", 10),
    133: ("ComTower", None),
    134: ("ComTower", 1),
    135: ("ComTower", 6),
    136: ("ComTower", 4),
    137: ("ComTower", 7),
    138: ("Lab", 9),
    139: ("Lab", 5),
    140: ("Lab", 2),
    141: ("Lab", 8),
    142: ("Lab", 3),
    143: ("Lab", 7),
    144: ("Lab", 10),
    145: ("Lab", None),
    146: ("Lab", 1),
    147: ("Lab", 6),
    148: ("Lab", 4),
    149: ("Airport", 16),
    150: ("Base", 16),
    151: ("City", 16),
    152: ("ComTower", 16),
    153: ("HQ", 16),
    154: ("Lab", 16),
    155: ("Port", 16),
    156: ("Airport", 17),
    157: ("Base", 17),
    158: ("City", 17),
    159: ("ComTower", 17),
    160: ("HQ", 17),
    161: ("Lab", 17),
    162: ("Port", 17),
    163: ("Airport", 19),
    164: ("Base", 19),
    165: ("City", 19),
    166: ("ComTower", 19),
    167: ("HQ", 19),
    168: ("Lab", 19),
    169: ("Port", 19),
    170: ("Airport", 20),
    171: ("Base", 20),
    172: ("City", 20),
    173: ("ComTower", 20),
    174: ("HQ", 20),
    175: ("Lab", 20),
    176: ("Port", 20),
    181: ("Airport", 21),
    182: ("Base", 21),
    183: ("City", 21),
    184: ("ComTower", 21),
    185: ("HQ", 21),
    186: ("Lab", 21),
    187: ("Port", 21),
    188: ("Airport", 22),
    189: ("Base", 22),
    190: ("City", 22),
    191: ("ComTower", 22),
    192: ("HQ", 22),
    193: ("Lab", 22),
    194: ("Port", 22),
    196: ("Airport", 23),
    197: ("Base", 23),
    198: ("City", 23),
    199: ("ComTower", 23),
    200: ("HQ", 23),
    201: ("Lab", 23),
    202: ("Port", 23),
    203: ("Airport", 24),
    204: ("Base", 24),
    205: ("City", 24),
    206: ("ComTower", 24),
    207: ("HQ", 24),
    208: ("Lab", 24),
    209: ("Port", 24),
    210: ("Airport", 25),
    211: ("Base", 25),
    212: ("City", 25),
    213: ("ComTower", 25),
    214: ("HQ", 25),
    215: ("Lab", 25),
    216: ("Port", 25),
    217: ("Airport", 26),
    218: ("Base", 26),
    219: ("City", 26),
    220: ("ComTower", 26),
    221: ("HQ", 26),
    222: ("Lab", 26),
    223: ("Port", 26),
}


def load_awbw_map_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def awbw_json_to_defendpeace_map(
    payload: dict[str, Any],
    *,
    seed_infantry: bool = True,
) -> str:
    """Convert one AWBW map_info payload into DefendPeace-style text."""

    width = int(payload["Size X"])
    height = int(payload["Size Y"])
    terrain_map = payload["Terrain Map"]
    country_to_player = _country_to_player(terrain_map)
    rows: list[str] = []
    player_seed_coords: dict[int, tuple[int, int]] = {}
    player_fallback_coords: dict[int, tuple[int, int]] = {}

    for y in range(height):
        cells: list[str] = []
        for x in range(width):
            awbw_id = int(terrain_map[x][y])
            terrain_code, owner = _convert_awbw_tile(awbw_id, country_to_player)
            if terrain_code == "HQ" and owner is not None:
                player_seed_coords.setdefault(owner, (x, y))
            elif owner is not None and terrain_code == "FC":
                player_fallback_coords.setdefault(owner, (x, y))
            elif owner is not None and owner not in player_fallback_coords:
                player_fallback_coords[owner] = (x, y)
            cells.append(_format_cell(owner, terrain_code))
        rows.append("".join(cells))

    unit_lines: list[str] = []
    if seed_infantry:
        seed_coords = player_fallback_coords | player_seed_coords
        for player_id, coord in sorted(seed_coords.items()):
            unit_lines.append(f"{player_id}, Infantry, {coord[0]}, {coord[1]}")

    if unit_lines:
        return "\n".join(rows) + "\n\n" + "\n".join(unit_lines) + "\n"
    return "\n".join(rows) + "\n"


def convert_awbw_file(
    source: str | Path,
    destination: str | Path,
    *,
    seed_infantry: bool = True,
) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        awbw_json_to_defendpeace_map(
            load_awbw_map_json(source), seed_infantry=seed_infantry
        )
    )
    return destination_path


def _country_to_player(terrain_map: list[list[int]]) -> dict[int, int]:
    country_ids = sorted(
        {
            country
            for column in terrain_map
            for awbw_id in column
            for _building_type, country in [BUILDING_ID_TO_TYPE_AND_COUNTRY.get(awbw_id, ("", None))]
            if country is not None
        }
    )
    return {country: index for index, country in enumerate(country_ids[:2])}


def _convert_awbw_tile(
    awbw_id: int,
    country_to_player: dict[int, int],
) -> tuple[str, int | None]:
    if awbw_id in BUILDING_ID_TO_TYPE_AND_COUNTRY:
        building_type, country = BUILDING_ID_TO_TYPE_AND_COUNTRY[awbw_id]
        terrain_code = BUILDING_TYPE_TO_CODE[building_type]
        return terrain_code, None if country is None else country_to_player.get(country)
    if awbw_id in TERRAIN_ID_TO_CODE:
        return TERRAIN_ID_TO_CODE[awbw_id], None
    return "GR", None


def _format_cell(owner: int | None, terrain_code: str) -> str:
    owner_text = "" if owner is None else str(owner)
    return f"{owner_text:>2}{terrain_code:<2}"
