from advancewars.engine.coordinates import Coord
from advancewars.engine.data import BUILTIN_MAPS
from advancewars.utils.defendpeace_map import parse_defendpeace_map


def test_parse_builtin_duel_map():
    parsed = parse_defendpeace_map(BUILTIN_MAPS["duel"])

    assert parsed.map_state.width == 7
    assert parsed.map_state.height == 5
    assert parsed.map_state.tile_at(Coord(0, 0)).terrain == "hq"
    assert parsed.map_state.tile_at(Coord(0, 0)).owner == 0
    assert parsed.map_state.tile_at(Coord(6, 0)).owner == 1
    assert len(parsed.units) == 2


def test_parse_missile_silo_codes():
    parsed = parse_defendpeace_map("  SR  BK\n")

    assert parsed.map_state.tile_at(Coord(0, 0)).terrain == "missile_silo"
    assert parsed.map_state.tile_at(Coord(1, 0)).terrain == "spent_missile_silo"
