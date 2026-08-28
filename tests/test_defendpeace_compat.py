from pathlib import Path

import pytest

from advancewars.engine import Action, ActionType, Game, load_map
from advancewars.engine.coordinates import Coord
from advancewars.engine.data import BUILD_LISTS, UNITS
from advancewars.utils.defendpeace_map import parse_defendpeace_map


DEFENDPEACE_ROOT = Path("/tmp/DefendPeace")


DEFENDPEACE_MAP_FIXTURES = (
    pytest.param(
        "res/map/Cartridge/AW1_Duo_Falls.map",
        33,
        13,
        (0, 1),
        0,
        id="cartridge-aw1-duo-falls",
    ),
    pytest.param(
        "res/map/Cartridge/AW2_Spann_Island.map",
        15,
        10,
        (0, 1),
        0,
        id="cartridge-aw2-spann-island",
    ),
    pytest.param(
        "res/map/Cartridge/Blizzard_Battle.map",
        19,
        13,
        (0, 1),
        20,
        id="cartridge-units",
    ),
    pytest.param(
        "res/map/Cartridge/Alakule.map",
        32,
        28,
        (0, 1, 2, 3),
        0,
        id="cartridge-four-player",
    ),
    pytest.param(
        "res/map/FoW/First_Step_Into_Fog.map",
        24,
        16,
        (0, 1),
        1,
        id="fog-basic",
    ),
    pytest.param(
        "res/map/FoW/Misty.map",
        19,
        21,
        (8, 10),
        9,
        id="fog-nonzero-players",
    ),
    pytest.param(
        "res/map/FFA/Light_Tanker.map",
        20,
        20,
        (7, 8, 9, 11),
        2,
        id="ffa-noncontiguous-players",
    ),
    pytest.param(
        "res/map/League/Brush_With_Death.map",
        25,
        22,
        (6, 11),
        6,
        id="league-units",
    ),
    pytest.param(
        "res/map/HF/Titans_Rise.map",
        24,
        19,
        (8, 10),
        10,
        id="hf-uppercase-hyphen-unit-names",
    ),
    pytest.param(
        "res/map/New/Parallel_Worlds.map",
        23,
        26,
        (0, 3),
        4,
        id="new-map",
    ),
)


DEFENDPEACE_COMBAT_FIXTURES = (
    pytest.param(
        "tank_vs_tank_plain",
        "tank",
        "tank",
        Coord(2, 1),
        Coord(3, 1),
        100,
        100,
        {"TankCannon": 9},
        {"TankCannon": 9},
        55,
        id="tank-vs-tank-plain",
    ),
    pytest.param(
        "tank_vs_tank_city",
        "tank",
        "tank",
        Coord(3, 1),
        Coord(3, 0),
        100,
        100,
        {"TankCannon": 9},
        {"TankCannon": 9},
        38,
        id="tank-vs-tank-city-defense",
    ),
    pytest.param(
        "low_hp_tank_vs_tank",
        "tank",
        "tank",
        Coord(2, 1),
        Coord(3, 1),
        50,
        100,
        {"TankCannon": 9},
        {"TankCannon": 9},
        27,
        id="low-hp-attacker",
    ),
    pytest.param(
        "artillery_vs_tank",
        "artillery",
        "tank",
        Coord(2, 1),
        Coord(4, 1),
        100,
        100,
        {"ArtilleryCannon": 9},
        {"TankCannon": 9},
        70,
        id="indirect-fire",
    ),
)


def _set_unit_for_fixture(unit, unit_type: str, coord: Coord, hp: int, ammo: dict[str, int]):
    unit.unit_type = unit_type
    unit.coord = coord
    unit.hp = hp
    unit.ammo = dict(ammo)


def test_awbw_roster_has_defendpeace_unit_count():
    # DefendPeace AWBWUnitEnum has 27 entries, including hidden transform states.
    assert len(UNITS) == 27
    assert set(BUILD_LISTS["factory"]) >= {
        "infantry",
        "mech",
        "apc",
        "tank",
        "md_tank",
        "neotank",
        "megatank",
        "artillery",
        "rockets",
        "piperunner",
        "anti_air",
        "mobilesam",
    }
    assert set(BUILD_LISTS["airport"]) >= {
        "t_copter",
        "b_copter",
        "fighter",
        "bomber",
        "stealth",
        "bbomb",
    }
    assert set(BUILD_LISTS["seaport"]) >= {
        "lander",
        "cruiser",
        "sub",
        "battleship",
        "carrier",
        "bboat",
    }


def test_awbw_damage_table_known_values():
    tank_cannon = UNITS["tank"].weapons[0]
    anti_air = UNITS["anti_air"].weapons[0]
    fighter = UNITS["fighter"].weapons[0]
    battleship = UNITS["battleship"].weapons[0]

    assert tank_cannon.damage["tank"] == 55
    assert tank_cannon.damage["infantry"] == 25
    assert anti_air.damage["b_copter"] == 120
    assert fighter.damage["bomber"] == 100
    assert battleship.min_range == 2
    assert battleship.max_range == 6
    assert battleship.damage["cruiser"] == 95


@pytest.mark.parametrize(
    (
        "name",
        "attacker_type",
        "defender_type",
        "attacker_coord",
        "defender_coord",
        "attacker_hp",
        "defender_hp",
        "attacker_ammo",
        "defender_ammo",
        "expected_damage",
    ),
    DEFENDPEACE_COMBAT_FIXTURES,
)
def test_defendpeace_combat_fixture_damage_values(
    name,
    attacker_type,
    defender_type,
    attacker_coord,
    defender_coord,
    attacker_hp,
    defender_hp,
    attacker_ammo,
    defender_ammo,
    expected_damage,
):
    game = Game.from_map("duel")
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    _set_unit_for_fixture(
        attacker, attacker_type, attacker_coord, attacker_hp, attacker_ammo
    )
    _set_unit_for_fixture(
        defender, defender_type, defender_coord, defender_hp, defender_ammo
    )

    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    attack = next(event for event in events if event.type == "attack")

    assert name
    assert attack.payload["damage"] == expected_damage


def test_defendpeace_combat_fixture_counterattack_uses_post_hit_hp():
    game = Game.from_map("duel")
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    _set_unit_for_fixture(attacker, "tank", Coord(2, 1), 100, {"TankCannon": 9})
    _set_unit_for_fixture(defender, "tank", Coord(3, 1), 100, {"TankCannon": 9})

    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    counter = next(event for event in events if event.type == "counterattack")

    assert counter.payload["damage"] == 24
    assert attacker.hp == 76
    assert defender.hp == 45
    assert defender.ammo["TankCannon"] == 8


def test_defendpeace_capture_fixture_uses_hp_tenths_progress():
    game = Game.from_map("duel")
    state = game.reset()
    infantry = state.units[1]
    infantry.coord = Coord(3, 0)
    infantry.hp = 55

    _state, events = game.step(Action(ActionType.CAPTURE, unit_id=infantry.id))

    assert events[0].type == "capture_progress"
    assert events[0].payload["progress"] == 5
    assert infantry.capture_progress == 5
    assert state.map.tile_at(Coord(3, 0)).owner is None


def test_defendpeace_production_fixture_factory_builds_unready_infantry():
    game = Game.from_map("duel")
    state = game.reset()
    state.players[0].funds = 1000
    target = Coord(1, 0)

    _state, events = game.step(
        Action(ActionType.BUILD, target=target, build_unit="infantry")
    )
    built = state.unit_at(target)

    assert events[0].type == "build"
    assert built is not None
    assert built.unit_type == "infantry"
    assert built.owner == 0
    assert built.can_act is False
    assert built.fuel == UNITS["infantry"].max_fuel
    assert state.players[0].funds == 0


@pytest.mark.skipif(
    not DEFENDPEACE_ROOT.exists(), reason="DefendPeace checkout not available"
)
@pytest.mark.parametrize(
    ("relative_path", "width", "height", "player_ids", "unit_count"),
    DEFENDPEACE_MAP_FIXTURES,
)
def test_parse_multiple_defendpeace_map_fixtures(
    relative_path,
    width,
    height,
    player_ids,
    unit_count,
):
    map_path = DEFENDPEACE_ROOT / relative_path
    parsed = parse_defendpeace_map(load_map(str(map_path)))

    assert parsed.map_state.width == width
    assert parsed.map_state.height == height
    assert parsed.player_ids == player_ids
    assert len(parsed.units) == unit_count

    game = Game.from_map(str(map_path))
    state = game.reset()
    assert tuple(sorted(state.players)) == player_ids
    assert len(state.units) == unit_count
    assert state.current_player == min(player_ids)


@pytest.mark.skipif(
    not DEFENDPEACE_ROOT.exists(), reason="DefendPeace checkout not available"
)
def test_defendpeace_map_parser_normalizes_hyphenated_unit_names():
    map_path = DEFENDPEACE_ROOT / "res/map/HF/Titans_Rise.map"
    parsed = parse_defendpeace_map(load_map(str(map_path)))

    anti_air_units = [
        unit for unit in parsed.units.values() if unit.unit_type == "anti_air"
    ]
    assert len(anti_air_units) == 2


@pytest.mark.skipif(
    not DEFENDPEACE_ROOT.exists(), reason="DefendPeace checkout not available"
)
def test_all_defendpeace_res_maps_parse_and_reset():
    map_paths = sorted((DEFENDPEACE_ROOT / "res/map").rglob("*.map"))
    failures: list[str] = []

    assert len(map_paths) >= 80
    for map_path in map_paths:
        relative_path = map_path.relative_to(DEFENDPEACE_ROOT)
        try:
            parsed = parse_defendpeace_map(load_map(str(map_path)))
            state = Game.from_map(str(map_path)).reset()
            assert state.map.width == parsed.map_state.width
            assert state.map.height == parsed.map_state.height
            assert tuple(sorted(state.players)) == parsed.player_ids
        except Exception as exc:  # pragma: no cover - failure details for pytest
            failures.append(f"{relative_path}: {type(exc).__name__}: {exc}")

    if failures:
        pytest.fail("\n".join(failures))


@pytest.mark.skipif(
    not DEFENDPEACE_ROOT.exists(), reason="DefendPeace checkout not available"
)
def test_parse_defendpeace_spann_island_map_file():
    map_path = DEFENDPEACE_ROOT / "res/map/Cartridge/AW2_Spann_Island.map"
    text = load_map(str(map_path))
    parsed = parse_defendpeace_map(text)

    assert parsed.map_state.width == 15
    assert parsed.map_state.height == 10
    assert parsed.map_state.tile_at(Coord(2, 6)).terrain == "hq"
    assert parsed.map_state.tile_at(Coord(2, 6)).owner == 0
    assert parsed.map_state.tile_at(Coord(12, 2)).owner == 1


@pytest.mark.skipif(
    not DEFENDPEACE_ROOT.exists(), reason="DefendPeace checkout not available"
)
def test_game_can_reset_from_external_defendpeace_map_path():
    map_path = DEFENDPEACE_ROOT / "res/map/Cartridge/AW2_Spann_Island.map"
    game = Game.from_map(str(map_path))
    state = game.reset()

    assert sorted(state.players) == [0, 1]
    assert state.current_player == 0
