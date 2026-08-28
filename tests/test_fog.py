import pytest

from advancewars import raw_env
from advancewars.engine import Action, ActionType, Game
from advancewars.engine.coordinates import Coord
from advancewars.env.observations import UNIT_KEYS


def _place_rocket_duel(game: Game):
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "rockets"
    attacker.coord = Coord(1, 1)
    attacker.ammo = {"RocketRockets": 6}
    defender.coord = Coord(5, 1)
    return state, attacker, defender


def test_fog_blocks_unseen_indirect_attack_targets():
    clear_game = Game.from_map("duel")
    _state, attacker, defender = _place_rocket_duel(clear_game)
    assert defender in clear_game.attack_targets(attacker, after_moving=False)

    fog_game = Game.from_map("duel", fog=True)
    _state, attacker, defender = _place_rocket_duel(fog_game)
    assert defender not in fog_game.attack_targets(attacker, after_moving=False)
    with pytest.raises(ValueError, match="visible enemy"):
        fog_game.step(
            Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
        )


def test_fog_reveals_regular_units_inside_vision():
    game = Game.from_map("duel", fog=True)
    state = game.reset()
    scout = state.units[1]
    enemy = state.units[2]
    scout.coord = Coord(0, 1)
    enemy.coord = Coord(2, 1)

    assert enemy.coord in game.visible_coords(scout.owner)
    assert game.is_unit_visible(scout.owner, enemy)
    assert enemy in game.observed_units(scout.owner)


def test_hidden_units_require_adjacency_in_fog():
    game = Game.from_map("duel", fog=True)
    state = game.reset()
    scout = state.units[1]
    hidden = state.units[2]
    scout.coord = Coord(0, 1)
    hidden.coord = Coord(2, 1)
    hidden.unit_type = "sub_sub"

    assert hidden.coord in game.visible_coords(scout.owner)
    assert not game.is_unit_visible(scout.owner, hidden)

    hidden.coord = Coord(1, 1)
    assert game.is_unit_visible(scout.owner, hidden)


def test_fog_observation_hides_unseen_enemy_unit():
    env = raw_env(fog=True)
    env.reset()
    state = env.game.state
    assert state is not None
    enemy = state.units[2]
    enemy.coord = Coord(6, 4)

    obs = env.observe("player_0")["observation"]
    enemy_channel = UNIT_KEYS.index(enemy.unit_type)
    assert obs[enemy_channel, enemy.coord.y, enemy.coord.x] == 0.0


def test_cover_terrain_hides_ground_units_until_adjacent():
    game = Game.from_map("duel", fog=True)
    state = game.reset()
    scout = state.units[1]
    enemy = state.units[2]
    scout.unit_type = "piperunner"
    scout.ammo = {"PipeGun": 9}
    scout.coord = Coord(1, 2)
    enemy.coord = Coord(3, 2)

    assert state.map.tile_at(enemy.coord).terrain == "mountain"
    state.map.tile_at(enemy.coord).terrain = "forest"
    assert enemy.coord in game.visible_coords(scout.owner)
    assert not game.is_unit_visible(scout.owner, enemy)
    assert enemy not in game.attack_targets(scout, after_moving=False)

    scout.coord = Coord(2, 2)
    assert game.is_unit_visible(scout.owner, enemy)
    assert enemy in game.observed_units(scout.owner)


def test_cover_terrain_hides_sea_units_on_reefs_until_adjacent():
    game = Game.from_map("duel", fog=True)
    state = game.reset()
    scout = state.units[1]
    enemy = state.units[2]
    scout.unit_type = "battleship"
    scout.coord = Coord(1, 2)
    scout.ammo = {"BattleshipCannon": 9}
    enemy.unit_type = "lander"
    enemy.coord = Coord(3, 2)
    state.map.tile_at(enemy.coord).terrain = "reef"

    assert enemy.coord in game.visible_coords(scout.owner)
    assert not game.is_unit_visible(scout.owner, enemy)

    scout.coord = Coord(2, 2)
    assert game.is_unit_visible(scout.owner, enemy)


def test_air_units_over_cover_are_visible_inside_normal_vision():
    game = Game.from_map("duel", fog=True)
    state = game.reset()
    scout = state.units[1]
    enemy = state.units[2]
    scout.coord = Coord(1, 2)
    enemy.unit_type = "bomber"
    enemy.coord = Coord(3, 2)
    state.map.tile_at(enemy.coord).terrain = "forest"

    assert enemy.coord in game.visible_coords(scout.owner)
    assert game.is_unit_visible(scout.owner, enemy)


def test_drive_by_vision_memory_keeps_units_visible_after_moving_past():
    game = Game.from_map("duel", fog=True)
    state = game.reset()
    scout = state.units[1]
    enemy = state.units[2]
    scout.unit_type = "tank"
    scout.coord = Coord(1, 2)
    scout.fuel = 70
    enemy.coord = Coord(4, 4)
    for coord in (Coord(2, 2), Coord(3, 2), Coord(4, 2)):
        state.map.tile_at(coord).terrain = "road"
    rocket_id = max(state.units) + 1
    state.units[rocket_id] = scout.__class__(
        id=rocket_id,
        owner=0,
        unit_type="rockets",
        coord=Coord(0, 4),
        fuel=50,
        ammo={"RocketRockets": 6},
        can_act=True,
    )
    rockets = state.units[rocket_id]

    assert not game.is_unit_visible(0, enemy)

    game.step(
        Action(
            ActionType.MOVE,
            unit_id=scout.id,
            path=(
                Coord(2, 2),
                Coord(3, 2),
                Coord(4, 2),
                Coord(3, 2),
                Coord(2, 2),
            ),
        )
    )

    assert scout.coord == Coord(2, 2)
    assert enemy.coord in game.visible_coords(0)
    assert game.is_unit_visible(0, enemy)
    assert enemy in game.attack_targets(rockets, after_moving=False)


def test_drive_by_vision_memory_keeps_cover_units_revealed():
    game = Game.from_map("duel", fog=True)
    state = game.reset()
    scout = state.units[1]
    enemy = state.units[2]
    scout.unit_type = "tank"
    scout.coord = Coord(1, 3)
    scout.fuel = 70
    enemy.coord = Coord(4, 4)
    state.map.tile_at(enemy.coord).terrain = "forest"
    for coord in (Coord(2, 3), Coord(3, 3), Coord(4, 3)):
        state.map.tile_at(coord).terrain = "road"
    rocket_id = max(state.units) + 1
    state.units[rocket_id] = scout.__class__(
        id=rocket_id,
        owner=0,
        unit_type="rockets",
        coord=Coord(0, 4),
        fuel=50,
        ammo={"RocketRockets": 6},
        can_act=True,
    )
    rockets = state.units[rocket_id]

    assert not game.is_unit_visible(0, enemy)

    game.step(
        Action(
            ActionType.MOVE,
            unit_id=scout.id,
            path=(
                Coord(2, 3),
                Coord(3, 3),
                Coord(4, 3),
                Coord(3, 3),
                Coord(2, 3),
            ),
        )
    )

    assert scout.coord == Coord(2, 3)
    assert game.is_unit_visible(0, enemy)
    assert enemy in game.attack_targets(rockets, after_moving=False)
