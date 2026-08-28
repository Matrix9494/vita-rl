import pytest

from advancewars.engine import Action, ActionType, Game
from advancewars.engine.coordinates import Coord
from advancewars.env.rewards import rewards_for


def test_none_reward_mode_is_always_zero():
    game = Game.from_map("duel")
    state = game.reset()
    rewards = rewards_for(state, "none")

    assert rewards == {"player_0": 0.0, "player_1": 0.0}


def test_dense_basic_rewards_combat_damage():
    game = Game.from_map("duel")
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(5, 1)
    defender.coord = Coord(6, 1)

    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    rewards = rewards_for(state, "dense_basic", events)

    assert rewards["player_0"] > 0
    assert rewards["player_1"] < 0


def test_dense_basic_rewards_destroyed_unit_value():
    game = Game.from_map("duel")
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(5, 1)
    defender.coord = Coord(6, 1)
    defender.hp = 30

    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    attack = next(event for event in events if event.type == "attack")
    rewards = rewards_for(state, "dense_basic", events)

    assert attack.payload["defender_hp_after"] == 0
    assert rewards["player_0"] > attack.payload["damage"] / 1000.0
    assert rewards["player_1"] < -attack.payload["damage"] / 1000.0


def test_dense_basic_rewards_property_income_swing_on_capture():
    game = Game.from_map("duel")
    state = game.reset()
    infantry = state.units[1]
    infantry.coord = Coord(3, 0)
    infantry.capture_progress = 10

    _state, events = game.step(Action(ActionType.CAPTURE, unit_id=infantry.id))
    rewards = rewards_for(state, "dense_basic", events)

    assert state.map.tile_at(Coord(3, 0)).definition.profitable
    assert rewards["player_0"] == pytest.approx(0.3)
    assert rewards["player_1"] == 0.0


def test_dense_basic_rewards_hq_capture_pressure():
    game = Game.from_map("duel")
    state = game.reset()
    infantry = state.units[1]
    infantry.coord = Coord(6, 0)

    _state, events = game.step(Action(ActionType.CAPTURE, unit_id=infantry.id))
    rewards = rewards_for(state, "dense_basic", events)

    assert state.map.tile_at(Coord(6, 0)).terrain == "hq"
    assert rewards["player_0"] == 0.1
