import json

import pytest

from advancewars import raw_env
from advancewars.engine import Action, ActionType, Game
from advancewars.engine.coordinates import Coord


def test_invalid_weather_is_rejected():
    with pytest.raises(ValueError, match="Unknown weather"):
        Game.from_map("duel", weather="hail")


def test_snow_changes_foot_movement_costs():
    clear_game = Game.from_map("duel", weather="clear")
    clear_state = clear_game.reset()
    clear_infantry = clear_state.units[1]
    clear_infantry.coord = Coord(3, 1)

    snow_game = Game.from_map("duel", weather="snow")
    snow_state = snow_game.reset()
    snow_infantry = snow_state.units[1]
    snow_infantry.coord = Coord(3, 1)

    assert Coord(3, 2) in clear_game.reachable_destinations(clear_infantry)
    assert Coord(3, 2) not in snow_game.reachable_destinations(snow_infantry)


def test_mechs_can_enter_snowy_mountains_where_infantry_cannot():
    game = Game.from_map("duel", weather="snow")
    state = game.reset()
    mech = state.units[1]
    mech.unit_type = "mech"
    mech.coord = Coord(3, 1)

    assert Coord(3, 2) in game.reachable_destinations(mech)


def test_rain_reduces_fog_vision():
    clear_game = Game.from_map("duel", fog=True, weather="clear")
    clear_state = clear_game.reset()
    clear_scout = clear_state.units[1]
    clear_enemy = clear_state.units[2]
    clear_scout.coord = Coord(0, 1)
    clear_enemy.coord = Coord(2, 1)

    rain_game = Game.from_map("duel", fog=True, weather="rain")
    rain_state = rain_game.reset()
    rain_scout = rain_state.units[1]
    rain_enemy = rain_state.units[2]
    rain_scout.coord = Coord(0, 1)
    rain_enemy.coord = Coord(2, 1)

    assert clear_game.is_unit_visible(clear_scout.owner, clear_enemy)
    assert not rain_game.is_unit_visible(rain_scout.owner, rain_enemy)


def test_weather_serialization_round_trip():
    game = Game.from_map("duel", fog=True, weather="snow")
    game.reset()
    restored = Game.from_json(json.loads(json.dumps(game.to_json())))

    assert restored.initial_weather == "snow"
    assert restored.state is not None
    assert restored.state.weather == "snow"


def test_temporary_weather_expires_after_one_round_and_restores_base_weather():
    game = Game.from_map("duel", commanders={0: "olaf", 1: "andy"}, weather="rain")
    state = game.reset()
    state.players[0].power_charge = 30

    _state, events = game.step(Action(ActionType.CO_ABILITY))
    weather_event = next(event for event in events if event.type == "weather")

    assert weather_event.payload["weather"] == "snow"
    assert state.weather == "snow"
    assert state.weather_base == "rain"
    assert state.weather_turns_remaining == 1

    game.step(Action.end_turn())
    assert state.current_player == 1
    assert state.weather == "snow"
    assert state.weather_turns_remaining == 0

    _state, events = game.step(Action.end_turn())
    restore_event = next(event for event in events if event.type == "weather")
    assert state.current_player == 0
    assert state.weather == "rain"
    assert restore_event.payload["weather"] == "rain"


def test_temporary_weather_serialization_round_trip_preserves_timer():
    game = Game.from_map("duel", commanders={0: "olaf", 1: "andy"})
    state = game.reset()
    state.players[0].power_charge = 30

    game.step(Action(ActionType.CO_ABILITY))
    restored = Game.from_json(json.loads(json.dumps(game.to_json())))

    assert restored.state is not None
    assert restored.state.weather == "snow"
    assert restored.state.weather_base == "clear"
    assert restored.state.weather_turns_remaining == 1


def test_env_observation_includes_weather_plane():
    env = raw_env(weather="snow")
    env.reset()
    obs = env.observe("player_0")["observation"]

    assert obs[-3].min() == 2.0
    assert obs[-3].max() == 2.0
