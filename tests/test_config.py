import json

import pytest

from advancewars.engine import Action, ActionType, Game, load_config
from advancewars.engine.coordinates import Coord


def test_named_config_filters_build_options():
    game = Game.from_map("duel", config="infantry_only")
    state = game.reset()
    state.players[0].funds = 20_000

    build_units = {
        action.build_unit
        for action in game.legal_actions()
        if action.type == ActionType.BUILD
    }

    assert build_units == {"infantry"}


def test_disabled_unit_cannot_be_built():
    game = Game.from_map("duel", enabled_units=["infantry"])
    state = game.reset()
    state.players[0].funds = 20_000

    with pytest.raises(ValueError, match="disabled by config"):
        game.step(Action(ActionType.BUILD, target=Coord(1, 0), build_unit="tank"))


def test_strict_units_rejects_disabled_initial_map_units():
    game = Game.from_map("duel", enabled_units=["tank"], strict_units=True)

    with pytest.raises(ValueError, match="disabled by config"):
        game.reset()


def test_json_round_trip_preserves_config():
    game = Game.from_map("duel", config={"enabled_units": ["infantry"]})
    game.reset()

    restored = Game.from_json(json.loads(json.dumps(game.to_json())))

    assert restored.config.enabled_units == frozenset({"infantry"})
    assert restored.config.unit_enabled("tank") is False


def test_load_config_normalizes_unit_aliases():
    config = load_config({"enabled_units": ["Anti-Air", "md tank"]})

    assert config.enabled_units == frozenset({"anti_air", "md_tank"})
