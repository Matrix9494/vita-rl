import json

from advancewars.engine import Action, ActionType, Game
from advancewars.engine.coordinates import Coord


def test_game_json_round_trip_preserves_state_and_options():
    game = Game.from_map("duel", fog=True, max_turns=7)
    state = game.reset()
    infantry = state.units[1]
    infantry.coord = Coord(0, 1)
    apc_id = max(state.units) + 1
    state.units[apc_id] = infantry.__class__(
        id=apc_id,
        owner=0,
        unit_type="apc",
        coord=Coord(1, 1),
        fuel=70,
        can_act=True,
    )
    game.step(Action(ActionType.LOAD, unit_id=infantry.id, target=Coord(1, 1)))

    payload = json.loads(json.dumps(game.to_json()))
    restored = Game.from_json(payload)
    restored_state = restored.state
    assert restored_state is not None

    assert restored.fog
    assert restored.max_turns == 7
    assert restored_state.current_player == state.current_player
    assert restored_state.units[infantry.id].carried_by == apc_id
    assert restored_state.units[apc_id].cargo == [infantry.id]
    assert len(restored.legal_actions()) > 0


def test_game_json_round_trip_preserves_luck_rng_state():
    game = Game.from_map("duel", luck=True, seed=3)
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)
    game.step(Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord))

    restored = Game.from_json(json.loads(json.dumps(game.to_json())))
    assert restored.luck is True
    assert restored.seed == 3
    assert restored._rng.getstate() == game._rng.getstate()


def test_game_json_round_trip_preserves_power_kind_and_stun_state():
    game = Game.from_map("duel", commanders={0: "von_bolt", 1: "andy"})
    state = game.reset()
    state.players[0].power_charge = 100
    state.units[2].coord = Coord(4, 1)

    game.step(Action(ActionType.CO_ABILITY))

    restored = Game.from_json(json.loads(json.dumps(game.to_json())))
    assert restored.state is not None
    assert restored.state.players[0].active_power_kind == "power"
    assert restored.state.units[2].stunned_turns == 1


def test_game_json_round_trip_preserves_fog_memory():
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

    restored = Game.from_json(json.loads(json.dumps(game.to_json())))

    assert restored.state is not None
    assert Coord(4, 4) in restored.visible_coords(0)
    assert restored.is_unit_visible(0, restored.state.units[2])


def test_game_json_round_trip_preserves_truncation_state():
    game = Game.from_map("duel", max_turns=1)
    state = game.reset()

    game.step(Action.end_turn())
    game.step(Action.end_turn())

    restored = Game.from_json(json.loads(json.dumps(game.to_json())))

    assert state.truncated is True
    assert state.done is False
    assert restored.state is not None
    assert restored.state.truncated is True
    assert restored.state.done is False
    assert restored.state.winner is None
