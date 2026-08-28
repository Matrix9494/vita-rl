from advancewars import raw_env
from advancewars.engine import ActionType
from advancewars.engine.coordinates import Coord
from advancewars.env.structured_action_codec import ACTION_KINDS


def test_structured_env_exposes_multidiscrete_masks():
    env = raw_env(action_mode="structured", config="no_production")
    env.reset(seed=0)

    observation, reward, termination, truncation, info = env.last()

    assert reward == 0.0
    assert not termination
    assert not truncation
    assert observation["observation"].shape == (61, 5, 7)
    assert observation["action_type_mask"].shape == (len(ACTION_KINDS),)
    assert observation["source_mask"].shape == (6, 8)
    assert observation["destination_mask"].shape == (6, 8)
    assert observation["target_mask"].shape == (6, 8)
    assert observation["payload_mask"].ndim == 1
    assert info["legal_structured_actions"]
    assert info["legal_semantic_actions"][0]["kind"] == "END_TURN"


def test_structured_env_steps_move_action():
    env = raw_env(action_mode="structured", config="no_production")
    env.reset(seed=0)
    state = env.game.state
    assert state is not None
    unit = state.units[1]
    start = unit.coord

    _observation, _reward, _termination, _truncation, info = env.last()
    move_kind = ACTION_KINDS.index(ActionType.MOVE)
    move_action = next(
        action for action in info["legal_structured_actions"] if action[0] == move_kind
    )

    env.step(move_action)

    assert unit.coord != start
    assert unit.coord.y == move_action[3]
    assert unit.coord.x == move_action[4]


def test_structured_env_rejects_illegal_tuple():
    env = raw_env(action_mode="structured", config="no_production")
    env.reset(seed=0)

    illegal_end_turn_with_source = (
        ACTION_KINDS.index(ActionType.END_TURN),
        0,
        0,
        5,
        7,
        5,
        7,
        0,
    )

    try:
        env.step(illegal_end_turn_with_source)
    except ValueError as exc:
        assert "Illegal encoded action" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Illegal structured action was accepted.")


def test_structured_env_can_move_then_capture():
    env = raw_env(action_mode="structured", config="no_production")
    env.reset(seed=0)
    state = env.game.state
    assert state is not None
    unit = state.units[1]

    _observation, _reward, _termination, _truncation, info = env.last()
    capture_kind = ACTION_KINDS.index(ActionType.CAPTURE)
    capture_action = next(
        action
        for action in info["legal_structured_actions"]
        if action[0] == capture_kind and action[3:5] == (2, 1)
    )

    env.step(capture_action)

    assert unit.coord.y == 2
    assert unit.coord.x == 1
    assert unit.capture_progress > 0
    assert unit.can_act is False


def test_structured_env_can_move_then_attack():
    env = raw_env(action_mode="structured", config="no_production")
    env.reset(seed=0)
    state = env.game.state
    assert state is not None
    attacker = state.units[1]
    defender = state.units[2]
    attacker.coord = Coord(2, 1)
    defender.coord = Coord(4, 1)
    env._refresh_action_maps()

    _observation, _reward, _termination, _truncation, info = env.last()
    attack_kind = ACTION_KINDS.index(ActionType.ATTACK)
    attack_action = next(
        action
        for action in info["legal_structured_actions"]
        if action[0] == attack_kind
        and action[3:5] == (1, 3)
        and action[5:7] == (1, 4)
    )

    env.step(attack_action)

    assert attacker.coord.x == 3
    assert attacker.coord.y == 1
    assert defender.hp < 100
    assert attacker.can_act is False


def test_structured_env_can_move_then_load():
    env = raw_env(action_mode="structured", config="no_production")
    env.reset(seed=0)
    state = env.game.state
    assert state is not None
    infantry = state.units[1]
    transport = state.units[2]
    infantry.coord = Coord(0, 1)
    transport.owner = 0
    transport.unit_type = "apc"
    transport.coord = Coord(2, 1)
    transport.cargo.clear()
    env._refresh_action_maps()

    _observation, _reward, _termination, _truncation, info = env.last()
    load_kind = ACTION_KINDS.index(ActionType.LOAD)
    load_action = next(
        action
        for action in info["legal_structured_actions"]
        if action[0] == load_kind
        and action[3:5] == (1, 1)
        and action[5:7] == (1, 2)
    )

    env.step(load_action)

    assert infantry.carried_by == transport.id
    assert infantry.id in transport.cargo
    assert infantry.coord == transport.coord
    assert infantry.can_act is False
