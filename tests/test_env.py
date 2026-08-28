from advancewars import env, parallel_env, raw_env
from advancewars.engine import Action, ActionType
from advancewars.engine.coordinates import Coord


def test_raw_env_reset_and_legal_step():
    env = raw_env(render_mode="ansi")
    env.reset()

    obs, reward, termination, truncation, info = env.last()
    assert reward == 0.0
    assert not termination
    assert not truncation
    assert obs["action_mask"].sum() > 0

    first_legal = int(obs["action_mask"].nonzero()[0][0])
    env.step(first_legal)
    assert env.render()
    assert "legal_actions" in env.infos[env.agent_selection]


def test_env_returns_wrapped_aec_env_that_can_step():
    wrapped = env(render_mode="ansi")
    wrapped.reset(seed=3)

    obs, reward, termination, truncation, _info = wrapped.last()
    assert reward == 0.0
    assert not termination
    assert not truncation
    first_legal = int(obs["action_mask"].nonzero()[0][0])

    wrapped.step(first_legal)

    assert wrapped.render()


def test_raw_env_legal_rollout_smoke():
    env = raw_env(render_mode="ansi", max_turns=5)
    env.reset()

    steps = 0
    for _agent in env.agent_iter(max_iter=50):
        obs, _reward, termination, truncation, _info = env.last()
        if termination or truncation:
            break
        legal = obs["action_mask"].nonzero()[0]
        env.step(int(legal[0]))
        steps += 1

    assert steps > 0


def test_raw_env_max_turns_reports_truncation_not_termination():
    env = raw_env(max_turns=1)
    env.reset()

    for _ in range(4):
        obs, _reward, termination, truncation, _info = env.last()
        assert not termination
        assert not truncation
        legal = obs["action_mask"].nonzero()[0]
        env.step(int(legal[0]))
        if not env.agents:
            break

    assert env.game.state is not None
    assert env.game.state.truncated is True
    assert env.game.state.done is False
    assert env.game.state.winner is None
    assert env.terminations == {"player_0": False, "player_1": False}
    assert env.truncations == {"player_0": True, "player_1": True}
    assert env.rewards == {"player_0": 0.0, "player_1": 0.0}


def test_raw_env_reset_seed_reproducibly_seeds_luck():
    def attack_luck(seed: int) -> int:
        env = raw_env(luck=True)
        env.reset(seed=seed)
        state = env.game.state
        assert state is not None
        attacker = state.units[1]
        defender = state.units[2]
        attacker.coord = Coord(2, 1)
        defender.coord = Coord(3, 1)
        _state, events = env.game.step(
            Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
        )
        return next(event for event in events if event.type == "attack").payload["luck"]

    assert attack_luck(11) == attack_luck(11)


def test_parallel_env_reset_and_legal_step():
    env = parallel_env(max_turns=3)
    observations, infos = env.reset(seed=5)

    assert set(observations) == {"player_0", "player_1"}
    assert set(infos) == {"player_0", "player_1"}

    actions = {}
    for agent, observation in observations.items():
        legal = observation["action_mask"].nonzero()[0]
        if len(legal):
            actions[agent] = int(legal[0])

    observations, rewards, terminations, truncations, infos = env.step(actions)

    assert set(rewards) == {"player_0", "player_1"}
    assert set(terminations) == {"player_0", "player_1"}
    assert set(truncations) == {"player_0", "player_1"}
    assert set(infos) == {"player_0", "player_1"}
    assert observations
