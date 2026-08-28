import json

from advancewars import raw_env
from advancewars.policies import InfantryDecisionTreePolicy
from examples.infantry_self_play import run_self_play


def test_infantry_decision_tree_selects_legal_structured_action():
    env = raw_env(action_mode="structured", config="infantry_only", max_turns=5)
    policy = InfantryDecisionTreePolicy()
    env.reset(seed=0)

    observation, _reward, _termination, _truncation, info = env.last()
    action = policy.choose_action(env, env.agent_selection, observation, info)

    assert action in info["legal_structured_actions"]


def test_infantry_decision_tree_structured_rollout_smoke():
    env = raw_env(action_mode="structured", config="infantry_only", max_turns=5)
    policy = InfantryDecisionTreePolicy()
    env.reset(seed=0)

    steps = 0
    for agent in env.agent_iter(max_iter=40):
        observation, _reward, termination, truncation, info = env.last()
        if termination or truncation:
            break
        action = policy.choose_action(env, agent, observation, info)
        assert action in info["legal_structured_actions"]
        env.step(action)
        steps += 1

    assert steps > 0


def test_infantry_self_play_saves_summary(tmp_path):
    output = tmp_path / "self_play.json"

    summary = run_self_play(rounds=2, output_path=output, max_steps=50, max_turns=5)
    saved = json.loads(output.read_text())

    assert output.exists()
    assert saved["metadata"]["policy"] == "InfantryDecisionTreePolicy"
    assert saved["metadata"]["config"] == "infantry_only"
    assert len(saved["rounds"]) == 2
    assert saved["wins"] == summary["wins"]
    assert all("final_render" in result for result in saved["rounds"])
