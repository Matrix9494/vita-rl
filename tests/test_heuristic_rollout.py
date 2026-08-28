import json

from advancewars import raw_env
from advancewars.policies import HeuristicPolicy
from examples.heuristic_duel_rollout import run_rollout


def test_raw_env_info_exposes_legal_action_indices():
    env = raw_env()
    env.reset(seed=0)

    observation, _reward, _termination, _truncation, info = env.last()

    assert info["legal_action_indices"]
    assert len(info["legal_action_indices"]) == len(info["legal_actions"])
    assert observation["action_mask"][info["legal_action_indices"][0]]


def test_heuristic_policy_steps_through_pettingzoo_api():
    env = raw_env(max_turns=3)
    policy = HeuristicPolicy()
    env.reset(seed=0)

    steps = 0
    for agent in env.agent_iter(max_iter=8):
        observation, _reward, termination, truncation, info = env.last()
        if termination or truncation:
            break
        action = policy.choose_action(env, agent, observation, info)
        assert observation["action_mask"][action]
        env.step(action)
        steps += 1

    assert steps > 0


def test_heuristic_duel_rollout_saves_json(tmp_path):
    output = tmp_path / "duel_rollout.json"

    trajectory = run_rollout(output_path=output, max_steps=6, seed=0)
    saved = json.loads(output.read_text())

    assert output.exists()
    assert saved["metadata"]["map_name"] == "duel"
    assert saved["metadata"]["policy"] == "HeuristicPolicy"
    assert saved["final"]["steps_taken"] == trajectory["final"]["steps_taken"]
    assert saved["steps"][0]["selected_action_index"] is not None
    assert saved["steps"][0]["state_before"]
    assert saved["steps"][0]["state_after"]
