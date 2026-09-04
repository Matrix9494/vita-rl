from vita_rl.baseline import BaselineSpec
from vita_rl.dressage_adapter import create_adapter
from vita_rl.reward import compute_reward
from vita_rl.state import State
from vita_rl.trajectory import Transition


def test_placeholder_interfaces_import():
    assert BaselineSpec().task_id == "10711001"
    assert Transition(observation={}, action={}).done is False
    assert isinstance(State(), dict)
    assert compute_reward(None) == 0.0
    assert callable(create_adapter)
