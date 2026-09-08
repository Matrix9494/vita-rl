from vita_rl.dressage_adapter import create_adapter
from vita_rl.environments import tool_environment_registry
from vita_rl.reward import compute_environment_reward
from vita_rl.state import State


def test_runtime_interfaces_import():
    assert isinstance(State(), dict)
    assert tool_environment_registry.names() == ("vita-mini",)
    assert compute_environment_reward(None) == 0.0
    assert callable(create_adapter)
