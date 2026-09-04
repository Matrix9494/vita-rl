"""Reward interface placeholder for post-training experiments."""

from typing import Any


def compute_reward(trajectory: Any) -> float:
    """Return a neutral placeholder reward until a method defines one."""

    del trajectory
    return 0.0
