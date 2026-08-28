"""Self-play training loop scaffolding."""

from advancewars.training.league import run_iteration
from advancewars.training.policy_config import InfantryPolicyConfig

__all__ = ["InfantryPolicyConfig", "run_iteration"]
