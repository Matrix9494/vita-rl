"""Trajectory interfaces reserved for future rollout collection."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Transition:
    """A minimal transition record; collection logic is intentionally absent."""

    observation: Any
    action: Any
    reward: float | None = None
    next_observation: Any = None
    done: bool = False
