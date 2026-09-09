"""Deterministic delayed-recall environment with Gymnasium-compatible methods."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

try:  # Gymnasium is an install dependency; the fallback keeps source tests lightweight.
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:  # pragma: no cover - exercised only without package installation.
    gym = None
    spaces = None


class _DiscreteFallback:
    """Small subset of ``gymnasium.spaces.Discrete`` used before installation."""

    def __init__(self, n: int) -> None:
        self.n = n

    def contains(self, value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < self.n

    def sample(self) -> int:
        return Random().randrange(self.n)


@dataclass(frozen=True)
class Memory1Config:
    """Static task parameters for one family of delayed-recall episodes."""

    num_items: int = 4
    cue_steps: int = 1
    delay_steps: int = 3
    max_episode_steps: int | None = None

    def __post_init__(self) -> None:
        if self.num_items < 2:
            raise ValueError("num_items must be at least 2")
        if self.cue_steps < 1:
            raise ValueError("cue_steps must be positive")
        if self.delay_steps < 0:
            raise ValueError("delay_steps cannot be negative")
        if self.max_episode_steps is not None and self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive when provided")

    @property
    def episode_limit(self) -> int:
        return self.max_episode_steps or self.cue_steps + self.delay_steps + 1


class Memory1Env(gym.Env if gym is not None else object):
    """Remember a cue across blank steps and select it at a terminal query.

    Actions during the cue and delay phases advance time but do not alter the
    target. Only the action issued in the query phase is scored.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(self, config: Memory1Config | None = None, render_mode: str | None = None) -> None:
        self.config = config or Memory1Config()
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        self.render_mode = render_mode
        self._rng = Random()
        self._target_action: int | None = None
        self._phase = "cue"
        self._cue_remaining = 0
        self._delay_remaining = 0
        self._elapsed_steps = 0
        self._finished = False
        self.action_space = self._discrete(self.config.num_items)
        self.observation_space = self._observation_space()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, int], dict[str, Any]]:
        """Start a seeded episode and reveal the target cue."""
        del options
        if seed is not None:
            self._rng.seed(seed)
            if gym is not None:
                super().reset(seed=seed)
        self._target_action = self._rng.randrange(self.config.num_items)
        self._phase = "cue"
        self._cue_remaining = self.config.cue_steps
        self._delay_remaining = self.config.delay_steps
        self._elapsed_steps = 0
        self._finished = False
        return self._observation(), {"phase": self._phase}

    def step(self, action: int) -> tuple[dict[str, int], float, bool, bool, dict[str, Any]]:
        """Advance one phase; score only the terminal query action."""
        if self._target_action is None:
            raise RuntimeError("Call reset() before step()")
        if self._finished:
            raise RuntimeError("Episode has finished; call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected an integer in [0, {self.config.num_items})")

        self._elapsed_steps += 1
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {"phase": self._phase}

        if self._phase == "cue":
            self._cue_remaining -= 1
            if self._cue_remaining == 0:
                self._phase = "delay" if self._delay_remaining else "query"
        elif self._phase == "delay":
            self._delay_remaining -= 1
            if self._delay_remaining == 0:
                self._phase = "query"
        else:
            terminated = True
            reward = float(action == self._target_action)
            info = {"phase": "query", "target_action": self._target_action, "correct": bool(reward)}

        if not terminated and self._elapsed_steps >= self.config.episode_limit:
            truncated = True
            info = {"phase": self._phase, "time_limit_reached": True}
        self._finished = terminated or truncated
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> str:
        """Return a compact human-readable representation without leaking the cue."""
        if self._target_action is None:
            return "Memory1Env(uninitialized)"
        observation = self._observation()
        return f"Memory1Env(phase={self._phase}, cue={observation['cue']}, remaining_steps={observation['remaining_steps']})"

    def close(self) -> None:
        """Match the Gymnasium lifecycle API; this environment owns no resources."""

    def _observation(self) -> dict[str, int]:
        assert self._target_action is not None
        cue = self._target_action if self._phase == "cue" else self.config.num_items
        remaining = self._cue_remaining + self._delay_remaining + int(self._phase == "query")
        return {"phase": {"cue": 0, "delay": 1, "query": 2}[self._phase], "cue": cue, "remaining_steps": remaining}

    def _discrete(self, n: int) -> Any:
        return spaces.Discrete(n) if spaces is not None else _DiscreteFallback(n)

    def _observation_space(self) -> Any:
        if spaces is None:
            return {
                "phase": _DiscreteFallback(3),
                "cue": _DiscreteFallback(self.config.num_items + 1),
                "remaining_steps": _DiscreteFallback(self.config.episode_limit + 1),
            }
        return spaces.Dict({
            "phase": spaces.Discrete(3),
            "cue": spaces.Discrete(self.config.num_items + 1),
            "remaining_steps": spaces.Discrete(self.config.episode_limit + 1),
        })


def register_gymnasium() -> None:
    """Register ``Memory1-v0`` once Gymnasium is installed."""
    if gym is None:
        raise RuntimeError("Install memory-1 (or gymnasium) before registering the Gymnasium environment")
    registry = gym.registry
    if "Memory1-v0" not in registry:
        gym.register(id="Memory1-v0", entry_point="memory_1:Memory1Env")
