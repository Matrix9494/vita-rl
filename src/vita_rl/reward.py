"""Vita terminal-reward bridge for slime/Dressage samples."""

from typing import Any, Callable


try:  # Keep this module importable by the repository's lightweight tests.
    from dressage.reward import register_reward as _dressage_register_reward
except ImportError:
    def _register_reward(_name: str) -> Callable[[Callable[..., float]], Callable[..., float]]:
        return lambda function: function
else:
    _register_reward = _dressage_register_reward


@_register_reward("vita")
def compute_reward(sample: Any, **_kwargs: Any) -> float:
    """Return the terminal VitaBench reward recorded by the whitebox adapter."""
    metadata = getattr(sample, "metadata", None)
    if metadata is None and isinstance(sample, dict): metadata = sample.get("metadata", sample)
    if not isinstance(metadata, dict): return 0.0
    try: return float(metadata.get("vita_reward", 0.0))
    except (TypeError, ValueError) as exc: raise ValueError("vita_reward must be numeric") from exc


@_register_reward("vita_smoke_dense")
def compute_smoke_dense_reward(sample: Any, **_kwargs: Any) -> float:
    """Terminal reward plus a small trajectory-progress signal for smoke runs.

    The production ``vita`` reward remains terminal-only. This separate,
    opt-in reward prevents an all-zero exploratory group from turning an
    end-to-end GRPO validation into a no-op before task-specific shaping has
    been selected.
    """
    metadata = getattr(sample, "metadata", None)
    if metadata is None and isinstance(sample, dict):
        metadata = sample.get("metadata", sample)
    if not isinstance(metadata, dict):
        return 0.0
    terminal = compute_reward(sample)
    try:
        turns = max(0, int(metadata.get("vita_num_agent_turns", 0)))
    except (TypeError, ValueError) as exc:
        raise ValueError("vita_num_agent_turns must be an integer") from exc
    return terminal + min(turns, 300) / 300.0
