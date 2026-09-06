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
