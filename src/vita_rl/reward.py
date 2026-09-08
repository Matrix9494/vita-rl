"""Vita terminal-reward bridge for slime/Dressage samples."""

from typing import Any, Callable


try:  # Keep this module importable by the repository's lightweight tests.
    from dressage.reward import register_reward as _dressage_register_reward
except ImportError:
    def _register_reward(_name: str) -> Callable[[Callable[..., float]], Callable[..., float]]:
        return lambda function: function
else:
    _register_reward = _dressage_register_reward


def _metadata(sample: Any) -> dict[str, Any] | None:
    metadata = getattr(sample, "metadata", None)
    if metadata is None and isinstance(sample, dict):
        metadata = sample.get("metadata", sample)
    return metadata if isinstance(metadata, dict) else None


@_register_reward("environment")
def compute_environment_reward(sample: Any, **_kwargs: Any) -> float:
    """Return the terminal reward emitted by the selected environment."""
    metadata = _metadata(sample)
    if metadata is None:
        return 0.0
    value = metadata.get("environment_reward", metadata.get("vita_reward", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("environment_reward must be numeric") from exc


@_register_reward("vita")
def compute_reward(sample: Any, **_kwargs: Any) -> float:
    """Backward-compatible alias for ``environment`` reward."""
    return compute_environment_reward(sample)


def _compute_smoke_dense_reward(sample: Any) -> float:
    """Terminal reward plus a small trajectory-progress signal for smoke runs.

    The production ``vita`` reward remains terminal-only. This separate,
    opt-in reward prevents an all-zero exploratory group from turning an
    end-to-end GRPO validation into a no-op before task-specific shaping has
    been selected.
    """
    metadata = _metadata(sample)
    if metadata is None:
        return 0.0
    terminal = compute_environment_reward(sample)
    try:
        turns = max(0, int(metadata.get("environment_num_agent_turns", metadata.get("vita_num_agent_turns", 0))))
    except (TypeError, ValueError) as exc:
        raise ValueError("environment_num_agent_turns must be an integer") from exc
    messages = metadata.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    assistant_characters = sum(
        len(str(message.get("content", "")))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    )
    # In a segmented Dressage trajectory, the reward callback sees only the
    # final (anchor) segment. That segment can have zero trainable tokens even
    # though the complete trajectory contains varied assistant generations.
    # The proxy preserves its full messages list on the anchor, so use the
    # observed assistant-content length as a tiny smoke-only tie breaker.
    return (
        terminal
        + min(turns, 300) / 300.0
        + min(assistant_characters, 100_000) / 10_000_000.0
    )


@_register_reward("environment_smoke_dense")
def compute_environment_smoke_dense_reward(sample: Any, **_kwargs: Any) -> float:
    """Generic smoke-only reward for every environment backend."""
    return _compute_smoke_dense_reward(sample)


@_register_reward("vita_smoke_dense")
def compute_smoke_dense_reward(sample: Any, **_kwargs: Any) -> float:
    """Backward-compatible alias for the generic smoke-only reward."""
    return _compute_smoke_dense_reward(sample)
