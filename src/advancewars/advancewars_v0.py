"""Versioned PettingZoo-style entry points."""

from advancewars.env.aec_env import AdvanceWarsAECEnv


def raw_env(**kwargs):
    """Create the raw AEC environment."""
    return AdvanceWarsAECEnv(**kwargs)


def env(**kwargs):
    """Create the default wrapped AEC environment.

    The raw environment remains available through `raw_env()`. When PettingZoo
    is installed, this default constructor follows the usual wrapper stack for
    stdout capture, action bounds assertions, and call-order enforcement.
    """
    environment = raw_env(**kwargs)
    try:
        from pettingzoo.utils import wrappers
    except Exception:  # pragma: no cover - optional dependency fallback
        return environment
    if getattr(environment, "render_mode", None) == "human":
        environment = wrappers.CaptureStdoutWrapper(environment)
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment


def parallel_env(**kwargs):
    """Create a turn-based ParallelEnv wrapper when PettingZoo is installed.

    Advance Wars is naturally sequential, so AEC remains the canonical API. The
    parallel entry point delegates to PettingZoo's turn-based AEC conversion for
    learners that expect a ParallelEnv surface.
    """
    try:
        from pettingzoo.utils.conversions import turn_based_aec_to_parallel
    except Exception as exc:  # pragma: no cover - optional dependency fallback
        raise RuntimeError(
            "parallel_env requires PettingZoo. Install pettingzoo or use raw_env()."
        ) from exc
    return turn_based_aec_to_parallel(raw_env(**kwargs))
