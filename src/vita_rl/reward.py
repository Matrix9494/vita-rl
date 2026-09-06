"""Minimal Vita reward bridge for slime/Dressage samples."""
from typing import Any
def compute_reward(sample: Any) -> float:
    metadata = getattr(sample, "metadata", None)
    if metadata is None and isinstance(sample, dict): metadata = sample.get("metadata", sample)
    if not isinstance(metadata, dict): return 0.0
    try: return float(metadata.get("vita_reward", 0.0))
    except (TypeError, ValueError) as exc: raise ValueError("vita_reward must be numeric") from exc
