"""Baseline experiment metadata.

The executable baseline lives in ``scripts/run_baseline.sh`` so it can reuse
the external VitaBench installation without copying benchmark code here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSpec:
    """Small, dependency-free description of a benchmark run."""

    domain: str = "delivery"
    task_id: str = "10711001"
    num_trials: int = 1
    max_steps: int = 300
