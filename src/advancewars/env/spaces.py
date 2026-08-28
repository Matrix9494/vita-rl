"""Small space fallback used when Gymnasium is not installed."""

from __future__ import annotations

import random


class Discrete:
    def __init__(self, n: int):
        self.n = int(n)

    def sample(self) -> int:
        return random.randrange(self.n)

    def contains(self, value: object) -> bool:
        return isinstance(value, int) and 0 <= value < self.n

    def __repr__(self) -> str:
        return f"Discrete({self.n})"


class MultiDiscrete:
    def __init__(self, nvec):
        self.nvec = tuple(int(value) for value in nvec)

    def sample(self):
        return tuple(random.randrange(limit) for limit in self.nvec)

    def contains(self, value: object) -> bool:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            return False
        if len(value) != len(self.nvec):
            return False
        return all(
            isinstance(item, int) and 0 <= item < limit
            for item, limit in zip(value, self.nvec, strict=True)
        )

    def __repr__(self) -> str:
        return f"MultiDiscrete({self.nvec})"
