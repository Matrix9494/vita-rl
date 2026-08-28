"""Coordinate helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Coord:
    x: int
    y: int

    def manhattan(self, other: Coord) -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def neighbors(self) -> tuple[Coord, Coord, Coord, Coord]:
        return (
            Coord(self.x + 1, self.y),
            Coord(self.x - 1, self.y),
            Coord(self.x, self.y + 1),
            Coord(self.x, self.y - 1),
        )
