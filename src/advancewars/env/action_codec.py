"""Encode legal engine actions into a fixed discrete action space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from advancewars.engine.actions import Action


@dataclass
class ActionCodec:
    max_actions: int = 4096

    def encode_legal(self, actions: list[Action]) -> tuple[np.ndarray, dict[int, Action]]:
        if len(actions) > self.max_actions:
            raise ValueError(
                f"Legal action count {len(actions)} exceeds codec size {self.max_actions}"
            )
        mask = np.zeros(self.max_actions, dtype=np.int8)
        mapping: dict[int, Action] = {}
        for index, action in enumerate(actions):
            mask[index] = 1
            mapping[index] = action
        return mask, mapping
