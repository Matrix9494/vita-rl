"""Structured MultiDiscrete action codec for semantic actions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from advancewars.engine.actions import Action, ActionType
from advancewars.engine.coordinates import Coord
from advancewars.engine.data import UNITS
from advancewars.engine.semantic_actions import SemanticAction, action_to_semantic
from advancewars.engine.state import GameState


ACTION_KINDS: tuple[ActionType, ...] = tuple(ActionType)
UNIT_PAYLOADS: tuple[str, ...] = tuple(sorted(UNITS))
PAYLOAD_NONE = 0
PAYLOAD_POWER = 1
PAYLOAD_SUPER = 2
UNIT_PAYLOAD_OFFSET = 3
CARGO_PAYLOAD_OFFSET = UNIT_PAYLOAD_OFFSET + len(UNIT_PAYLOADS)
DEFAULT_MAX_UNIT_ID_PAYLOADS = 128


@dataclass(frozen=True)
class StructuredActionCodec:
    """Encode complete actions as stable fields.

    The tuple format is:

    ``[kind, source_y, source_x, dest_y, dest_x, target_y, target_x, payload]``.

    For coordinates, ``height``/``width`` are the none sentinels.
    """

    width: int
    height: int
    max_unit_id_payloads: int = DEFAULT_MAX_UNIT_ID_PAYLOADS

    @property
    def nvec(self) -> tuple[int, ...]:
        return (
            len(ACTION_KINDS),
            self.height + 1,
            self.width + 1,
            self.height + 1,
            self.width + 1,
            self.height + 1,
            self.width + 1,
            CARGO_PAYLOAD_OFFSET + self.max_unit_id_payloads,
        )

    def encode_legal(
        self,
        actions: list[Action],
        state: GameState,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[tuple[int, ...], Action],
        list[SemanticAction],
    ]:
        masks = self.empty_masks()
        mapping: dict[tuple[int, ...], Action] = {}
        semantic_actions: list[SemanticAction] = []
        for action in actions:
            semantic = action_to_semantic(action, state)
            encoded = self.encode_semantic(semantic)
            mapping[encoded] = action
            semantic_actions.append(semantic)
            self._mark_masks(masks, encoded)
        return masks, mapping, semantic_actions

    def encode_legal_semantic(
        self,
        actions: list[SemanticAction],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[tuple[int, ...], SemanticAction],
        list[SemanticAction],
    ]:
        masks = self.empty_masks()
        mapping: dict[tuple[int, ...], SemanticAction] = {}
        for action in actions:
            encoded = self.encode_semantic(action)
            mapping[encoded] = action
            self._mark_masks(masks, encoded)
        return masks, mapping, actions

    def empty_masks(self) -> dict[str, np.ndarray]:
        return {
            "action_type_mask": np.zeros((len(ACTION_KINDS),), dtype=np.int8),
            "source_mask": np.zeros((self.height + 1, self.width + 1), dtype=np.int8),
            "destination_mask": np.zeros(
                (self.height + 1, self.width + 1),
                dtype=np.int8,
            ),
            "target_mask": np.zeros((self.height + 1, self.width + 1), dtype=np.int8),
            "payload_mask": np.zeros((self.nvec[-1],), dtype=np.int8),
        }

    def encode_semantic(self, action: SemanticAction) -> tuple[int, ...]:
        return (
            ACTION_KINDS.index(action.kind),
            *self._encode_coord(action.source),
            *self._encode_coord(action.destination),
            *self._encode_coord(action.target),
            self._encode_payload(action.payload),
        )

    def decode_semantic(self, encoded: Sequence[int]) -> SemanticAction:
        values = self.normalize(encoded)
        return SemanticAction(
            kind=ACTION_KINDS[values[0]],
            source=self._decode_coord(values[1], values[2]),
            destination=self._decode_coord(values[3], values[4]),
            target=self._decode_coord(values[5], values[6]),
            payload=self._decode_payload(values[7]),
        )

    def normalize(self, encoded: Sequence[int] | np.ndarray) -> tuple[int, ...]:
        values = tuple(int(value) for value in encoded)
        if len(values) != len(self.nvec):
            raise ValueError(f"Structured action must have {len(self.nvec)} fields.")
        for value, limit in zip(values, self.nvec, strict=True):
            if value < 0 or value >= limit:
                raise ValueError(
                    f"Structured action value {value} outside [0,{limit})."
                )
        return values

    def _encode_coord(self, coord: Coord | None) -> tuple[int, int]:
        if coord is None:
            return self.height, self.width
        return coord.y, coord.x

    def _decode_coord(self, y: int, x: int) -> Coord | None:
        if y == self.height and x == self.width:
            return None
        return Coord(x, y)

    def _encode_payload(self, payload: str | int | None) -> int:
        if payload is None:
            return PAYLOAD_NONE
        if payload == "power":
            return PAYLOAD_POWER
        if payload == "super":
            return PAYLOAD_SUPER
        if isinstance(payload, str) and payload in UNIT_PAYLOADS:
            return UNIT_PAYLOAD_OFFSET + UNIT_PAYLOADS.index(payload)
        if isinstance(payload, int):
            value = CARGO_PAYLOAD_OFFSET + payload
            if value < self.nvec[-1]:
                return value
        raise ValueError(f"Unsupported structured action payload: {payload!r}")

    def _decode_payload(self, payload: int) -> str | int | None:
        if payload == PAYLOAD_NONE:
            return None
        if payload == PAYLOAD_POWER:
            return "power"
        if payload == PAYLOAD_SUPER:
            return "super"
        if UNIT_PAYLOAD_OFFSET <= payload < CARGO_PAYLOAD_OFFSET:
            return UNIT_PAYLOADS[payload - UNIT_PAYLOAD_OFFSET]
        return payload - CARGO_PAYLOAD_OFFSET

    @staticmethod
    def _mark_masks(masks: dict[str, np.ndarray], encoded: tuple[int, ...]) -> None:
        masks["action_type_mask"][encoded[0]] = 1
        masks["source_mask"][encoded[1], encoded[2]] = 1
        masks["destination_mask"][encoded[3], encoded[4]] = 1
        masks["target_mask"][encoded[5], encoded[6]] = 1
        masks["payload_mask"][encoded[7]] = 1


def semantic_to_dict(action: SemanticAction) -> dict[str, Any]:
    return {
        "kind": action.kind.value,
        "source": _coord_to_dict(action.source),
        "destination": _coord_to_dict(action.destination),
        "target": _coord_to_dict(action.target),
        "payload": action.payload,
        "metadata": dict(action.metadata),
    }


def _coord_to_dict(coord: Coord | None) -> dict[str, int] | None:
    if coord is None:
        return None
    return {"x": coord.x, "y": coord.y}
