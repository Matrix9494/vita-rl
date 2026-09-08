"""Pure CPU application of structurally validated state-delta proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .schema import clone_state
from .validator import (
    DeltaValidationError,
    ValidatedOp,
    apply_validated_state_op,
    validate_ops,
)


@dataclass
class StateUpdateResult:
    state: dict[str, Any]
    metadata: dict[str, dict[str, Any]]
    accepted_ops: list[dict[str, Any]]
    rejected_ops: list[dict[str, Any]]
    unresolved_evidence: list[dict[str, Any]]
    error: str | None = None


def _next_evidence_id(evidence: list[dict[str, Any]]) -> str:
    numbers = [
        int(item["id"][1:]) for item in evidence
        if isinstance(item.get("id"), str) and item["id"].startswith("e") and item["id"][1:].isdigit()
    ]
    return f"e{max(numbers, default=0) + 1}"


def _resolved(evidence: list[dict[str, Any]], evidence_id: str) -> list[dict[str, Any]]:
    return [item for item in evidence if item.get("id") != evidence_id]


def apply_state_delta(
    *,
    state: dict[str, Any],
    delta: Any,
    metadata: dict[str, dict[str, Any]],
    observation: dict[str, Any],
    turn_idx: int,
    unresolved_evidence: list[dict[str, Any]] | None = None,
) -> StateUpdateResult:
    """Atomically apply a proposal with no model/GPU or semantic parsing.

    A rejected transaction preserves state/provenance and records exactly one
    raw observation with a stable evidence ID. Only a valid LLM/oracle
    ``resolve_evidence`` operation may remove existing evidence.
    """
    original_state = clone_state(state)
    original_metadata = clone_state(metadata)
    original_evidence = clone_state(unresolved_evidence or [])
    try:
        operations = validate_ops(original_state, delta, unresolved_evidence=original_evidence)
        candidate_state = clone_state(original_state)
        candidate_metadata = clone_state(original_metadata)
        candidate_evidence = clone_state(original_evidence)
        for op in operations:
            if op.op == "resolve_evidence":
                candidate_evidence = _resolved(candidate_evidence, op.evidence_id or "")
                continue
            _apply(candidate_state, candidate_metadata, op, turn_idx, observation)
        return StateUpdateResult(
            state=candidate_state,
            metadata=candidate_metadata,
            accepted_ops=[_serialized_op(op) for op in operations],
            rejected_ops=[],
            unresolved_evidence=candidate_evidence,
        )
    except (DeltaValidationError, KeyError, TypeError, ValueError) as exc:
        evidence = {
            "id": _next_evidence_id(original_evidence),
            "turn": turn_idx,
            "source": observation.get("kind", "unknown"),
            "observation": clone_state(observation),
            "reason": str(exc),
        }
        return StateUpdateResult(
            state=original_state,
            metadata=original_metadata,
            accepted_ops=[],
            rejected_ops=[{"delta": delta, "reason": str(exc)}],
            unresolved_evidence=[*original_evidence, evidence],
            error=str(exc),
        )


def _serialized_op(op: ValidatedOp) -> dict[str, Any]:
    result = asdict(op)
    return {key: value for key, value in result.items() if value is not None and not (key == "path" and not value)}


def _apply(
    state: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
    op: ValidatedOp,
    turn_idx: int,
    observation: dict[str, Any],
) -> None:
    if op.op == "noop":
        return
    apply_validated_state_op(state, op)
    if op.op == "delete":
        _drop_metadata(metadata, op.path)
        return
    if op.op == "set":
        _drop_metadata(metadata, op.path)
    _provenance(metadata, op.path, turn_idx, observation)
    if op.op == "set_status":
        _provenance(metadata, op.path + "/status", turn_idx, observation)


def _provenance(metadata: dict[str, dict[str, Any]], path: str, turn_idx: int, observation: dict[str, Any]) -> None:
    source = "explicit_user" if observation.get("kind") == "user" else "tool_observation"
    metadata[path] = {"source_turn": turn_idx, "source_type": source}


def _drop_metadata(metadata: dict[str, dict[str, Any]], path: str) -> None:
    for key in [key for key in metadata if key == path or key.startswith(path + "/")]:
        del metadata[key]
