"""Structural validation and deterministic JSON-pointer mechanics.

No function here reads natural language or tool text. Validation applies a
proposal to a private preview so multi-op proposals remain transactional.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from numbers import Number
from typing import Any

from .schema import (
    ALLOWED_GOAL_STATUSES,
    ALLOWED_OPS,
    LEGAL_STATUS_TRANSITIONS,
    clone_state,
    goal_slot_expected_type,
)


@dataclass(frozen=True)
class ValidatedOp:
    op: str
    path: str = ""
    value: Any = None
    evidence_id: str | None = None


class DeltaValidationError(ValueError):
    pass


def _parts(path: Any) -> list[str]:
    if not isinstance(path, str) or not path.startswith("/"):
        raise DeltaValidationError("path must be a JSON pointer beginning with '/'")
    parts = path.split("/")[1:]
    if not parts or any(not part or "~" in part for part in parts):
        raise DeltaValidationError("path has empty or escaped segments")
    return parts


def _nonempty_string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DeltaValidationError(f"{label} must be a non-empty string")


def _json_compatible(value: Any, label: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DeltaValidationError(f"{label} must be JSON-compatible: {exc}") from exc


def _entity(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"type", "name", "attributes"}:
        raise DeltaValidationError("entity records require exactly type, name, attributes (no embedded id)")
    _nonempty_string(value["type"], "entity.type")
    _nonempty_string(value["name"], "entity.name")
    if not isinstance(value["attributes"], dict):
        raise DeltaValidationError("entity.attributes must be an object")
    _json_compatible(value, "entity record")


def _goal(value: Any, record_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DeltaValidationError("goal record must be an object")
    # Compatibility only for existing replay traces. Entity records do not
    # have this exception: their identity belongs exclusively in the path.
    if set(value) == {"id", "type", "status", "slots"}:
        if value["id"] != record_id:
            raise DeltaValidationError("goal value.id must exactly match its path ID")
        value = {key: value[key] for key in ("type", "status", "slots")}
    if set(value) != {"type", "status", "slots"}:
        raise DeltaValidationError("goal records require exactly type, status, slots")
    _nonempty_string(value["type"], "goal.type")
    if value["status"] not in ALLOWED_GOAL_STATUSES:
        raise DeltaValidationError("goal.status is invalid")
    if not isinstance(value["slots"], dict):
        raise DeltaValidationError("goal.slots must be an object")
    for key, expected in (("items", list), ("requirements", dict), ("selected", dict),
                          ("transaction", dict), ("location", dict), ("timing", dict)):
        if key in value["slots"] and not isinstance(value["slots"][key], expected):
            raise DeltaValidationError(f"goal.slots.{key} must be a {expected.__name__}")
    requirements = value["slots"].get("requirements")
    if isinstance(requirements, dict):
        for key in ("hard", "soft"):
            if key in requirements and not isinstance(requirements[key], list):
                raise DeltaValidationError(f"goal.slots.requirements.{key} must be a list")
    _json_compatible(value, "goal record")
    return value


def _constraint(value: Any, known_goals: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != {"goal", "strength", "field", "operator", "value"}:
        raise DeltaValidationError("constraint records require goal, strength, field, operator, value")
    if value["goal"] not in known_goals:
        raise DeltaValidationError("constraint references an unknown goal")
    if value["strength"] not in {"hard", "soft"}:
        raise DeltaValidationError("constraint.strength must be hard or soft")
    _nonempty_string(value["field"], "constraint.field")
    _nonempty_string(value["operator"], "constraint.operator")
    _json_compatible(value, "constraint record")


def validate_delta_shape(delta: Any) -> list[dict[str, Any]]:
    if not isinstance(delta, dict) or set(delta) != {"ops"} or not isinstance(delta["ops"], list):
        raise DeltaValidationError("delta must be exactly {'ops': [...]} ")
    if not delta["ops"]:
        raise DeltaValidationError("delta.ops must contain one explicit noop or edit")
    if len(delta["ops"]) > 32:
        raise DeltaValidationError("delta.ops exceeds the v1 limit of 32")
    if not all(isinstance(op, dict) for op in delta["ops"]):
        raise DeltaValidationError("every delta operation must be an object")
    return delta["ops"]


def _json_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, Number):
        return "number"
    raise DeltaValidationError("value must be JSON-compatible")


def _require_type_preserved(existing: Any, replacement: Any, path: str) -> None:
    """Forbid coercion; strings are never parsed as JSON."""
    old_kind = _json_kind(existing)
    new_kind = _json_kind(replacement)
    if old_kind != "null" and old_kind != new_kind:
        raise DeltaValidationError(
            f"set must preserve existing JSON type at {path}: {old_kind} -> {new_kind}"
        )


def _parse_index(segment: str, length: int) -> int:
    if not segment.isdigit():
        raise DeltaValidationError("list path segment must be a non-negative integer")
    index = int(segment)
    if index >= length:
        raise DeltaValidationError("list index is out of range")
    return index


def _get_at(state: dict[str, Any], parts: list[str]) -> Any:
    current: Any = state
    for segment in parts:
        if isinstance(current, dict):
            if segment not in current:
                raise DeltaValidationError("path does not exist")
            current = current[segment]
        elif isinstance(current, list):
            current = current[_parse_index(segment, len(current))]
        else:
            raise DeltaValidationError("cannot traverse into a scalar value")
    return current


def _parent_at(state: dict[str, Any], parts: list[str]) -> tuple[Any, str]:
    if len(parts) < 2:
        raise DeltaValidationError("cannot modify the canonical root")
    parent = _get_at(state, parts[:-1])
    if not isinstance(parent, (dict, list)):
        raise DeltaValidationError("cannot modify a field on a scalar value")
    return parent, parts[-1]


def _declared_slot_type(parts: list[str]) -> type | None:
    if len(parts) >= 4 and parts[0] == "goals" and parts[2] == "slots":
        return goal_slot_expected_type(parts[3:])
    return None


def _validate_generic_mutation(op: str, state: dict[str, Any], parts: list[str], value: Any) -> None:
    expected = _declared_slot_type(parts)
    if op == "set":
        parent, leaf = _parent_at(state, parts)
        if isinstance(parent, dict):
            if leaf in parent:
                _require_type_preserved(parent[leaf], value, "/" + "/".join(parts))
        else:
            index = _parse_index(leaf, len(parent))
            _require_type_preserved(parent[index], value, "/" + "/".join(parts))
        if expected is not None and not isinstance(value, expected):
            raise DeltaValidationError(
                f"declared slot container at /{'/'.join(parts)} must be a {expected.__name__}"
            )
        return
    target = _get_at(state, parts)
    if op == "append":
        if isinstance(value, list):
            raise DeltaValidationError("append value must be exactly one element, not a list")
        if not isinstance(target, list):
            raise DeltaValidationError("append target must already be a list")
        return
    if op == "extend":
        if not isinstance(target, list):
            raise DeltaValidationError("extend target must already be a list")
        if not isinstance(value, list):
            raise DeltaValidationError("extend value must be a list")
        return
    if op == "delete":
        _parent_at(state, parts)
        _get_at(state, parts)
        return
    raise DeltaValidationError(f"unsupported nested operation {op}")


def _validate_entity_op(op: str, state: dict[str, Any], parts: list[str], value: Any) -> None:
    entity_id = parts[1]
    exists = entity_id in state["entities"]
    if len(parts) == 2:
        if op == "set":
            if exists:
                raise DeltaValidationError("set may create /entities/<id> only when the entity does not exist")
            _entity(value)
            return
        if op == "delete" and exists:
            return
        raise DeltaValidationError("invalid entity operation/path")
    if not exists:
        raise DeltaValidationError("nested entity mutation requires an existing entity")
    if len(parts) == 3 and parts[2] in {"type", "name"} and op == "set":
        _nonempty_string(value, f"entity.{parts[2]}")
        _require_type_preserved(state["entities"][entity_id][parts[2]], value, "/" + "/".join(parts))
        return
    if parts[2] == "attributes":
        _validate_generic_mutation(op, state, parts, value)
        return
    raise DeltaValidationError("invalid entity operation/path")


def _validate_goal_op(op: str, state: dict[str, Any], parts: list[str], value: Any) -> Any:
    goal_id = parts[1]
    exists = goal_id in state["goals"]
    if len(parts) == 2:
        if op == "set":
            if exists:
                raise DeltaValidationError("set may create /goals/<id> only when the goal does not exist")
            return _goal(value, goal_id)
        if op == "set_status" and exists:
            if isinstance(value, dict):
                if set(value) != {"status"}:
                    raise DeltaValidationError("set_status object value may contain only status")
                value = value["status"]
            if value not in ALLOWED_GOAL_STATUSES:
                raise DeltaValidationError("invalid goal status")
            old_status = state["goals"][goal_id]["status"]
            if value not in LEGAL_STATUS_TRANSITIONS[old_status]:
                raise DeltaValidationError(f"illegal goal status transition {old_status!r} -> {value!r}")
            return value
        if op == "delete" and exists:
            if any(c["goal"] == goal_id for c in state["constraints"].values()):
                raise DeltaValidationError("cannot delete a goal still referenced by a constraint")
            return value
        raise DeltaValidationError("invalid goal operation/path")
    if not exists:
        raise DeltaValidationError("nested goal mutation requires an existing goal")
    if len(parts) == 3 and parts[2] == "type" and op == "set":
        _nonempty_string(value, "goal.type")
        _require_type_preserved(state["goals"][goal_id]["type"], value, "/" + "/".join(parts))
        return value
    if parts[2] == "slots":
        _validate_generic_mutation(op, state, parts, value)
        return value
    raise DeltaValidationError("invalid goal operation/path")


def _validate_constraint_op(op: str, state: dict[str, Any], parts: list[str], value: Any) -> None:
    constraint_id = parts[1]
    exists = constraint_id in state["constraints"]
    if len(parts) == 2:
        if op == "set":
            if exists:
                raise DeltaValidationError("set may create /constraints/<id> only when the constraint does not exist")
            _constraint(value, set(state["goals"]))
            return
        if op == "delete" and exists:
            return
        raise DeltaValidationError("invalid constraint operation/path")
    if not exists or len(parts) != 3 or op != "set":
        raise DeltaValidationError("invalid constraint operation/path")
    field = parts[2]
    if field not in {"goal", "strength", "field", "operator", "value"}:
        raise DeltaValidationError("invalid constraint operation/path")
    old = state["constraints"][constraint_id][field]
    _require_type_preserved(old, value, "/" + "/".join(parts))
    if field == "goal" and value not in state["goals"]:
        raise DeltaValidationError("constraint references an unknown goal")
    if field == "strength" and value not in {"hard", "soft"}:
        raise DeltaValidationError("constraint.strength must be hard or soft")
    if field in {"field", "operator"}:
        _nonempty_string(value, f"constraint.{field}")


def apply_validated_state_op(state: dict[str, Any], op: ValidatedOp) -> None:
    """Apply one already-validated non-evidence operation to ``state``."""
    if op.op in {"noop", "resolve_evidence"}:
        return
    parts = _parts(op.path)
    if op.op == "set_status":
        state["goals"][parts[1]]["status"] = clone_state(op.value)
        return
    if op.op == "set":
        if len(parts) == 2:
            state[parts[0]][parts[1]] = clone_state(op.value)
            return
        parent, leaf = _parent_at(state, parts)
        if isinstance(parent, dict):
            parent[leaf] = clone_state(op.value)
        else:
            parent[_parse_index(leaf, len(parent))] = clone_state(op.value)
        return
    if op.op == "delete":
        if len(parts) == 2:
            del state[parts[0]][parts[1]]
            return
        parent, leaf = _parent_at(state, parts)
        if isinstance(parent, dict):
            del parent[leaf]
        else:
            del parent[_parse_index(leaf, len(parent))]
        return
    target = _get_at(state, parts)
    if op.op == "append":
        target.append(clone_state(op.value))
        return
    if op.op == "extend":
        target.extend(clone_state(op.value))
        return
    raise DeltaValidationError(f"cannot apply {op.op}")


def validate_ops(
    state: dict[str, Any], delta: Any, *, unresolved_evidence: list[dict[str, Any]] | None = None,
) -> list[ValidatedOp]:
    """Validate a complete proposal against a preview state without mutation."""
    raw_ops = validate_delta_shape(delta)
    working = clone_state(state)
    evidence_ids = {item.get("id") for item in (unresolved_evidence or [])}
    validated: list[ValidatedOp] = []
    for raw in raw_ops:
        op = raw.get("op")
        if op not in ALLOWED_OPS:
            raise DeltaValidationError(f"unsupported operation: {op!r}")
        if op == "noop":
            if set(raw) != {"op"}:
                raise DeltaValidationError("noop may contain only op")
            validated_op = ValidatedOp(op="noop")
        elif op == "resolve_evidence":
            if set(raw) != {"op", "evidence_id"}:
                raise DeltaValidationError("resolve_evidence requires only op and evidence_id")
            evidence_id = raw["evidence_id"]
            _nonempty_string(evidence_id, "evidence_id")
            if evidence_id not in evidence_ids:
                raise DeltaValidationError("resolve_evidence references unknown evidence_id")
            evidence_ids.remove(evidence_id)
            validated_op = ValidatedOp(op=op, evidence_id=evidence_id)
        else:
            required = {"op", "path"} if op == "delete" else {"op", "path", "value"}
            if set(raw) != required:
                raise DeltaValidationError(f"{op} has missing or unexpected fields")
            parts = _parts(raw["path"])
            if parts[0] not in {"entities", "goals", "constraints"} or len(parts) < 2:
                raise DeltaValidationError("path must target an entity, goal, or constraint record")
            value = raw.get("value")
            if op != "delete":
                _json_compatible(value, f"{op}.value")
            if parts[0] == "entities":
                _validate_entity_op(op, working, parts, value)
            elif parts[0] == "goals":
                value = _validate_goal_op(op, working, parts, value)
            else:
                _validate_constraint_op(op, working, parts, value)
            validated_op = ValidatedOp(op=op, path=raw["path"], value=value)
            apply_validated_state_op(working, validated_op)
        validated.append(validated_op)
    return validated
