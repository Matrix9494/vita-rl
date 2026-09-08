"""Prompts for the separate delta-proposer and action-policy phases."""

from __future__ import annotations

import json
from typing import Any

from .schema import ALLOWED_OPS, canonical_schema_description


EDIT_SCHEMA = {
    "ops": [{
        "op": " | ".join(sorted(ALLOWED_OPS)),
        "path": "/entities/<id> | /goals/<id> | /constraints/<id> or a documented child path",
        "value": "required for set, append, extend, and set_status",
    }]
}


def state_update_messages(
    *,
    state: dict[str, Any],
    unresolved_evidence: list[dict[str, Any]],
    observation: dict[str, Any],
) -> tuple[str, str]:
    system = """You are an internal state-update component, not a user-facing agent.
Return ONLY one JSON object matching the delta schema. Do not answer the user,
call a tool, explain your choice, or regenerate the state. Propose only edits
justified by the newest observation or the supplied unresolved evidence. Do not
rewrite unrelated fields. Prefer {{"ops":[{{"op":"noop"}}]}} when no edit is
justified. A newer explicit user statement overrides older information. Do not
turn a soft preference into a hard constraint unless the observation explicitly
requires it. The database will reject paths and records outside this schema.

Canonical schema:
{schema}

Allowed operations: set, append, extend, delete, set_status, noop,
resolve_evidence. set may create a complete entity/goal/constraint record
only at a currently absent record path. Never set an existing whole
`/goals/<id>` or `/entities/<id>` record: use a nested path so accumulated
state is retained. Nested paths traverse normal object keys and list indices.

append adds EXACTLY ONE non-list element to an existing list. For example:
`{{"ops":[{{"op":"append","path":"/goals/g1/slots/items","value":{{"product_name":"..."}}}}]}}`.
Never pass a list to append. To add several elements, use extend with a list:
`{{"ops":[{{"op":"extend","path":"/goals/g1/slots/requirements/hard","value":["no fried food","no high-purine food"]}}]}}`.
Do not JSON-encode objects or lists inside strings. set preserves the existing
JSON type (list, object, string, number, or boolean). Use nested paths to
update an item, e.g.
`{{"ops":[{{"op":"set","path":"/goals/g1/slots/items/0/quantity","value":1}}]}}`.

Lightweight container rules: items is a list; requirements is an object and
requirements.hard/requirements.soft are lists; selected, transaction,
location, and timing are objects. entity IDs belong only in `/entities/<id>`;
an entity value is exactly type, name, attributes and must not contain `id`.
set_status targets `/goals/<id>` and its value MUST be a bare status string,
for example `"active"`, never `{{"status":"active"}}`.

For a rejected raw observation, unresolved evidence has an ID such as `e4`.
Only emit `{{"op":"resolve_evidence","evidence_id":"e4"}}` after you
have handled it in your own state update; the CPU never resolves evidence on
its own.""".format(schema=canonical_schema_description())
    user = "\n".join((
        "CURRENT CANONICAL STATE:",
        json.dumps(state, ensure_ascii=False, sort_keys=True),
        "UNRESOLVED EVIDENCE (may be empty):",
        json.dumps(unresolved_evidence, ensure_ascii=False, sort_keys=True),
        "NEWEST OBSERVATION:",
        json.dumps(observation, ensure_ascii=False, sort_keys=True),
        "REQUIRED DELTA SHAPE:",
        json.dumps(EDIT_SCHEMA, ensure_ascii=False),
    ))
    return system, user


def action_state_protocol(*, state: dict[str, Any], observation: dict[str, Any]) -> str:
    """Exact action-policy input: canonical state plus only the latest raw event."""
    payload = {
        "canonical_semantic_state": state,
        "latest_observation": observation,
    }
    return """\
## State-Delta Protocol
The canonical semantic state below was created only by accepted LLM/oracle
deltas. The CPU does not infer facts from user language or tool text. The raw
latest observation is explicit context rather than hidden state. Use it to
choose an action or propose later semantic updates. Do not assume a goal is
complete merely because a status says done; verify work through tools before
ending. A stop request is accepted only when every canonical goal is
done/cancelled with accepted provenance. Unresolved evidence is deliberately
excluded from this action-policy input.

<canonical_context>
{payload}
</canonical_context>
""".format(payload=json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
