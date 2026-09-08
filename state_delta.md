# State-delta harness

`vita_rl_state_delta` keeps semantic ownership with the LLM/oracle updater:

```text
raw observation -> LLM/oracle delta -> CPU structural validation -> CPU apply -> canonical state
```

The CPU never parses user language or tool text into goals, dietary rules,
addresses, quantities, IDs, preferences, transactions, or completion. It only
checks JSON structure and applies accepted operations atomically. This harness
does not instantiate, update, render, or consult `TaskState`; the separate
`vita_rl_stateful` harness may use deterministic parsing.

Canonical state is `{"entities": {}, "goals": {}, "constraints": {}}`.
The authoritative schema definition is in `state_delta/schema.py`: an entity
at `/entities/<id>` is exactly `{"type", "name", "attributes"}`. Its ID is
only the map key/path—an embedded `id` is invalid. Goals are exactly
`{"type", "status", "slots"}` and constraints are exactly
`{"goal", "strength", "field", "operator", "value"}`.

## Operations and pointers

The exact operation set is `set`, `append`, `extend`, `delete`, `set_status`,
`noop`, and `resolve_evidence`.

- `set` may create a child object key when its parent exists. It creates a
  complete record only at an absent `/entities/<id>`, `/goals/<id>`, or
  `/constraints/<id>` path. Replacing an existing whole record is rejected.
- `append` appends exactly one non-list element to an existing list. It never
  initializes or flattens lists.
- `extend` requires both an existing list target and a list value, then
  concatenates it.
- `delete` removes an existing record, object key, or indexed list element.
- `set_status` targets `/goals/<id>`.
- `resolve_evidence` removes only the explicitly named unresolved evidence ID.

Pointers support object keys, list indices, and arbitrary nested objects below
`attributes` and `slots`. Thus
`/goals/g1/slots/items/0/quantity` is valid when every intermediate container
exists and the list index is in range. Indexing a non-list, traversing a
scalar, or using an out-of-range index is rejected.

For every existing concrete JSON value, `set` preserves its JSON type: list,
object, string, number, and boolean cannot be replaced by a different type.
`null` is handled conservatively as an unconstrained placeholder. The CPU
never parses a string as JSON. Common slot container paths are structurally
typed only: `items` is a list; `requirements` is an object with list-valued
`hard` and `soft`; `selected`, `transaction`, `location`, and `timing` are
objects. These checks impose no semantic content on their values.

## Evidence, prompts, and termination

Rejected proposals are atomic: state and provenance stay unchanged, while the
raw observation is retained as unresolved evidence with a stable `eN` ID. The
CPU removes evidence only after a valid `resolve_evidence` delta. It does not
guess that a later update incorporated the observation.

The updater input is exactly canonical state, newest raw observation, and
unresolved evidence. The action input is exactly canonical state plus newest
raw observation; unresolved evidence is intentionally absent. Transition
traces log both input payloads, raw model delta, validation result, canonical
state, provenance, unresolved evidence, action, and termination decision.

Stop is mechanical: canonical goals must exist, all must be terminal, and
their terminal status needs accepted tool provenance (`done`) or explicit-user
provenance (`cancelled`). Unresolved evidence alone does not deadlock stop; a
stop with remaining evidence is accepted when the other gates pass and is
logged with `unresolved_evidence_warning`.

Use `VITA_STATE_DELTA_UPDATER=llm|noop|replay`; replay also requires
`VITA_STATE_DELTA_REPLAY_PATH`. Set
`VITA_STATE_DELTA_TRACE_PATH=/path/trace.jsonl` to persist transitions.
