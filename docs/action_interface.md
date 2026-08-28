# Action Interface

The default PettingZoo interface remains the original flat legal-action index:

```python
env = raw_env(action_mode="flat")
```

For training, the environment also supports a structured action interface:

```python
env = raw_env(action_mode="structured")
```

This is the current implementation of the "scheme B" action design: one
`env.step(...)` still submits one complete game action, but the action has
stable semantic fields rather than a state-local list index.

## Structured Tuple

The structured action is a `MultiDiscrete` tuple:

```text
[kind, source_y, source_x, dest_y, dest_x, target_y, target_x, payload]
```

Coordinate fields use `height`/`width` as the `None` sentinel. On `duel`,
the board is `5x7`, so `source=(5,7)` means no source coordinate.

Field meanings:

- `kind`: action type such as `MOVE`, `ATTACK`, `BUILD`, or `END_TURN`
- `source`: acting unit's current coordinate, if a unit acts
- `destination`: movement destination; for non-moving current actions this is
  usually the same as `source`
- `target`: attack/load/join/repair/launch/build target coordinate, if needed
- `payload`: build unit, power kind, cargo id, or none

Example initial `duel` structured actions:

```text
END_TURN:
  [15, 5, 7, 5, 7, 5, 7, 0]

WAIT infantry from (0,1) without moving:
  [13, 1, 0, 1, 0, 5, 7, 0]

MOVE infantry from (0,1) to (1,1):
  [0, 1, 0, 1, 1, 5, 7, 0]

CAPTURE after moving from (0,1) to (1,2):
  [2, 1, 0, 2, 1, 5, 7, 0]
```

## Observation Masks

Structured observations include the board tensor plus field-level masks:

```python
obs = {
    "observation": board_planes,
    "action_type_mask": ...,
    "source_mask": ...,
    "destination_mask": ...,
    "target_mask": ...,
    "payload_mask": ...,
}
```

These masks are marginal masks over legal complete actions. A serious policy can
use them directly as a baseline, or use `info["legal_structured_actions"]` for
exact legal tuple filtering.

## Debug Info

`info` contains both the structured tuples and readable semantic actions:

```python
info["legal_structured_actions"]
info["legal_semantic_actions"]
info["legal_actions"]
```

`legal_actions` is still the engine `Action` object list for debugging and
compatibility.

## Composite Unit Actions

Structured mode has a semantic execution path for core composite unit actions:

```text
source + destination + terminal action + target/payload
```

For example, `ATTACK` can move to `destination` and then attack `target` in the
same `env.step(...)`. The structured legal-action generator evaluates terminal
actions from each reachable destination, including wait, capture, attack,
load/unload, join, resupply, repair, launch, delete, and transform when the
underlying engine rules allow them.

The masks are currently marginal masks over complete legal structured actions.
Future training code can add conditional/autoregressive masks for stronger
policy heads.
