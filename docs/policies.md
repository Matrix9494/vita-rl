# Policies

## InfantryDecisionTreePolicy

`InfantryDecisionTreePolicy` is the first baseline meant to play an actual
infantry-only match rather than just smoke-test the API.

It is designed for:

```python
raw_env(action_mode="structured", config="infantry_only")
```

The policy scores legal semantic actions with a hand-written decision tree:

- attack valuable or threatening units
- capture HQ, enemy factories, and cities
- build infantry when funds allow
- move toward strategic targets
- use end turn only when no better action exists

It is deterministic. On a fixed map with no luck, repeated self-play rounds are
expected to produce the same result unless the map, seed-sensitive rules, or
policy tie-breaking are changed.

Run ten self-play rounds:

```bash
PYTHONPATH=src python3 examples/infantry_self_play.py \
  --rounds 10 \
  --output runs/infantry_self_play_10.json
```

The output records winner, step count, action counts, final board text, and final
state for each round.
