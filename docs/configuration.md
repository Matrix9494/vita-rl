# Game Configuration

The engine supports small YAML experiment configs for selecting a subset of the
rules without editing engine code.

## Unit Pools

`enabled_units` controls which unit types can be produced or spawned by rules:

```yaml
name: infantry_only
enabled_units:
  - infantry
strict_units: false
```

Use `null` to allow the full ruleset unit roster:

```yaml
name: standard
enabled_units: null
strict_units: false
```

Built-in configs live under `src/advancewars/data/configs/`:

- `standard`: all units
- `infantry_only`: only infantry production/spawns
- `basic_ground`: infantry, mech, APC, recon, tank, and artillery
- `no_production`: no new unit production/spawns; initial map units remain

Example usage:

```python
from advancewars.engine import Game

game = Game.from_map("duel", config="basic_ground")
```

Inline overrides are also supported:

```python
game = Game.from_map("duel", enabled_units=["infantry", "tank"])
```

PettingZoo-style envs pass the same options through:

```python
from advancewars import raw_env

env = raw_env(config="infantry_only")
```

By default, maps may still contain pre-placed units outside the enabled pool.
Set `strict_units: true` to reject such maps during `Game.reset()`.
