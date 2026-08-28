# Advance Wars Simulator Architecture

This document is the initial implementation plan for `/u/dz13/advancewars`.
It defines the repository layout, module boundaries, and public APIs before the
first simulator code lands.

## Design Goals

- Build a rules-first simulator, not a UI-first clone.
- Keep the core engine independent from PettingZoo, Gymnasium, rendering, and
  training code.
- Make every rule deterministic and unit-testable: movement, combat, capture,
  income, production, fog of war, victory, and turn order.
- Use data files for units, terrain, CO/rule variants, and maps so the rule set
  can follow the DefendPeace reference implementation.
- Expose a PettingZoo AEC environment as the main RL interface because Advance
  Wars is sequential and turn based.

## Proposed File Structure

```text
advancewars/
├── README.md
├── pyproject.toml
├── docs/
│   ├── architecture.md
│   ├── rules.md
│   └── action_space.md
├── src/
│   └── advancewars/
│       ├── __init__.py
│       ├── advancewars_v0.py
│       ├── env/
│       │   ├── __init__.py
│       │   ├── aec_env.py
│       │   ├── parallel_env.py
│       │   ├── observations.py
│       │   ├── action_codec.py
│       │   └── rewards.py
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── state.py
│       │   ├── game.py
│       │   ├── actions.py
│       │   ├── legal_actions.py
│       │   ├── movement.py
│       │   ├── combat.py
│       │   ├── capture.py
│       │   ├── production.py
│       │   ├── fog.py
│       │   ├── economy.py
│       │   ├── victory.py
│       │   └── rng.py
│       ├── data/
│       │   ├── rulesets/
│       │   │   ├── aw2.yaml
│       │   │   └── defendpeace_awbw.yaml
│       │   ├── units.yaml
│       │   ├── terrain.yaml
│       │   ├── movement.yaml
│       │   ├── damage_tables/
│       │   │   └── awbw.yaml
│       │   ├── commanders/
│       │   │   └── awbw/
│       │   └── maps/
│       │       ├── duel.yaml
│       │       └── triangle.yaml
│       ├── render/
│       │   ├── __init__.py
│       │   ├── ansi.py
│       │   └── rgb_array.py
│       └── utils/
│           ├── __init__.py
│           ├── defendpeace_map.py
│           ├── serialization.py
│           └── coordinates.py
├── tests/
│   ├── engine/
│   │   ├── test_movement.py
│   │   ├── test_combat.py
│   │   ├── test_capture.py
│   │   ├── test_economy.py
│   │   └── test_victory.py
│   ├── env/
│   │   ├── test_aec_api.py
│   │   ├── test_action_masks.py
│   │   └── test_observations.py
│   └── fixtures/
│       └── maps/
└── examples/
    ├── random_rollout.py
    ├── scripted_match.py
    └── train_stub.py
```

## Module Boundaries

`engine/` is the source of truth. It owns immutable-ish game state, validates
actions, applies state transitions, and reports terminal conditions. It should
not import PettingZoo, Gymnasium, NumPy spaces, pygame, or RL libraries.

`env/` adapts the engine to PettingZoo. It handles agent iteration, action
encoding, observation tensors, rewards, truncation, and action masks.

`data/` stores rule parameters. The engine loads these into typed config objects
at reset time. This is where DefendPeace-compatible tables should live.

`render/` is optional. Initial render modes should be `ansi` and `rgb_array`;
`human` can be added later.

`tests/` should start with engine tests. PettingZoo API tests come after the
core state machine can run a minimal match.

## Public Package API

The top-level Python package should expose only stable entry points:

```python
from advancewars import env, raw_env, parallel_env
from advancewars.engine import Game, GameState, Action, Ruleset
```

Recommended public imports:

```python
# PettingZoo-style constructors
advancewars.env(**kwargs)          # AEC env with wrappers
advancewars.raw_env(**kwargs)      # raw AEC env
advancewars.parallel_env(**kwargs) # optional compatibility wrapper

# Engine-level API for search/planning/testing
advancewars.engine.Game
advancewars.engine.GameState
advancewars.engine.Action
advancewars.engine.ActionType
advancewars.engine.Ruleset
advancewars.engine.load_map
advancewars.engine.load_ruleset
```

Everything else should be considered internal until a real use case needs it.

## PettingZoo-Style Environment API

Primary constructor:

```python
from advancewars import env

aw_env = env(
    map_name="duel",
    ruleset="defendpeace_awbw",
    players=2,
    max_turns=100,
    fog=False,
    render_mode=None,
    observation_mode="planes",
    reward_mode="win_loss",
)
```

AEC usage:

```python
aw_env.reset(seed=0)

for agent in aw_env.agent_iter():
    obs, reward, termination, truncation, info = aw_env.last()
    if termination or truncation:
        action = None
    else:
        mask = obs["action_mask"]
        action = policy(obs, mask)
    aw_env.step(action)
```

Expected PettingZoo methods and attributes:

```python
env.reset(seed=None, options=None) -> (observations, infos) | None
env.step(action) -> None
env.last(observe=True)
env.observe(agent)
env.render()
env.close()
env.action_space(agent)
env.observation_space(agent)
env.agents
env.possible_agents
env.agent_selection
env.rewards
env.terminations
env.truncations
env.infos
env.metadata
```

Because one Advance Wars player can move many units before ending the turn, the
AEC environment should keep `agent_selection` on the same player after ordinary
unit commands. It advances to the next player only after an `END_TURN` action or
terminal state.

## Engine API

The engine API should be usable without PettingZoo:

```python
from advancewars.engine import Game, Action

game = Game.from_map("duel", ruleset="defendpeace_awbw", seed=0)
state = game.reset()

while not state.done:
    player = state.current_player
    legal = game.legal_actions(player)
    action = choose_action(state, legal)
    state, event_log = game.step(action)
```

Core methods:

```python
Game.from_map(map_name, ruleset, seed=None, **options) -> Game
Game.reset(seed=None) -> GameState
Game.step(action: Action) -> tuple[GameState, list[Event]]
Game.legal_actions(player_id: int) -> list[Action]
Game.action_mask(player_id: int, codec: ActionCodec) -> np.ndarray
Game.observe(player_id: int, mode="planes") -> Observation
Game.clone() -> Game
Game.to_json() -> dict
Game.from_json(payload: dict) -> Game
```

State model:

```python
GameState(
    map: MapState,
    players: tuple[PlayerState, ...],
    units: dict[UnitId, UnitState],
    current_player: int,
    turn: int,
    phase: Phase,
    weather: Weather,
    done: bool,
    winner: int | None,
)
```

## Action Model

Engine actions should be explicit typed objects:

```python
ActionType = Enum(
    "MOVE",
    "ATTACK",
    "CAPTURE",
    "JOIN",
    "LOAD",
    "UNLOAD",
    "LAUNCH",
    "RESUPPLY",
    "REPAIR",
    "DELETE",
    "TRANSFORM",
    "CO_ABILITY",
    "SWAP_CO",
    "WAIT",
    "BUILD",
    "END_TURN",
)

Action(
    type: ActionType,
    unit_id: int | None = None,
    path: tuple[Coord, ...] = (),
    target: Coord | None = None,
    build_unit: str | None = None,
    metadata: dict = {},
)
```

PettingZoo actions should be encoded to an integer or compact `MultiDiscrete`
value through `ActionCodec`. The first implementation should prefer a fixed
`Discrete(n)` action space plus `action_mask`, because many RL libraries handle
that path best.

`info` should expose the decoded legal actions for debugging:

```python
info = {
    "legal_actions": list[Action],
    "current_player": int,
    "turn": int,
    "phase": str,
}
```

## Observation Model

Default observation should be a dictionary:

```python
observation = {
    "observation": np.ndarray,  # board planes, shape [C, H, W]
    "action_mask": np.ndarray,  # shape [action_space.n]
}
```

Initial board planes:

- terrain type one-hot
- terrain defense stars
- property owner
- unit owner
- unit type one-hot
- unit HP / fuel / ammo
- unit can-act flag
- current player plane
- visible tile mask when fog is enabled

Later optional observations:

- `"flat"` for simple baselines
- `"graph"` for GNN policies
- `"dict"` for model-based agents that want structured state

## Rewards

Initial reward modes:

- `win_loss`: `+1` winner, `-1` loser, `0` otherwise.
- `dense_basic`: terminal reward plus small shaping for capture, unit value
  destroyed, property income swing, and HQ pressure.
- `none`: all zero rewards for external evaluators.

Keep reward logic in `env/rewards.py`, not in `engine/`, so the simulator rules
stay independent from learning objectives.

## Rule Coverage Order

Recommended implementation order:

1. Map loading, coordinates, terrain, players, units.
2. Turn order, wait/end turn, funds/income.
3. Ground movement and occupancy.
4. Capture and HQ capture victory.
5. Direct combat damage tables.
6. Production from bases.
7. Fog of war and vision.
8. Indirect fire, transport load/unload, fuel/ammo.
9. CO powers and weather.
10. Full compatibility pass against DefendPeace reference rules.

## DefendPeace Reference Mapping

Reference repository:

```text
https://github.com/ThislsAUsername/DefendPeace
```

DefendPeace is a Java Advance Wars clone. It is not organized as flat rule
tables; the rules are expressed through Java classes and event lifecycles. Our
Python simulator should therefore use DefendPeace in two ways:

- Extract stable constants into YAML where possible.
- Build compatibility tests from small scripted states so Python behavior can be
  compared against the Java behavior for movement, combat, capture, production,
  fog, and turn transitions.

Important source files and their Python targets:

```text
DefendPeace source                                Python target
------------------------------------------------  -------------------------------
src/Engine/GameScenario.java                      data/rulesets/defendpeace_awbw.yaml
src/Engine/GameInstance.java                      engine/game.py, engine/state.py
src/Engine/GameAction.java                        engine/actions.py
src/Engine/UnitActionFactory.java                 engine/legal_actions.py
src/Engine/UnitActionLifecycles/*.java            engine/* rule transition modules
src/Engine/Combat/CombatEngine.java               engine/combat.py
src/Engine/Combat/StrikeParams.java               engine/combat.py
src/Terrain/TerrainType.java                      data/terrain.yaml
src/Terrain/Maps/MapReader.java                   utils/defendpeace_map.py
src/Units/AWBWUnits.java                          data/units.yaml
src/Units/AWBWWeapons.java                        data/damage_tables/awbw.yaml
src/Units/MoveTypes/*.java                        data/movement.yaml
src/CommandingOfficers/AWBW/*.java                data/commanders/awbw/*.yaml
res/map/**/*.map                                  data/maps/defendpeace/
```

Initial DefendPeace defaults to mirror:

- `GameScenario.DEFAULT_INCOME = 1000`
- `GameScenario.DEFAULT_STARTING_FUNDS = 0`
- `GameScenario.DEFAULT_UNIT_CAP = 50`
- default unit scheme is `AWBWUnits`
- map tile records are fixed-width text cells: optional owner prefix plus a
  two-letter terrain code, e.g. `SE`, `GR`, `1HQ`, `0FC`
- optional map units use `team, unit type, x, y`

For the first playable Python milestone, use the AWBW subset as the canonical
`defendpeace_awbw` ruleset. Other DefendPeace variants such as AW1/AW2/AW3/AW4
CO behavior can come after the core engine is stable.

## References Checked

- PettingZoo documents AEC for sequential turn-based environments and Parallel
  API for simultaneous-action environments:
  https://pettingzoo.farama.org/index.html
- PettingZoo custom environment tutorial suggests a minimal env repository
  layout and shows `ParallelEnv` skeletons, action masks, and tests:
  https://pettingzoo.farama.org/tutorials/custom_environment/1-project-structure/
- Advance Wars By Web is a useful public feature/rules reference for maps,
  multiplayer, COs, tools, and charts:
  https://awbw.amarriner.com/
- DefendPeace is the primary implementation reference for this simulator:
  https://github.com/ThislsAUsername/DefendPeace
- Commander Wars is an additional open-source Advance Wars-inspired project with
  modding and custom rule support:
  https://commander-wars.pages.dev/
