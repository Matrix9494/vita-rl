# Advance Wars Simulator Project Plan

This is the execution plan for bringing the simulator from an empty project to a
usable PettingZoo-compatible research environment.

## North Star

Build a deterministic Python Advance Wars-style simulator that can be used both
as:

- a clean engine API for search, planning, scripted agents, and debugging
- a PettingZoo AEC environment for multi-agent reinforcement learning

DefendPeace is the primary rule reference. The first target ruleset is the
AWBW-style subset exposed by DefendPeace's `AWBWUnits`, `AWBWWeapons`, terrain,
map reader, and default `GameScenario`.

## Phase 0: Project Skeleton

Goal: Make the repository installable and testable before writing rules.

Deliverables:

- `pyproject.toml`
- `src/advancewars/` package
- empty but importable `engine`, `env`, `data`, `render`, and `utils` modules
- `pytest` setup
- basic `ruff` or formatting config
- smoke test that imports `advancewars`

Done when:

- `python -m pytest` runs successfully
- `python -c "import advancewars"` works from the repo root

## Phase 1: Data Model and DefendPeace Map Parsing

Goal: Represent maps, terrain, players, properties, and units without gameplay.

Deliverables:

- typed coordinate helpers
- terrain definitions copied from DefendPeace `TerrainType.java`
- ruleset defaults:
  - income per city: `1000`
  - starting funds: `0`
  - unit cap: `50`
  - unit scheme: `defendpeace_awbw`
- a small native YAML map format
- parser for DefendPeace `.map` files
- at least one built-in tiny test map
- tests for terrain lookup, property ownership, and map dimensions

Done when:

- a DefendPeace `.map` file can be parsed into our `MapState`
- terrain/property ownership matches the source text map

## Phase 2: Minimal Engine Loop

Goal: Run a deterministic match with no combat yet.

Initial supported actions:

- `WAIT`
- `END_TURN`
- `BUILD`
- `CAPTURE`

Deliverables:

- `GameState`, `PlayerState`, `UnitState`, `MapState`
- `Game.reset()`
- `Game.step(action)`
- turn order
- funds and income
- property capture with 20 capture threshold
- HQ capture victory
- legal action generation for the supported actions
- event log for debugging state transitions

Done when:

- a scripted infantry can capture a neutral city
- income updates on the next turn
- capturing HQ ends the game with the correct winner

## Phase 3: Movement and Occupancy

Goal: Make units move like an Advance Wars board game.

Deliverables:

- movement types based on DefendPeace `Units/MoveTypes/*.java`
- path validation
- terrain movement costs
- allied/enemy occupancy blocking
- unit ready/can-act flags
- transport placeholders, even if full load/unload arrives later

Done when:

- legal move destinations are correct on the tiny test map
- units cannot pass through enemies or stop on illegal terrain
- wait after movement consumes the unit action

## Phase 4: Combat, Production, and Core AWBW Units

Goal: Support the core tactical loop.

Deliverables:

- AWBW unit definitions from DefendPeace `AWBWUnits.java`
- direct and indirect weapons from `AWBWWeapons.java`
- damage table extraction into `data/damage_tables/awbw.yaml`
- simplified no-luck combat first, then seeded luck
- counterattacks
- ammo consumption
- unit death
- production from factory/airport/seaport
- repair/resupply on owned properties

Done when:

- tank-vs-tank, infantry-vs-infantry, artillery range, and anti-air match
  expected DefendPeace damage in compatibility fixtures
- production spends funds and creates ready-state units correctly

## Phase 5: PettingZoo AEC Environment

Goal: Wrap the engine for multi-agent RL.

Deliverables:

- `advancewars.raw_env()`
- `advancewars.env()` with standard PettingZoo wrappers
- `advancewars.parallel_env()` only if easy via conversion
- `agent_iter()` behavior where the same player stays active until `END_TURN`
- `Discrete(n)` action space plus `action_mask`
- observation dict with board planes and mask
- reward modes:
  - `win_loss`
  - `dense_basic`
  - `none`
- `ansi` render mode
- PettingZoo API tests

Done when:

- a random legal-action rollout reaches truncation or termination without error
- `pettingzoo.test.api_test` passes for the AEC environment

## Phase 6: DefendPeace Compatibility Pass

Goal: Replace "close enough" with concrete rule parity where it matters.

Deliverables:

- compatibility fixtures generated from simple DefendPeace scenarios
- map parser tests against several `res/map/**/*.map` examples
- combat comparison cases covering:
  - no terrain
  - terrain defense
  - low HP attacker
  - counterattack
  - indirect fire
- capture/funds/production comparison cases
- fog of war comparison cases

Done when:

- every supported feature has at least one DefendPeace-derived regression test
- unsupported DefendPeace features are explicitly listed

## Deferred Features

These are important, but not blockers for the first usable RL environment:

- CO powers
- AW1/AW2/AW3/AW4 rule variants beyond the AWBW subset
- full Dual Strike tag power parity
- full fog-of-war stealth/submarine edge cases
- human GUI
- sprite/audio assets
- large map catalog packaging
- fast vectorized rollouts

## Immediate Next Coding Task

Phase 0 has started and the first executable slice is in place.

Completed in the first patch:

- `pyproject.toml` and compatibility `setup.py`
- `src/advancewars/` package
- public constructors `env()`, `raw_env()`, and `parallel_env()`
- engine API with `Game`, `Action`, `ActionType`, `load_map`, and `load_ruleset`
- config API with `GameConfig`, `load_config`, and allowed unit-pool filtering
- JSON-friendly `Game.to_json()` and `Game.from_json()` state round-trip
- DefendPeace-style `.map` parser
- initial built-in `duel` and `triangle` maps
- initial `defendpeace_awbw` data resources
- full AWBW unit roster and base damage chart in executable data
- external DefendPeace `.map` path loading
- minimal engine actions: `END_TURN`, `WAIT`, `MOVE`, `CAPTURE`, `BUILD`,
  `ATTACK`
- owned-property repair/resupply on turn start
- cargo load/unload
- hidden-state transforms for Sub and Stealth
- DefendPeace-style best-weapon selection for multi-weapon units
- DefendPeace-style air-unit terrain-defense handling and deterministic integer
  damage truncation
- optional seeded combat luck with reproducible env reset and JSON RNG-state
  round-trip
- global `clear`/`rain`/`snow` weather with DefendPeace-style movement costs,
  rain vision reduction, temporary weather duration/restoration, observation
  encoding, and serialization
- movement fuel limits/costs plus air/sea idle fuel burn and fuel-death events
- executable movement path validation, legal-action paths, and path-based
  movement fuel costs
- commander state, CO charge, `CO_ABILITY`, active-power observation planes,
  and first-pass Andy/Max/Sami/Grit/Colin AWBW behavior
- expanded day-to-day CO slice for Kanbei, Grimm, Jake, Eagle, Drake, and Jess
- expanded CO slice for Olaf, Sasha, Rachel, Hachi, Sensei, Hawke, Adder, and
  Von Bolt covering their day-to-day/basic power hooks
- expanded CO slice for Sonja, Lash, Koal, Kindle, Javier, Flak, Jugger, Sturm,
  and Nell covering additional terrain, road, urban, vision, luck, indirect
  defense, perfect-movement, and counter hooks
- Drake `Tsunami`/`Typhoon` mass HP/fuel effects and Typhoon temporary rain
- selectable COP/SCOP actions through `CO_ABILITY` metadata plus active
  power-kind serialization and initial super-power behavior for existing COs
- first-pass event-style CO powers for Rachel/Sturm/Von Bolt meteor damage,
  Rachel `Covering Fire` preplanned targeting, Von Bolt stun, and Sensei
  city-spawn powers
- first-pass Sonja `Counter Break` preemptive counter ordering plus pytest
  source-tree import guard
- first-pass Kanbei `Samurai Spirit` counter attack bonus and corrected AWBW
  COP/SCOP stat totals
- basic tag-mode commander teams with `SWAP_CO`
- special actions: `JOIN`, `DELETE`, adjacent APC/BBoat `RESUPPLY`, and BBoat
  `REPAIR`
- missile silo terrain parsing for `SR`/`BK` and AW-style `LAUNCH`
- carrier cargo `LAUNCH` with ready carried cargo and post-launch cargo action
  availability
- basic fog-of-war visibility, drive-by vision memory, hidden-unit adjacency
  reveal, and fog-aware attack targeting
- forest/reef cover handling for fog observations and attack targeting
- AEC-style environment with discrete action masks
- `env()` with PettingZoo order/bounds wrappers and `raw_env()` for unwrapped
  access
- `max_turns` truncation tracked separately from victory termination through
  engine state, JSON, and env APIs
- reward modes: `win_loss`, `dense_basic`, and `none`
- `dense_basic` shaping for combat damage, destroyed unit value, property
  income swing, capture, and HQ pressure
- official PettingZoo `api_test` coverage with and without fog when optional
  dependencies are present
- `parallel_env()` through PettingZoo's turn-based AEC-to-parallel conversion
  plus official `parallel_api_test` coverage
- `examples/random_rollout.py` and `examples/scripted_match.py`
- smoke, parser, engine, and env tests
- DefendPeace compatibility tests for roster, damage values,
  combat/counter/capture/production fixtures, multiple `res/map/**/*.map`
  parser/reset fixtures, all current checkout `res/map/**/*.map` files, and
  `AW2_Spann_Island.map`
- explicit unsupported/partial DefendPeace coverage list in
  `docs/unsupported.md`

Current verification:

```bash
python3 -m pytest -q
PYTHONPATH=src python3 examples/scripted_match.py
PYTHONPATH=src python3 examples/random_rollout.py
```

Next coding task:

1. Split `engine/game.py` into dedicated rule modules once behavior stabilizes.
2. Broaden DefendPeace-derived compatibility fixtures for fog and more combat
   edge cases.
3. Continue replacing simplified combat with closer DefendPeace `StrikeParams`
   parity, especially remaining CO attack/defense and special luck modifiers.
4. Expand fog-of-war parity for exact DefendPeace edge cases and broader
   fixture coverage.
5. Expand weather parity for forecast edge cases, CO immunities, and additional
   weather variants.
6. Continue expanding CO powers toward remaining AWBW commanders, exact SCOP
   event parity, meteor/missile tie-breaks, remaining counter modifiers,
   remaining spawn/stun edge cases, and exact modifier timing.
7. Expand tag mode toward Dual Strike/tag power parity and tighten the current
   carrier launch support toward DefendPeace's exact composed cargo-action
   parity.
