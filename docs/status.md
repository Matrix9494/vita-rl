# Implementation Status

Last updated after the fog-cover, combat, seeded-luck, fuel, weather, expanded
CO, basic tag-mode, special-action, missile-silo, cargo-launch, and config
slices.

## Done

- Project package skeleton
- Local packaging metadata
- DefendPeace-style map parser
- Terrain/unit/ruleset resource files
- Full DefendPeace AWBW unit roster in executable data
- DefendPeace AWBW 27x27 damage chart in executable data
- External DefendPeace `.map` path loading
- Minimal deterministic engine loop
- JSON-friendly `Game`/`GameState` serialization round-trip
- Basic movement with executable path validation and path-based fuel costs,
  capture, build, and combat
- Global `clear`/`rain`/`snow` weather state with DefendPeace-style movement
  costs, rain vision reduction, temporary weather duration/restoration,
  observation encoding, and serialization
- Commander state, charge, active-power tracking, and first-pass AWBW CO
  behavior for Andy, Max, Sami, Grit, and Colin
- Additional AWBW day-to-day CO behavior for Kanbei, Grimm, Jake, Eagle, Drake,
  and Jess, including their main attack/defense/move/range/cost/fuel/resupply
  hooks
- Additional AWBW day-to-day/basic power hooks for Olaf, Sasha, Rachel, Hachi,
  Sensei, Hawke, Adder, and Von Bolt, including income, repair, weather
  movement, city production, discounts, transport movement, and mass HP hooks
- Additional AWBW day-to-day/basic hooks for Sonja, Lash, Koal, Kindle, Javier,
  Flak, Jugger, Sturm, and Nell, including vision, terrain/road/urban attack,
  indirect defense, perfect movement, counter damage, and commander-specific
  luck ranges
- Drake `Tsunami`/`Typhoon` mass HP/fuel effects, including Typhoon temporary
  rain duration
- Selectable COP/SCOP action variants via `ActionType.CO_ABILITY` metadata,
  active power-kind serialization, and first-pass super-power behavior for
  Andy, Sami, Olaf, Hawke, Adder, Nell, Hachi, Kindle, Koal, Grit, Jake, Jess,
  Javier, Flak, and Jugger
- First-pass event-style CO powers: meteor targeting/damage for Rachel, Sturm,
  and Von Bolt, including Rachel `Covering Fire` preplanned target selection;
  Von Bolt stun with next-turn skip serialization; and Sensei infantry/mech
  city-spawn powers
- First-pass preemptive counter hook for Sonja's `Counter Break`, including
  counter-before-attack ordering and source-tree test configuration
- First-pass broader counter modifier support for Kanbei's AWBW `Samurai
  Spirit`, including corrected COP/SCOP attack-defense stats and +65 counter
  attack bonus
- Basic tag-mode commander teams with one-per-turn `SWAP_CO`, active commander
  serialization, and AEC env configuration
- Combat chooses the highest-base-damage legal weapon and ignores terrain
  defense for air-unit defenders
- Optional seeded combat luck: default combat remains no-luck for deterministic
  fixtures; `luck=True` enables DefendPeace-style +0 to +9 luck rolls with
  seed/reset control and JSON RNG-state serialization
- Owned-property repair/resupply on turn start
- Fuel rules: movement is limited by remaining fuel, movement consumes fuel,
  air/sea units burn idle fuel on turn start, friendly supply tiles prevent
  idle burn, and air/sea units die when fuel reaches zero
- Cargo load/unload for APC, T-Copter, Lander, Cruiser, Carrier, and BBoat
- Special actions: `JOIN`, `DELETE`, adjacent APC/BBoat `RESUPPLY`, and BBoat
  `REPAIR`
- Missile silo terrain parsing for `SR`/`BK` and AW-style `LAUNCH`: infantry or
  mech can fire a ready silo once, damaging all map units in radius 2 by 3 HP
  without killing them
- Carrier cargo `LAUNCH`: carried air units become ready on owner turn start,
  carriers can launch ready cargo to reachable destinations, and launched cargo
  remains able to act while the carrier's action is consumed
- Hidden-state transforms for Sub and Stealth
- Basic fog-of-war visibility, hidden-unit adjacency reveal, and fog-aware
  attack targeting
- Forest and reef cover in fog: ground/sea units in cover require adjacent
  vision to be observed or targeted
- AEC-style environment wrapper with action masks, including `env()` standard
  PettingZoo order/bounds wrappers and raw access via `raw_env()`
- Proper `max_turns` truncation state propagation through engine, JSON, and
  PettingZoo env APIs
- `parallel_env()` via PettingZoo's turn-based AEC-to-parallel conversion
- Reward modes: `win_loss`, `dense_basic`, and `none`; `dense_basic` includes
  combat damage, destroyed unit value, property income swing, capture, and HQ
  pressure shaping
- Official PettingZoo AEC and parallel API tests pass when optional dependencies
  are installed
- Smoke, parser, engine, and env tests
- Example random rollout and scripted match
- DefendPeace compatibility tests for roster, known damage-table values,
  combat/counter/capture/production fixtures, multiple `res/map/**/*.map`
  parser/reset fixtures, all current checkout `res/map/**/*.map` files, and
  `AW2_Spann_Island.map`
- Explicit DefendPeace unsupported/partial coverage list in
  `docs/unsupported.md`
- Experiment config layer with built-in YAML configs and selectable allowed unit
  pools for production/rule-spawn filtering
- Structured PettingZoo action mode with stable semantic fields
  (`kind/source/destination/target/payload`) alongside the original flat legal
  action index mode, including core move-then-wait/capture/attack composite
  actions
- Infantry-only decision-tree policy and a 10-round structured self-play script

## Partially Done

- Phase 1 data model: present, with full AWBW unit roster and expanded map
  code coverage for the current DefendPeace checkout's `res/map/**/*.map`
  files.
- Phase 2 engine loop: present, with basic state serialization; richer event
  coverage is not finished.
- Phase 3 movement: present for foot/tread/tires with executable path
  validation and path-based fuel costs, but not full DefendPeace movement
  parity. `clear`/`rain`/`snow` costs are now represented for the current AWBW
  movement types.
- Phase 4 combat: full AWBW base damage table, best-weapon selection, air-unit
  terrain-defense handling, DefendPeace-style integer truncation, and several
  AWBW CO attack/defense/range/cost/capture modifiers are present. Optional
  seeded luck is present, but the full commander roster and full `StrikeParams`
  parity are not finished.
- Phase 5 PettingZoo API: `raw_env()`, wrapped `env()`, and `parallel_env()`
  are present; official AEC `api_test` and `parallel_api_test` pass with the
  optional dependency installed; max-turn truncation is represented separately
  from victory termination; and the planned reward modes are implemented.
- Phase 6 fog compatibility: basic visibility, drive-by vision memory, hidden
  Sub/Stealth reveal, and forest/reef cover are present, but exact DefendPeace
  edge cases and broader fixture parity are not finished.
- Weather compatibility: global `clear`/`rain`/`snow`, temporary weather
  duration/restoration, and base fuel burn are present. Olaf/Drake
  movement-weather modifiers plus Olaf/Drake weather powers are present, but
  full forecast parity and sand/smoke variants are not finished.
- CO powers: commander state, charge, action legality, serialization,
  observation planes, and first-pass Andy/Max/Sami/Grit/Colin/Kanbei/Grimm/
  Jake/Eagle/Drake/Jess/Olaf/Sasha/Rachel/Hachi/Sensei/Hawke/Adder/Von Bolt
  /Sonja/Lash/Koal/Kindle/Javier/Flak/Jugger/Sturm/Nell behavior are present,
  and COP/SCOP action selection is now represented. Exact charge formulas,
  exact meteor tie-break parity, remaining counter-modifier edge cases, and
  remaining power-specific event parity are not finished.
- Tag mode: commander teams and active-CO swapping are present, but full Dual
  Strike rules, tag powers, team bonuses, and exact AWBW tag parity are not
  finished.
- Special actions: join/delete/manual resupply/BBoat repair, missile-silo
  launch, and carrier cargo launch are present. DefendPeace's exact one-action
  "launch plus cargo action" composition and edge-case parity are not finished.
- Configs: selectable unit pools are present for build options and rule spawns;
  broader rule toggles, per-building production policies, and per-player config
  overrides are not finished.
- Training action interface: structured action tuples, masks, and core
  move-plus-terminal composite actions are present. The masks are currently
  marginal masks; conditional/autoregressive masks are still a future training
  optimization.
- Baseline policies: `InfantryDecisionTreePolicy` can play infantry-only games
  and produce self-play summaries, but it is deterministic and still a hand
  written baseline rather than a strong search or learned bot.

## Not Done

- Broader DefendPeace compatibility fixture generation for rule edge cases
- Full fog-of-war compatibility fixtures and exact edge cases
- Full weather forecast parity, sand/smoke variants, and remaining weather/CO
  interaction parity
- Remaining AWBW CO roster and exact power/super-power parity
- Exact CO power/super-power event parity for remaining meteor/missile tie-breaks,
  remaining counter modifiers, remaining spawn/stun edge cases, and remaining
  multi-power edge cases
- Full Dual Strike tag power parity
- Exact DefendPeace composed cargo-launch action parity
