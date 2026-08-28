# advancewars

Advance Wars-style simulator for multi-agent reinforcement learning.

The first target is a deterministic, testable Python rules engine wrapped by a
PettingZoo-style environment. Advance Wars is a sequential turn-based game, so
the primary public environment API should be PettingZoo AEC. A parallel API can
be added later only as a compatibility wrapper if a downstream learner needs it.

See [docs/architecture.md](docs/architecture.md) for the proposed file structure
and public API plan.

See [docs/plan.md](docs/plan.md) for the implementation roadmap.

See [docs/status.md](docs/status.md) for current implementation status.

See [docs/configuration.md](docs/configuration.md) for selectable experiment
configs such as allowed unit pools.

See [docs/action_interface.md](docs/action_interface.md) for flat and structured
PettingZoo action formats.

See [docs/policies.md](docs/policies.md) for built-in baseline policies and
self-play scripts.
