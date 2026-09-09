# memory_1

`memory_1` is a small deterministic Gymnasium-style environment for delayed
recall. At reset, the environment shows a cue identifying one correct action.
The cue is then hidden for a configurable delay; the agent receives reward only
when it selects the remembered action at the final query.

It is deliberately independent of `vita-mini`: `memory_1` is a conventional
step-based RL environment, not a tool-use environment. It therefore lives in
`external/` alongside the other independently packaged environments and does
not use the shared tool-environment registry.

## API

```python
from memory_1 import Memory1Config, Memory1Env

env = Memory1Env(Memory1Config(num_items=4, delay_steps=3))
observation, info = env.reset(seed=7)
while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

The API follows Gymnasium:

- `reset(seed=None, options=None) -> (observation, info)`
- `step(action) -> (observation, reward, terminated, truncated, info)`
- `action_space` is `Discrete(num_items)`
- `observation_space` contains `phase`, `cue`, and `remaining_steps`

The target cue is visible only during the `cue` phase. The terminal `info`
contains `target_action` for analysis, but it is never exposed before the
episode ends.

Install it independently when Gymnasium integration is needed:

```bash
pip install -e external/memory_1
```

Then register `Memory1-v0` explicitly with Gymnasium if desired:

```python
from memory_1 import register_gymnasium

register_gymnasium()
```

## Tests

```bash
PYTHONPATH=external/memory_1/src python -m unittest discover -s external/memory_1/tests -v
```
