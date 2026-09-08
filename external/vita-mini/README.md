# vita-mini

`vita-mini` is a small, self-contained environment for training and evaluating
tool-using agents.  It deliberately does not import VitaBench, call an LLM for
user simulation, or use an LLM judge.  Every state transition and score is
deterministic and inspectable.

The first domain is a compact storefront.  An agent can inspect an account and
catalogue, manage a cart, provide a delivery address, submit an order, and
inspect or cancel its orders.

## Install and run

```bash
cd external/vita-mini
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

```python
from vita_mini import MiniEnvironment

env = MiniEnvironment()
observation = env.reset("buy_coffee")

# Supply this to an OpenAI-compatible model client's `tools=` parameter.
tools = env.openai_tools()

result = env.call_tool("search_catalog", {"query": "coffee"})
print(result.to_dict())
```

## Agent-facing tools

| Tool | Purpose |
| --- | --- |
| `get_account` | Read the active customer's account. |
| `update_account_profile` | Change the customer's display name or email. |
| `search_catalog` | Search the fixed product catalogue. |
| `get_product` | Read one product and its remaining inventory. |
| `get_cart` | Read the current cart and subtotal. |
| `add_to_cart` | Add an in-stock product to the cart. |
| `remove_from_cart` | Remove a line from the cart. |
| `set_delivery_address` | Set the address used by checkout. |
| `checkout` | Convert a non-empty cart into a submitted order. |
| `list_orders` | List this customer's orders. |
| `get_order` | Read a specific order. |
| `cancel_order` | Cancel a submitted order and restore inventory. |
| `get_current_time` | Read the environment's fixed logical time. |

`MiniEnvironment.openai_tools()` returns these tools in the Chat Completions
function-tool format.  `call_tool(name, arguments)` is the single execution
boundary: it validates arguments, catches expected tool errors, and returns a
JSON-safe `ToolResult` rather than leaking Python exceptions to an agent.

## Episodes, scripted users, and evaluation

`reset(task_id)` loads a built-in `MiniTask`, resets all mutable state, and
returns an observation containing the deterministic user request.  The
environment currently includes `buy_coffee` and `cancel_order` tasks.  A
task's user text is static data, not model-generated.

Use `env.evaluate()` after the episode.  The result checks declarative goal
conditions such as product quantity, submitted order status, and cancellation.
It returns a reward in `[0.0, 1.0]`, failed conditions, and an exact state
snapshot.  This makes the evaluator appropriate for RL reward computation and
unit testing.

The public `snapshot()` method is intended for reproducibility/debugging.  It
is not included in `openai_tools()`, so agents can interact only through the
documented business APIs.

## Running Qwen through vita-rl harnesses

From the repository root, use the environment-neutral rollout runner. It uses
the root-owned harness classes (`vita_rl_standard`, `vita_rl_stateful`,
`vita_rl_summary`, `vita_rl_recent_turns`, and `vita_rl_state_delta`) but owns
the episode loop locally.  There is no VitaBench import, GPT user, or LLM
evaluator in this path.

```bash
PYTHONPATH=src:external/vita-mini/src \
VITA_RL_PROTOCOL=mini \
.venv-eval/bin/python -m vita_rl.environment_runner \
  --environment vita-mini \
  --task-id buy_coffee \
  --harness vita_rl_standard \
  --model /u/dz13/vita-rl/models/Qwen3.5-4B
```

The runner expects an OpenAI-compatible endpoint at
`http://127.0.0.1:30000/v1/chat/completions`; override it with
`--base-url` or `ENVIRONMENT_BASE_URL`. It sends Qwen exactly the harness's
native transcript and the tool schemas, executes each returned tool call via
`MiniEnvironment`, then gives Qwen role-`tool` results. At termination it
calls `env.evaluate()` for the deterministic reward. Future custom
environments implement the same contract and register once with
`vita_rl.environments.tool_environment_registry`; they use this exact runner,
not an environment-specific copy.
