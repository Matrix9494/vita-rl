# vita-mini

`vita-mini` is a procedural, deterministic delivery environment for research
on tool-use RL, long-horizon memory, and state abstraction. It does not import
VitaBench and never uses GPT or any other LLM for its user or evaluation.

## Agent contract

An episode follows the normal OpenAI tool-use transcript:

```text
system + user message → assistant tool call → tool result
→ optional scripted user event → … → final answer → exact evaluation
```

The agent sees only user messages, tool schemas, and tool results. The
database, latent task constraints, unrevealed preferences, revision history,
and evaluator state remain hidden.

The delivery API exposed through `env.openai_tools()` is:

- `search_stores`, `search_products`, `get_store`, `get_product`
- `create_order`, `modify_order`, `cancel_order`, `pay_order`, `get_order`

`env.call_tool(name, arguments)` returns a JSON-safe `ToolResult`; invalid
calls are structured agent-visible errors rather than Python tracebacks.

## Tasks, revisions, and evaluation

`delivery_revision` starts with a request for cold brew at home. After a
successful product search, the scripted user reveals that caffeine and spice
are unacceptable. After order creation, they revise the address to the office
and request payment. Succeeding requires retaining and applying all final
constraints via `modify_order` and `pay_order`.

`env.next_user_event(agent_turn=..., tool_name=..., tool_success=...)` reveals
condition-triggered scripted messages exactly once. Event triggers include
agent turns, tools, successful tools, and order creation. The shared runner
calls this automatically; direct tests can call it themselves.

`env.evaluate()` is exact and deterministic. It reports strict success,
binary terminal reward, constraint score, tool validity, every constraint
result, and an executor-only final-state snapshot. Strict success requires one
non-cancelled final order to satisfy all active objective constraints.

## Procedural generation

`TaskGenerator(seed).generate(difficulty=DifficultyConfig(...))` creates a
reproducible delivery database, composite constraints, distractor stores and
products, a disclosure/revision script, and an oracle action plan. The
configuration records the requested number of constraints, objectives,
revisions, distractors, tools, and horizon/retention targets. Use
`vita_mini.oracle.verify_oracle_solution(task)` before accepting generated
tasks in a dataset pipeline.

## Running through the shared harnesses

There is no mini-specific runner. From the repository root, the generic
environment runner uses the existing standard, stateful, summary,
recent-turns, and state-delta harnesses:

```bash
PYTHONPATH=src:external/vita-mini/src \
VITA_RL_PROTOCOL=mini \
.venv-eval/bin/python -m vita_rl.environment_runner \
  --environment vita-mini \
  --task-id delivery_revision \
  --harness vita_rl_standard \
  --model /u/dz13/vita-rl/models/Qwen3.5-4B
```

The runner sends native OpenAI-style messages to the configured Qwen/SGLang
endpoint, stores messages, tool calls/results, scripted user events,
termination reason, final state, reward, and constraint-level evaluation.
For Dressage/Vessl, select the same environment in prompt metadata:

```json
{
  "environment": "vita-mini",
  "task_id": "delivery_revision",
  "environment_args": {"harness": "vita_rl_stateful"}
}
```

## Tests

```bash
PYTHONPATH=external/vita-mini/src python3 -m unittest discover -s external/vita-mini/tests -v
```
