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

The delivery API exposed through `env.openai_tools()` is fixed for the whole
episode and is available from the first agent turn, matching VitaBench's
exposure timing. It contains only vita-mini's nine stateful tools:

- `search_stores`, `search_products`, `get_store`, `get_product`
- `create_order`, `modify_order`, `cancel_order`, `pay_order`, `get_order`

The API names and arguments are part of vita-mini's task definition; no
VitaBench-only aliases or unrelated domain APIs are injected.

Every required action is discoverable from a user message, a schema, or an
earlier tool result. In particular, product search results provide the product
and store identifiers used by `create_order`; order creation returns the
identifier used by `modify_order`, `pay_order`, `cancel_order`, and
`get_order`. Each episode's `address` schema lists its valid saved locations,
and `delivery_time` requires `YYYY-MM-DD HH:MM:SS` strictly after the displayed
logical time. If no delivery time is requested, the agent should choose one.

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

The stable `delivery_revision` fixture is intentionally a single cold-brew
example. In contrast, procedural `generated:<seed>` tasks sample five product
domains (beverages, meals, and dietary substitutions), four saved delivery
locations, varied quantities and stores, and one of four action graphs:
cancel-and-replace, in-place line modification, direct ordering with a final
address change, or delivery rescheduling. The default suite has four
distractor stores, a same-name dietary-violating product decoy, and two
separated revisions in its standard non-replacement graphs. Seeds therefore
change the task semantics and required tool trajectory, not just prices or
identifiers.

The generator also samples three delayed-commitment memory families. They
disclose persistent requirements (quantity, dietary constraints, and/or a
budget) before any order exists; later events select the product and provide
delivery details without repeating the earlier requirements. The decisive
`create_order` must therefore combine information from several user events,
not recover it from an existing order. Each generated task records internal
`metadata.memory_requirements`, including disclosure/use event IDs,
persistence or supersession, the required `create_order` action, database
recoverability before use, and an event-level retention distance. This
metadata appears in executor snapshots as `task_metadata` but is never shown
to the agent.

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
