# vita-rl

Experiment and method code for tool-use post-training across interchangeable
environments.

## Architecture

- **vita-mini** is the first self-contained deterministic environment under
  `external/`; it requires neither GPT user simulation nor LLM evaluation.
- **memory_1** is a standalone Gymnasium-style delayed-recall environment
  under `external/`, with `reset`/`step` rather than a tool-use transcript.
- **VitaBench** remains an optional legacy benchmark backend at
  `external/vitabench`.
- **SGLang** is the inference backend serving the local Qwen model on
  `http://127.0.0.1:30000/v1/chat/completions`.
- **Dressage/slime** are later RL infrastructure. They are external runtime
  dependencies, not part of this repository.
- **vita-rl** contains shared harnesses, the environment runtime, Dressage
  adapter, deterministic rewards, and reproducible launch scripts.

GitHub (`Matrix9494/vita-rl`) is the source of truth. GiveMeANode node
`vita-dev` is the primary development and experiment machine; this checkout is
for initial setup, review, and occasional local edits.

## Standard VitaBench setup

The baseline uses:

- agent: local Qwen3.5-4B at `/workspace/models/Qwen3.5-4B`;
- user simulator: `gpt-4.1` through OpenRouter;
- evaluator: `gpt-4.1` through OpenRouter;
- VitaBench virtual environment: `/workspace/venvs/vita`;
- SGLang: a running local endpoint on port `30000`.

Run the required one-task validation from `vita-dev` with:

```bash
./scripts/smoke_test.sh
```

The script runs delivery task `10711001` once with a maximum of 300 steps.
It expects the VitaBench checkout, model, venv, and SGLang endpoint at the
paths above. Use `VITA_ROOT`, `VITA_VENV`, `QWEN_MODEL`, or
`SGLANG_BASE_URL` to override them when appropriate.
Set `SMOKE_OUTPUT` when a stable VitaBench save-to name is preferred.

OpenRouter credentials are injected externally through the GiveMeANode secret
named `openrouter` (as `OPENROUTER_API_KEY`). They are never written to this
repository. The launch wrapper creates and removes a temporary runtime model
configuration when that environment variable is present. The default path uses
a short-lived localhost proxy so the real key stays only in the proxy process;
VitaBench receives a harmless placeholder header and cannot serialize the key.
Alternatively,
`VITA_MODEL_CONFIG_PATH` may point to an already prepared external config.

## Baseline runner

For a general run, use flags such as:

```bash
./scripts/run_baseline.sh \
  --domain delivery \
  --task-id 10711001 \
  --agent-llm /workspace/models/Qwen3.5-4B \
  --user-llm gpt-4.1 \
  --evaluator-llm gpt-4.1 \
  --max-steps 300 \
  --output baseline_delivery
```

Generated simulations belong to the external VitaBench `data/` directory and
are ignored by Git. Large models, checkpoints, trajectories, benchmark
outputs, node-specific configurations, and secrets must not be committed.

## Development workflow

1. Pull the latest `main` on `vita-dev` in `/workspace/projects/vita-rl`.
2. Make changes there with Luna/Codex.
3. Run focused tests or experiments and inspect `git diff`.
4. Commit only reproducible source, config, and script changes.
5. Push tested changes to GitHub.
6. Pull locally only when review or local editing is useful.

Keep the upstream VitaBench and Dressage checkouts separate. Future RL
integration should use adapters in this repository and should not modify those
upstream projects unless a separate, deliberate change is required.

## Environment runtime

The root-owned runtime is environment-neutral. `vita-mini` is a deterministic
tool environment and `vitabench` is a legacy backend; both are selected by an
`environment` field in the same Dressage episode API. New deterministic
environments register the `reset/openai_tools/call_tool/evaluate` contract in
`vita_rl.environments` and run through the existing harnesses unchanged.

For Dressage/Vessl, prompt metadata uses the same neutral fields:

```json
{
  "environment": "vita-mini",
  "task_id": "delivery_revision",
  "agent_model": "proxy-model",
  "environment_args": {"harness": "vita_rl_standard"}
}
```

Start the runtime with `python -m vita_rl.runtime_server` and point Dressage
at it with `ENVIRONMENT_RUNTIME_URL`. The Vessl smoke launcher is
`scripts/run_environment_grpo_smoke.sh`; set `ENVIRONMENT=vita-mini` to avoid
the VitaBench-only OpenRouter user/evaluator relay.

## Procedural vita-mini evaluation

To run five independently generated, replayable delivery environments with the
summary harness retaining the last three completed interaction turns (`k=3`):

```bash
PYTHONPATH=src:external/vita-mini/src VITA_RL_PROTOCOL=mini \
  .venv-eval/bin/python -m vita_rl.environment_runner \
  --environment vita-mini --num-environments 5 --generation-seed 20260908 \
  --harness vita_rl_summary --summary-window-size 3 --max-steps 100 \
  --max-concurrency 30 \
  --model /u/dz13/vita-rl/models/Qwen3.5-4B \
  --output outputs/vita_mini_summary_k3_eval.json
```

The output records its generation seed and every `generated:<seed>` task ID;
pass any one of those IDs with `--task-id` to reproduce an individual episode.
