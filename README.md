# vita-rl

Experiment and method code for VitaBench post-training research.

## Architecture

- **VitaBench** is the external environment and benchmark. It remains at
  `/workspace/projects/vitabench` and is not vendored here.
- **SGLang** is the inference backend serving the local Qwen model on
  `http://127.0.0.1:30000/v1/chat/completions`.
- **Dressage/slime** are later RL infrastructure. They are external runtime
  dependencies, not part of this repository.
- **vita-rl** contains reproducible experiment configuration, launch scripts,
  and our future trajectory/state/reward/method code.

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

OpenRouter credentials are injected externally through the GiveMeANode secret
named `openrouter` (as `OPENROUTER_API_KEY`). They are never written to this
repository. The launch wrapper creates and removes a temporary runtime model
configuration when that environment variable is present. Alternatively,
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

## Placeholders

`trajectory.py`, `state.py`, `reward.py`, and `dressage_adapter.py` are
intentionally small placeholders. They define only the initial interfaces and
documentation needed to begin experiments; they do not implement RL yet.
