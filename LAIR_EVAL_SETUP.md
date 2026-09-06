# LAIR evaluation setup

## Scope and locations

- This is an evaluation-only setup. It does not install Dressage, slime, VERL,
  Megatron, DeepSpeed, or other training dependencies.
- Activate with `source /u/dz13/vita-rl/.venv-eval/bin/activate`. The
  project-local path is a symlink to `/data/user/dz13/vita-rl/.venv-eval` so
  the environment does not consume the constrained home quota.
- The existing local checkpoint is
  `/u/dz13/vita-rl/models/Qwen3.5-4B`; no model download was performed.
- VitaBench is the editable checkout at `external/vitabench` commit
  `742e240855bf8686a0842360749d5ea970ea3987`.

## Working CUDA-12.8 inference stack

The compatible stack is installed with native CUDA-12.8 PyTorch wheels:

- PyTorch `2.9.1+cu128` (`torch.version.cuda == 12.8`)
- SGLang `0.5.10.post1`
- `sglang-kernel 0.4.1`
- `flashinfer-python` and `flashinfer-cubin` `0.6.7.post3`
- Transformers `5.3.0`, OpenAI Python client `2.6.1`, and LiteLLM `1.65.0`

`pip check` passes. SGLang 0.5.10.post1's wheel metadata requests
`torch==2.9.1` but does not require CUDA 13; the official
`torch==2.9.1+cu128`, `torchaudio==2.9.1+cu128`, and
`torchvision==0.24.1+cu128` wheels satisfy that requirement without replacing
the CUDA-12.8 runtime.

SGLang 0.5.10.post1 contains `sglang/srt/models/qwen3_5.py` and
`sglang/srt/configs/qwen3_5.py`. The local checkpoint resolves through
Transformers as `model_type=qwen3_5`.

## Qwen3.5 parser and thinking configuration

The relevant VESSL reference is its live SGLang `0.5.15.post1` launch:

```bash
python -m sglang.launch_server --model-path /root/models/Qwen3.5-4B \
  --host 127.0.0.1 --port 30000 --mem-fraction-static 0.32
```

Its startup log auto-detected `reasoning_parser=qwen3` and
`tool_call_parser=qwen3_coder` from the Qwen3.5 chat template. However, those
auto-detected values were not reflected in the VESSL process arguments, and a
direct VESSL request returned XML tool text and `<think>` text in `content`
with `tool_calls` and `reasoning_content` both null. Therefore LAIR pins the
same detected parser names explicitly rather than relying on that auto-detect
path.

The final LAIR server command is:

```bash
sglang serve --model-path /u/dz13/vita-rl/models/Qwen3.5-4B \
  --served-model-name qwen35-4b-local --host 127.0.0.1 --port 30000 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

`configs/eval/local_sglang.yaml` retains the local Qwen agent sampling
parameters:

```yaml
temperature: 1.0
top_p: 0.95
top_k: 20
presence_penalty: 1.5
repetition_penalty: 1.0
n: 1
max_tokens: 8192
max_input_tokens: 65536
```

## VESSL-aligned role configuration

The LAIR launchers now use the same role split as the VESSL baseline:

- agent: local `qwen35-4b-local` served by SGLang;
- user simulator: `gpt-4.1` through OpenRouter;
- evaluator: `gpt-4.1` through OpenRouter.

Each launcher starts `scripts/openrouter_proxy.py` on a short-lived localhost
port. The job must receive `OPENROUTER_API_KEY`; the relay inherits it and the
launcher immediately unsets it before starting VitaBench. The generated
per-job model configuration contains only the localhost relay and the inert
`Bearer vita-rl-local-proxy` header, never the API key. It is removed during
cleanup. The Qwen request-side `max_input_tokens: 65536` is scoped only to the
local Qwen model, so the local tokenizer is not incorrectly applied to GPT
requests.

The Qwen model configuration defaults to
`chat_template_kwargs.enable_thinking: false`. In
`external/vitabench/src/vita/utils/llm_utils.py`, Qwen template thinking is
then set from VitaBench's `enable_think` argument on each Qwen request:

- Agent: configurable through `VITA_AGENT_THINKING=true|false` in the
  one-task script. The 40-task script deliberately enables it.
- User simulator and evaluator: GPT-4.1 provider defaults; neither receives a
  Qwen `chat_template_kwargs` field.

The same adapter enforces the 65,536-token request-side input cap using the
local checkpoint template. When replaying an OpenAI assistant tool call, it
normalizes JSON-string arguments only in its tokenizer copy; the actual HTTP
history remains standards-compliant OpenAI tool-call data. The cap does not
change SGLang's native 262,144-token model context.

## Verified one-task tool-use baseline

On LAIR's H100 NVL (CUDA 12.8), job `93861` ran the final command above and
completed normally. Before launching VitaBench it ran
`scripts/lair/qwen35_openai_smoke.py`:

- `tool_choice=auto` produced one real structured `message.tool_calls` entry
  for `echo({"text": "ping"})`, with empty normal content (not XML tool text).
- Replaying the standard assistant-tool-call plus tool-result history returned
  final content `ping`.
- With thinking enabled, `reasoning_content` was populated and final content
  was `56`, with no raw `<think>...</think>` tags.

The machine-readable direct-smoke result is
`outputs/sglang_qwen35_smoke-93861.json`.

The genuine VitaBench task `10711001` then completed in 47.44 seconds with 56
agent steps, 15 structured assistant tool calls, and 15 corresponding Python
tool results. It had zero user-simulator reasoning leaks, did not hit the
8,192-token generation cap, and terminated at `agent_stop` with reward `0.0`.
That reward is a model/task outcome, not an infrastructure failure. Its result
and summary are `outputs/vita_single-93861.json` and
`outputs/vita_diagnostic-93861.json`. This historical tool-use smoke predates
the VESSL-aligned GPT-4.1 user/evaluator configuration; it verifies the local
Qwen/SGLang agent path only.

For a future one-task check, submit:

```bash
cd /u/dz13/vita-rl
OPENROUTER_API_KEY='injected-secret' sbatch scripts/lair/vita_single.sbatch
```

It defaults to agent thinking off for the simplest tool-use baseline. Set
`VITA_AGENT_THINKING=true` only when deliberately evaluating agent reasoning.
Do not submit another multi-task batch until this launcher remains green for
the selected model and configuration. Never place the actual key in a script,
configuration file, commit, or Slurm output.
