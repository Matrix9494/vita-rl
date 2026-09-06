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

## H100 validation and run command

On LAIR's H100 NVL (driver `570.195.03`, CUDA 12.8), PyTorch reports
`torch.cuda.is_available() == True`. The local SGLang server successfully
loads the checkpoint as `Qwen3_5ForConditionalGeneration` and serves on
`127.0.0.1:30000` with:

```bash
sglang serve --model-path /u/dz13/vita-rl/models/Qwen3.5-4B \
  --served-model-name qwen35-4b-local --host 127.0.0.1 --port 30000
```

The OpenAI-compatible `/v1/chat/completions` smoke request returned HTTP 200
with a non-empty completion.

For one genuine delivery task, submit:

```bash
cd /u/dz13/vita-rl
sbatch scripts/lair/vita_single.sbatch
```

The script starts that local server, verifies `/v1/models`, makes the smoke
request, and runs VitaBench task `10711001`. Results are written under
`outputs/vita_single-<Slurm-job-id>.json` and server logs under `logs/`.

Job `93838` completed task `10711001` successfully as a VitaBench simulation
in 15.63 minutes. Its evaluation reward was `0.0` because the simulated
conversation reached the configured 100-step limit (`termination_reason:
max_steps`); this is a task outcome, not an inference or CUDA failure.
