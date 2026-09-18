#!/usr/bin/env bash
# Run a full VitaBench upper-bound evaluation with OpenRouter's GPT-5.6 Terra.
# The OpenRouter key arrives only in this process environment from the submit
# wrapper and is retained solely by the localhost forwarding proxy.
set -euo pipefail

source /root/setup_env.sh

REPO="${VITA_EVAL_REPO:-${VITA_RL_ROOT:?VITA_RL_ROOT is required}}"
VITA_ROOT="${VITA_ROOT:?VITA_ROOT is required}"
VITA_PYTHON="${VITA_PYTHON:-/root/venvs/dressage-cu129-py312/bin/python}"
TOKENIZER="${VITA_INPUT_TOKENIZER_PATH:-/root/models/Qwen3.5-4B}"
TASK_SET="${VITA_TASK_SET:-delivery}"
TASK_LANGUAGE="${VITA_TASK_LANGUAGE:-english}"
TASK_COUNT="${VITA_TASK_COUNT:-100}"
TASK_IDS_OVERRIDE="${VITA_TASK_IDS:-}"
RECOVERY_OF="${VITA_RECOVERY_OF:-}"
SELECTION_SEED=20260906
MAX_STEPS=100
CONCURRENCY="${VITA_MAX_CONCURRENCY:-20}"
AGENT_MODEL="openai/gpt-5.6-terra"
AGENT_REASONING_EFFORT="${VITA_TERRA_REASONING_EFFORT:-high}"
USER_IMPLEMENTATION="${VITA_USER_IMPLEMENTATION:-user_simulator}"
USER_LLM="${VITA_USER_LLM:-gpt-4.1}"
RUN_ID="${VITA_EVAL_RUN_ID:-vessl-vitabench-${TASK_SET}-terra-upper-bound-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${VITA_EVAL_OUTPUT_ROOT:-/root/outputs/vita-rl/vitabench/${RUN_ID}}"

case "$TASK_SET" in
    delivery|ota) ;;
    *) echo "VITA_TASK_SET must be delivery or ota, got: $TASK_SET" >&2; exit 2 ;;
esac
case "$TASK_LANGUAGE" in
    english|chinese) ;;
    *) echo "VITA_TASK_LANGUAGE must be english or chinese, got: $TASK_LANGUAGE" >&2; exit 2 ;;
esac
case "$USER_IMPLEMENTATION" in
    user_simulator|vita_rl_deterministic_task_user) ;;
    *) echo "Unsupported VITA_USER_IMPLEMENTATION: $USER_IMPLEMENTATION" >&2; exit 2 ;;
esac

RESULT="$RUN_ROOT/result.json"
SUMMARY="$RUN_ROOT/summary.json"
SELECTION="$RUN_ROOT/task-selection.json"
TERRA_SMOKE="$RUN_ROOT/terra-tool-smoke.json"
OPENROUTER_LOG="$RUN_ROOT/logs/openrouter-proxy.log"
TASK_FILE="$VITA_ROOT/data/vita/domains/$TASK_SET/tasks.json"
if [[ "$TASK_LANGUAGE" == "english" ]]; then
    TASK_FILE="$VITA_ROOT/data/vita/domains/$TASK_SET/tasks_en.json"
fi

[[ -d "$REPO" ]] || { echo "Missing evaluation repository: $REPO" >&2; exit 2; }
[[ -d "$VITA_ROOT" ]] || { echo "Missing VitaBench checkout: $VITA_ROOT" >&2; exit 2; }
[[ -f "$TASK_FILE" ]] || { echo "Missing VitaBench task file: $TASK_FILE" >&2; exit 2; }
[[ -x "$VITA_PYTHON" ]] || { echo "Missing VitaBench Python: $VITA_PYTHON" >&2; exit 2; }
[[ -d "$TOKENIZER" ]] || { echo "Missing tokenizer used for VitaBench input accounting: $TOKENIZER" >&2; exit 2; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY is required in-memory" >&2; exit 2; }

mkdir -p "$RUN_ROOT/logs"
export VITA_RL_ROOT="$REPO"
export VITA_INPUT_TOKENIZER_PATH="$TOKENIZER"

OPENROUTER_PID=""
OPENROUTER_PORT_FILE=""
VITA_MODEL_CONFIG=""
cleanup() {
    local exit_code=$?
    [[ -n "$OPENROUTER_PID" ]] && kill "$OPENROUTER_PID" 2>/dev/null || true
    [[ -z "$OPENROUTER_PORT_FILE" ]] || rm -f "$OPENROUTER_PORT_FILE"
    [[ -z "$VITA_MODEL_CONFIG" ]] || rm -f "$VITA_MODEL_CONFIG"
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

# The benchmark receives a placeholder credential only. The proxy inherits the
# raw credential, then the runner clears its own environment copy.
OPENROUTER_PORT_FILE="$(mktemp /tmp/vita-rl-openrouter.XXXXXX)"
"$VITA_PYTHON" "$REPO/scripts/openrouter_proxy.py" --port-file "$OPENROUTER_PORT_FILE" \
    >"$OPENROUTER_LOG" 2>&1 &
OPENROUTER_PID=$!
for _ in $(seq 1 100); do
    [[ -s "$OPENROUTER_PORT_FILE" ]] && break
    kill -0 "$OPENROUTER_PID" 2>/dev/null || { echo "OpenRouter proxy exited early" >&2; exit 1; }
    sleep 0.1
done
[[ -s "$OPENROUTER_PORT_FILE" ]] || { echo "OpenRouter proxy startup timed out" >&2; exit 1; }

VITA_MODEL_CONFIG="$(mktemp /tmp/vita-rl-models.XXXXXX.yaml)"
"$VITA_PYTHON" - "$VITA_MODEL_CONFIG" "$(<"$OPENROUTER_PORT_FILE")" "$AGENT_MODEL" "$AGENT_REASONING_EFFORT" <<'PY'
import json
import sys
from pathlib import Path

path, proxy_port, agent_model, reasoning_effort = sys.argv[1:]
proxy_headers = {"Authorization": "Bearer vita-rl-local-proxy", "Content-Type": "application/json"}
config = {
    "default": {"base_url": f"http://127.0.0.1:{proxy_port}/v1/chat/completions", "headers": proxy_headers},
    "models": [
        {"name": "gpt-4.1", "temperature": 0.0, "max_tokens": 4096, "max_input_tokens": 32768},
        {
            "name": agent_model,
            "max_tokens": 8192,
            "max_input_tokens": 1050000,
            "reasoning_effort": reasoning_effort,
        },
    ],
}
Path(path).write_text(json.dumps(config, indent=2) + "\n")
PY
export VITA_MODEL_CONFIG_PATH="$VITA_MODEL_CONFIG"
unset OPENROUTER_API_KEY

# Verify the exact remote model, reasoning setting, and OpenAI-style tool-call
# shape before starting a costly full benchmark. The smoke artifact contains no
# prompt text, response text, or credentials.
"$VITA_PYTHON" - "$(<"$OPENROUTER_PORT_FILE")" "$AGENT_MODEL" "$AGENT_REASONING_EFFORT" "$TERRA_SMOKE" <<'PY'
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

port, model, reasoning_effort, output = sys.argv[1:]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Use the available function to look up the weather in Beijing."}],
    "tools": [{"type": "function", "function": {
        "name": "weather_lookup", "description": "Look up weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": False},
    }}],
    "tool_choice": "auto",
    "reasoning_effort": reasoning_effort,
    "max_tokens": 256,
}
request = Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Authorization": "Bearer vita-rl-local-proxy", "Content-Type": "application/json"},
    method="POST",
)
with urlopen(request, timeout=600) as response:
    reply = json.loads(response.read())
choice = reply["choices"][0]
tool_calls = (choice.get("message") or {}).get("tool_calls") or []
if reply.get("model") != model or not tool_calls:
    raise SystemExit("GPT Terra tool/reasoning smoke did not return the requested model and tool call")
usage = reply.get("usage") or {}
Path(output).write_text(json.dumps({
    "model": reply.get("model"),
    "finish_reason": choice.get("finish_reason"),
    "tool_call_count": len(tool_calls),
    "usage": usage,
    "reasoning_tokens": ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")),
}, indent=2) + "\n")
PY

cd "$VITA_ROOT"
if [[ -n "$TASK_IDS_OVERRIDE" ]]; then
    read -r -a TASK_IDS <<< "$TASK_IDS_OVERRIDE"
    TASK_COUNT="${#TASK_IDS[@]}"
    [[ "$TASK_COUNT" -gt 0 ]] || { echo "VITA_TASK_IDS did not contain a task ID" >&2; exit 2; }
    PYTHONPATH="$VITA_ROOT/src" "$VITA_PYTHON" - "$SELECTION" "$TASK_SET" "$TASK_LANGUAGE" "$SELECTION_SEED" "${TASK_IDS[@]}" <<'PY'
import json
import sys
from vita.run import load_tasks

selection_path, task_set, language, seed, *task_ids = sys.argv[1:]
known = {task.id for task in load_tasks(task_set, language=language)}
unknown = [task_id for task_id in task_ids if task_id not in known]
if unknown:
    raise SystemExit(f"Unknown task IDs for {task_set}/{language}: {unknown}")
with open(selection_path, "w", encoding="utf-8") as handle:
    json.dump({"task_set": task_set, "task_language": language, "task_count": len(task_ids),
               "selection_seed": int(seed), "task_ids": task_ids}, handle, indent=2)
    handle.write("\n")
PY
else
PYTHONPATH="$VITA_ROOT/src" "$VITA_PYTHON" - "$SELECTION" "$TASK_SET" "$TASK_LANGUAGE" "$TASK_COUNT" "$SELECTION_SEED" <<'PY'
import json
import random
import sys
from vita.run import load_tasks

selection_path, task_set, language, task_count, seed = sys.argv[1:]
tasks = load_tasks(task_set, language=language)
selected = random.Random(int(seed)).sample(tasks, int(task_count))
with open(selection_path, "w", encoding="utf-8") as handle:
    json.dump({"task_set": task_set, "task_language": language, "task_count": int(task_count),
               "selection_seed": int(seed), "task_ids": [task.id for task in selected]}, handle, indent=2)
    handle.write("\n")
PY
fi
mapfile -t TASK_IDS < <("$VITA_PYTHON" - "$SELECTION" <<'PY'
import json
import sys
for task_id in json.load(open(sys.argv[1], encoding="utf-8"))["task_ids"]:
    print(task_id)
PY
)
[[ "${#TASK_IDS[@]}" -eq "$TASK_COUNT" ]] || { echo "Task selection was incomplete" >&2; exit 1; }

PYTHONPATH="$REPO/src:$VITA_ROOT/src" "$VITA_PYTHON" -m vita_rl.vita_cli run \
    --domain "$TASK_SET" --task-set-name "$TASK_SET" --task-ids "${TASK_IDS[@]}" \
    --agent llm_agent --agent-llm "$AGENT_MODEL" --user "$USER_IMPLEMENTATION" --user-llm "$USER_LLM" --evaluator-llm gpt-4.1 \
    --enable-think --max-steps "$MAX_STEPS" --num-trials 1 --max-concurrency "$CONCURRENCY" \
    --language "$TASK_LANGUAGE" --save-to "$RESULT"

REPO_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
VITABENCH_COMMIT="$(git -C "$VITA_ROOT" rev-parse HEAD)"
"$VITA_PYTHON" - "$RESULT" "$SUMMARY" "$SELECTION" "$TERRA_SMOKE" "$REPO_COMMIT" "$VITABENCH_COMMIT" \
    "$RUN_ID" "$CONCURRENCY" "$AGENT_MODEL" "$AGENT_REASONING_EFFORT" "$USER_IMPLEMENTATION" "$USER_LLM" "$TASK_COUNT" "$RECOVERY_OF" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

(result_path, summary_path, selection_path, smoke_path, repo_commit, vitabench_commit,
 run_id, concurrency, agent_model, reasoning_effort, user_implementation, user_llm,
 expected_task_count, recovery_of) = sys.argv[1:]
result = json.loads(Path(result_path).read_text())
simulations = result.get("simulations", [])
if len(simulations) != int(expected_task_count):
    raise SystemExit(f"Expected {expected_task_count} completed tasks, got {len(simulations)}")
selection = json.loads(Path(selection_path).read_text())
records = []
user_turn_count = 0
for simulation in simulations:
    messages = simulation.get("messages", [])
    user_messages = [message for message in messages if message.get("role") == "user"]
    if user_implementation == "vita_rl_deterministic_task_user":
        for message in user_messages:
            raw_data = message.get("raw_data") or {}
            usage = message.get("usage") or {}
            if raw_data.get("llm_called") is not False or any(
                usage.get(key, 0) != 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            ):
                raise SystemExit("A deterministic-user record contained model usage or lacked its no-LLM marker")
    usages = [message.get("usage") or {} for message in messages]
    records.append({
        "task_id": simulation.get("task_id"),
        "reward": (simulation.get("reward_info") or {}).get("reward"),
        "success": (simulation.get("reward_info") or {}).get("reward") == 1.0,
        "agent_steps": sum(message.get("role") == "assistant" for message in messages),
        "wall_clock_seconds": simulation.get("duration"),
        "prompt_tokens_total": sum(usage.get("prompt_tokens", 0) for usage in usages),
        "output_tokens_total": sum(usage.get("completion_tokens", 0) for usage in usages),
        "tool_error_count": sum(bool(message.get("error")) for message in messages if message.get("role") == "tool"),
        "termination_reason": simulation.get("termination_reason"),
    })
    user_turn_count += len(user_messages)
summary = {
    "benchmark": "VitaBench",
    "run_id": run_id,
    "role_models": {"agent": agent_model, "user_simulator": user_llm, "evaluator": "gpt-4.1"},
    "agent_inference": {"thinking": True, "reasoning_effort": reasoning_effort, "max_tokens": 8192, "max_input_tokens": 1050000},
    "harness": "llm_agent",
    "task_set": selection["task_set"],
    "task_language": selection["task_language"],
    "selection_seed": selection["selection_seed"],
    "task_count": len(records),
    "max_steps": 100,
    "requested_concurrency": int(concurrency),
    "result_file": result_path,
    "task_selection_file": selection_path,
    "terra_smoke_file": smoke_path,
    "provenance": {"vita_rl_commit": repo_commit, "vitabench_commit": vitabench_commit},
    "aggregate": {
        "successes": sum(record["success"] for record in records),
        "mean_reward": sum((record["reward"] or 0.0) for record in records) / len(records),
        "termination_reasons": dict(Counter(record["termination_reason"] for record in records)),
        "tool_error_count": sum(record["tool_error_count"] for record in records),
        "agent_steps": sum(record["agent_steps"] for record in records),
        "prompt_tokens": sum(record["prompt_tokens_total"] for record in records),
        "output_tokens": sum(record["output_tokens_total"] for record in records),
    },
    "tasks": records,
}
if user_implementation == "vita_rl_deterministic_task_user":
    summary["comparison_label"] = "deterministic-one-shot-autonomous; not comparable to GPT-4.1-user results"
    summary["user_simulator"] = {
        "implementation": user_implementation, "llm_calls": False,
        "policy": "request-plus-public-profile-once_then_two_fact_free_reviews_then_agent_stop",
        "hidden_state_access": False, "user_turn_count": user_turn_count,
        "user_prompt_tokens": 0, "user_output_tokens": 0, "user_cost": 0.0,
    }
else:
    summary["user_simulator"] = {"implementation": user_implementation, "llm_calls": True}
if recovery_of:
    summary["recovery_of"] = recovery_of
Path(summary_path).write_text(json.dumps(summary, indent=2) + "\n")
print(f"summary={summary_path}")
PY
