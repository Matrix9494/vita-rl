#!/usr/bin/env bash
# Stop only an SGLang service previously started by resume_stack.sh.
set -euo pipefail

PID_FILE=/root/.vita_rl_runtime/sglang.pid

[[ -f "$PID_FILE" ]] || {
    echo "No managed SGLang PID file at $PID_FILE; no process was stopped."
    exit 0
}

SERVER_PID="$(<"$PID_FILE")"
[[ "$SERVER_PID" =~ ^[0-9]+$ ]] || { echo "Invalid PID file: $PID_FILE" >&2; exit 1; }

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "Managed SGLang process is already stopped."
    exit 0
fi

COMMAND="$(ps -p "$SERVER_PID" -o args=)"
case "$COMMAND" in
  *'sglang.launch_server'*) ;;
  *) echo "Refusing to stop unexpected process $SERVER_PID: $COMMAND" >&2; exit 1 ;;
esac

kill -TERM "$SERVER_PID"
for _ in $(seq 1 30); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 1
done
kill -0 "$SERVER_PID" 2>/dev/null && { echo "SGLang did not stop within 30 seconds." >&2; exit 1; }
rm -f "$PID_FILE"
echo "Stopped managed SGLang process $SERVER_PID."
