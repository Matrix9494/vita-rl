# GiveMeNode / `vita-dev` handoff

This file records the current GiveMeNode context for this repository. It contains no credentials.

## Node

- Node name: `vita-dev`
- Node ID: `83c53f4d-41aa-49f8-9ec4-09faaba9b12b`
- Mission: `vita-dev`
- Shape: one NVIDIA H100 (`chip: h100`, `gpu_count: 1`)
- Image: `huang3eng/dressage:nightly-dev-20260704a`
- Persistent home on the node: `/workspace`
- Repository path on the node: `/workspace/projects/vita-rl`

At handoff the node is stopped with its disk snapshotted. Its last durable snapshot timestamp was `2026-09-05T01:39:12.271Z`. Stopping/waking preserves files under `/workspace`; running processes do not survive a stop.

## Wake status and capacity

The latest wake request, `cmd-sktce`, was explicitly cancelled before it ever ran. Therefore no wake should be pending; confirm with `get_node` before submitting another one.

Recent H100 wake estimates were capacity-contended, with no new capacity being added. The last quote was approximately 94 minutes to ready, with a p90 near seven hours. This is only a live estimate, not a reservation, and it can change substantially.

Wake requests are served before new node creates. A new H100 node would likely also have queued under those conditions.

## Safe management pattern

Use the GiveMeNode tools with mission `vita-dev`:

1. Call `list_nodes({ mission: "vita-dev" })`, then `get_node({ name: "vita-dev", mission: "vita-dev" })`.
2. To request a wake, submit one detached `run_command` against `vita-dev` with `on_wake: true`. A short harmless verification command is sufficient, for example:

   ```sh
   cd /workspace/projects/vita-rl && hostname && nvidia-smi
   ```

3. Poll `get_node` until the node is running, then poll the detached command with `get_command`.
4. Cancel a pending wake by calling `kill_command` for its `cmd-...` ID. This does not delete the node or its disk.
5. When the node is no longer needed, call `stop_node` promptly to avoid billing. Do not use a sleep loop as a keepalive.

Do not create a second node named `vita-dev` and do not delete this node unless the disk is intentionally disposable.

## Billing

The platform charges only while a node is running or in its idle grace period. The most recently reported H100 weekend rate was `$0.05328/min`, with a 15-minute session minimum (`$0.79920`). Calendar discounts can change the displayed rate.

## Prior diagnostic baseline

A 20-task VitaBench diagnostic baseline completed successfully on this node before it was stopped. Results remain on the node disk:

- Output root: `/workspace/projects/vita-rl/outputs/diagnostic_baseline_20`
- Simulation trajectories: `/workspace/projects/vitabench/data/simulations/diagnostic_baseline_20-*`

Recorded aggregate result: 6/20 success (mean reward `0.300`). The local repository was not modified for that run; it should remain the source of truth for any future code changes.

## Local-repository rule

Make repository changes in the local checkout `/u/dz13/vita-rl`, test them there, then commit/push from local. Do not edit or commit source changes directly in `/workspace/projects/vita-rl` on the node.
