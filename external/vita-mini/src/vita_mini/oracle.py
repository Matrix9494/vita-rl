"""Exact feasibility verifier for procedurally generated mini tasks."""

from __future__ import annotations

from copy import deepcopy

from .tasks import MiniTask


def verify_oracle_solution(task: MiniTask) -> bool:
    """Replay the declared oracle plan and require strict deterministic success."""
    from .environment import MiniEnvironment

    environment = MiniEnvironment(tasks={task.task_id: task})
    environment.reset(task.task_id)
    for action in task.oracle_solution:
        result = environment.call_tool(action["tool"], action["arguments"])
        if not result.ok:
            return False
        environment.next_user_event(tool_name=action["tool"], tool_success=True)
    return environment.evaluate().success
