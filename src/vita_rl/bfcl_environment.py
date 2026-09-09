"""Deterministic BFCL V4 multi_turn_base tool environment.

This adapter delegates tool execution and terminal checking to the pinned BFCL
checkout. It does not duplicate BFCL's executable APIs or relax its checker.
"""

from __future__ import annotations

import copy
import os
import random
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BFCL_ENVIRONMENT_NAME = "bfcl-multi-turn-base"
BFCL_CATEGORY = "multi_turn_base"
BFCL_SPLIT_SEED = 300
BFCL_TRAIN_TASKS = 160


@dataclass(frozen=True)
class BFCLToolResult:
    """Environment-compatible result of one native BFCL executable call."""

    ok: bool
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BFCLEvaluationResult:
    """Terminal result returned by BFCL's exact multi-turn checker."""

    reward: float
    success: bool
    constraint_score: float
    tool_validity: float
    constraints: dict[str, bool]
    failed_conditions: list[str]
    state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_bfcl_root() -> Path:
    configured = os.environ.get("BFCL_SOURCE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path(__file__).resolve().parents[2]
        / "external"
        / "gorilla"
        / "berkeley-function-call-leaderboard"
    )


def _ensure_bfcl_importable(root: Path) -> None:
    if not root.is_dir():
        raise RuntimeError(
            f"BFCL source is unavailable at {root}; initialize external/gorilla "
            "or set BFCL_SOURCE_ROOT."
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def load_bfcl_multi_turn_base(
    *, source_root: Path | None = None
) -> tuple[dict[str, dict[str, Any]], dict[str, list[list[str]]]]:
    """Load native BFCL prompt and ground-truth records keyed by task ID."""
    root = source_root or _default_bfcl_root()
    _ensure_bfcl_importable(root)
    from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry

    entries = load_dataset_entry(
        BFCL_CATEGORY,
        include_prereq=False,
        include_language_specific_hint=False,
    )
    ground_truth = load_ground_truth_entry(BFCL_CATEGORY)
    by_id = {str(entry["id"]): entry for entry in entries}
    answers = {str(entry["id"]): list(entry["ground_truth"]) for entry in ground_truth}
    if len(by_id) != 200 or set(by_id) != set(answers):
        raise RuntimeError(
            "Expected the official BFCL V4 multi_turn_base corpus with 200 "
            "one-to-one prompt/ground-truth entries."
        )
    return by_id, answers


def bfcl_task_splits(
    *, source_root: Path | None = None, seed: int = BFCL_SPLIT_SEED
) -> tuple[list[str], list[str]]:
    """Return the fixed 160/40 BFCL train/eval task-ID partition."""
    entries, _ = load_bfcl_multi_turn_base(source_root=source_root)
    task_ids = sorted(entries)
    random.Random(seed).shuffle(task_ids)
    return task_ids[:BFCL_TRAIN_TASKS], task_ids[BFCL_TRAIN_TASKS:]


class BFCLMultiTurnBaseEnvironment:
    """One official BFCL multi-turn task using its native executors."""

    # BFCL advances from one user request to the next only after the model
    # returns an empty function-call response.  In particular, a request can
    # contain several function calls and several model/tool round trips.  The
    # generic runner must therefore not ask BFCL for a user event after every
    # individual tool result (as vita-mini legitimately does).
    user_events_after_tool_calls = False

    def __init__(self, *, source_root: Path | None = None) -> None:
        self._entries, self._ground_truth = load_bfcl_multi_turn_base(
            source_root=source_root
        )
        self._entry: dict[str, Any] | None = None
        self._task_id: str | None = None
        self._active_user_turn = 0
        self._model_calls: list[list[list[str]]] = []
        self._user_events: list[dict[str, Any]] = []
        self._tool_calls = 0
        self._tool_errors = 0
        # BFCL's native executor stores instances in module globals. A unique
        # namespace prevents samples in the same GRPO group sharing state.
        self._instance_namespace = f"vita_rl_bfcl_{uuid.uuid4().hex}"

    def reset(self, task_id: str | None = None) -> dict[str, Any]:
        if task_id is None:
            task_id = sorted(self._entries)[0]
        try:
            entry = self._entries[task_id]
        except KeyError as exc:
            raise ValueError(
                f"Unknown BFCL task {task_id!r}; expected an official "
                f"{BFCL_CATEGORY} task ID."
            ) from exc
        self._entry = copy.deepcopy(entry)
        self._task_id = task_id
        self._active_user_turn = 0
        self._model_calls = [[] for _ in self._entry["question"]]
        self._user_events = []
        self._tool_calls = 0
        self._tool_errors = 0
        self._instance_namespace = f"vita_rl_bfcl_{uuid.uuid4().hex}"
        self._initialize_native_state()
        return {
            "task_id": task_id,
            "user_message": self._turn_message(0),
            # The generic standard harness requires a parseable clock even
            # though BFCL itself has no wall-clock dependency.
            "logical_time": "2026-01-01T00:00:00Z",
        }

    @staticmethod
    def agent_policy() -> str:
        # BFCL FC tasks supply schemas directly and have no benchmark policy
        # system message. An empty policy suppresses the Vita default prompt.
        return ""

    def openai_tools(self) -> list[dict[str, Any]]:
        entry = self._require_entry()
        from bfcl_eval.constants.enums import ModelStyle
        from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
        from bfcl_eval.model_handler.utils import convert_to_tool

        return convert_to_tool(
            entry["function"], GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS
        )

    def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> BFCLToolResult:
        entry = self._require_entry()
        arguments = arguments or {}
        if not isinstance(arguments, dict):
            self._tool_errors += 1
            return BFCLToolResult(ok=False, error="Tool arguments must be an object.")
        allowed = {str(function["name"]) for function in entry["function"]}
        if name not in allowed:
            self._tool_errors += 1
            return BFCLToolResult(ok=False, error=f"Unknown BFCL tool {name!r}.")

        function_call = self._render_function_call(name, arguments)
        self._model_calls[self._active_user_turn].append([function_call])
        self._tool_calls += 1
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
            execute_multi_turn_func_call,
        )

        results, _ = execute_multi_turn_func_call(
            [function_call],
            entry["initial_config"],
            entry["involved_classes"],
            self._instance_namespace,
            entry["id"],
            long_context=False,
            is_evaL_run=False,
        )
        result = str(results[0])
        if result.startswith("Error during execution:"):
            self._tool_errors += 1
            return BFCLToolResult(ok=False, error=result)
        return BFCLToolResult(ok=True, result=result)

    def next_user_event(
        self,
        *,
        agent_turn: int | None = None,
        tool_name: str | None = None,
        tool_success: bool | None = None,
    ) -> list[dict[str, Any]]:
        del agent_turn, tool_name, tool_success
        entry = self._require_entry()
        next_turn = self._active_user_turn + 1
        if next_turn >= len(entry["question"]):
            return []
        self._active_user_turn = next_turn
        event = {
            "event_id": f"{self._task_id}:user_turn:{next_turn}",
            "message": self._turn_message(next_turn),
        }
        self._user_events.append(event)
        return [event]

    def evaluate(self) -> BFCLEvaluationResult:
        entry = self._require_entry()
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
            multi_turn_checker,
        )

        result = multi_turn_checker(
            self._model_calls,
            self._ground_truth[str(entry["id"])],
            entry,
            BFCL_CATEGORY,
            self._instance_namespace,
        )
        success = bool(result.get("valid"))
        error_type = str(result.get("error_type", ""))
        return BFCLEvaluationResult(
            reward=1.0 if success else 0.0,
            success=success,
            constraint_score=1.0 if success else 0.0,
            tool_validity=1.0 - self._tool_errors / max(1, self._tool_calls),
            constraints={"bfcl_exact_match": success},
            failed_conditions=[] if success else [error_type or "bfcl_exact_match"],
            state={
                "task_id": entry["id"],
                "active_user_turn": self._active_user_turn,
                "num_user_turns": len(entry["question"]),
                "model_calls": self._model_calls,
                "checker": result,
            },
        )

    def _initialize_native_state(self) -> None:
        entry = self._require_entry()
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
            execute_multi_turn_func_call,
        )

        execute_multi_turn_func_call(
            [],
            entry["initial_config"],
            entry["involved_classes"],
            self._instance_namespace,
            entry["id"],
            long_context=False,
            is_evaL_run=False,
        )

    def _turn_message(self, index: int) -> str:
        messages = self._require_entry()["question"][index]
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise RuntimeError(
                f"BFCL {BFCL_CATEGORY} entry {self._task_id} has an unsupported "
                "non-user turn shape."
            )
        return str(messages[0]["content"])

    def _require_entry(self) -> dict[str, Any]:
        if self._entry is None:
            raise RuntimeError("Call reset(task_id) before interacting with BFCL.")
        return self._entry

    @staticmethod
    def _render_function_call(name: str, arguments: dict[str, Any]) -> str:
        rendered_arguments = ",".join(
            f"{key}={value!r}" for key, value in arguments.items()
        )
        return f"{name}({rendered_arguments})"
