"""Replaceable delta proposers; only the updater's output crosses into CPU apply."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .prompts import state_update_messages


@dataclass(frozen=True)
class StateDeltaProposal:
    raw_delta: str
    delta: Any
    updater_name: str


class StateUpdater(Protocol):
    def update(
        self,
        *,
        state: dict[str, Any],
        metadata: dict[str, dict[str, Any]],
        observation: dict[str, Any],
        unresolved_evidence: list[dict[str, Any]],
        turn_idx: int,
    ) -> StateDeltaProposal: ...


def _parse_json_object(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return json.loads(text)


class LLMStateUpdater:
    """Calls an injected model function solely for delta proposal generation."""

    def __init__(self, model_call: Callable[[str, str], str]):
        self._model_call = model_call

    def update(self, **kwargs: Any) -> StateDeltaProposal:
        system, user = state_update_messages(
            state=kwargs["state"],
            unresolved_evidence=kwargs["unresolved_evidence"],
            observation=kwargs["observation"],
        )
        raw = self._model_call(system, user)
        try:
            delta = _parse_json_object(raw)
        except (json.JSONDecodeError, IndexError, TypeError) as exc:
            delta = {"malformed_model_output": raw, "parse_error": str(exc)}
        return StateDeltaProposal(raw_delta=raw, delta=delta, updater_name="llm")


class NoOpStateUpdater:
    """Oracle/ablation implementation that never invokes an LLM."""

    def update(self, **kwargs: Any) -> StateDeltaProposal:
        del kwargs
        return StateDeltaProposal(
            raw_delta='{"ops":[{"op":"noop"}]}',
            delta={"ops": [{"op": "noop"}]},
            updater_name="noop",
        )


class ReplayStateUpdater:
    """Load gold/precomputed deltas from JSON or JSONL keyed by turn index.

    Accepted records are ``{"turn": 1, "delta": {"ops": [...]}}`` or a
    bare mapping ``{"1": {"ops": [...]}}``. Missing turns deliberately emit
    an invalid proposal so the observation is preserved in unresolved evidence.
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
            self._deltas = {int(record["turn"]): record.get("delta", record.get("raw_delta")) for record in records}
        else:
            payload = json.loads(text)
            records = payload.get("records", payload) if isinstance(payload, dict) else payload
            if isinstance(records, dict) and "ops" not in records:
                self._deltas = {int(turn): delta for turn, delta in records.items()}
            else:
                self._deltas = {int(record["turn"]): record.get("delta", record.get("raw_delta")) for record in records}

    def update(self, *, turn_idx: int, **kwargs: Any) -> StateDeltaProposal:
        del kwargs
        delta = self._deltas.get(turn_idx)
        if delta is None:
            delta = {"missing_replay_delta": turn_idx}
        if isinstance(delta, str):
            raw = delta
            try:
                parsed = _parse_json_object(delta)
            except (json.JSONDecodeError, IndexError, TypeError):
                parsed = {"malformed_replay_delta": delta}
        else:
            raw = json.dumps(delta, ensure_ascii=False, sort_keys=True)
            parsed = delta
        return StateDeltaProposal(raw_delta=raw, delta=parsed, updater_name="replay")
