"""Explicit, serializable working state for VitaBench agent harnesses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentWorkingState:
    """State carried across an agent's action/observation transitions.

    An action is recorded when the agent emits it. At the next callback, the
    incoming user or tool observation advances this state before the following
    action is selected. The raw VitaBench conversation remains authoritative;
    this is a compact, explicit working view supplied to the harness.
    """

    turn: int = 0
    latest_observation: dict[str, Any] | None = None
    latest_user_request: str | None = None
    latest_tool_results: list[dict[str, Any]] = field(default_factory=list)
    pending_action: dict[str, Any] | None = None
    recent_observations: list[dict[str, Any]] = field(default_factory=list)
    recent_actions: list[dict[str, Any]] = field(default_factory=list)
    tool_error_count: int = 0

    _RECENT_LIMIT = 8

    def observe(self, observation: dict[str, Any]) -> None:
        """Advance state with the observation resulting from the prior action."""
        self.turn += 1
        self.latest_observation = observation
        self.recent_observations.append(observation)
        self.recent_observations = self.recent_observations[-self._RECENT_LIMIT :]

        if observation.get("kind") == "user":
            self.latest_user_request = str(observation.get("content") or "")
        if observation.get("kind") == "tool":
            results = list(observation.get("results") or [])
            self.latest_tool_results = results
            self.tool_error_count += sum(bool(result.get("error")) for result in results)

        # The pending action has now received its successor observation.
        self.pending_action = None

    def record_action(self, action: dict[str, Any]) -> None:
        """Record the action selected from the current state."""
        self.pending_action = action
        self.recent_actions.append(action)
        self.recent_actions = self.recent_actions[-self._RECENT_LIMIT :]

    def as_prompt(self) -> str:
        """Render a bounded state view for the action policy."""
        payload = {
            "turn": self.turn,
            "latest_user_request": self.latest_user_request,
            "latest_tool_results": self.latest_tool_results,
            "pending_action": self.pending_action,
            "recent_observations": self.recent_observations,
            "recent_actions": self.recent_actions,
            "tool_error_count": self.tool_error_count,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


# Preserve the original public placeholder alias for existing callers. New
# harnesses should use ``AgentWorkingState`` explicitly.
State = dict[str, Any]
