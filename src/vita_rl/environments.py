"""Environment-neutral contracts and registry for deterministic tool tasks."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Protocol


class ToolUseEnvironment(Protocol):
    """The common agent-facing contract for custom tool environments."""

    def reset(self, task_id: str | None = None) -> dict[str, Any]: ...
    def openai_tools(self) -> list[dict[str, Any]]: ...
    def agent_policy(self) -> str: ...
    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any: ...
    def evaluate(self) -> Any: ...


EnvironmentFactory = Callable[[], ToolUseEnvironment]


class EnvironmentRegistry:
    """Names and lazily constructs runnable deterministic environments."""

    def __init__(self) -> None:
        self._factories: dict[str, EnvironmentFactory] = {}

    def register(self, name: str, factory: EnvironmentFactory) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("Environment name cannot be empty")
        if normalized in self._factories:
            raise ValueError(f"Environment {normalized!r} is already registered")
        self._factories[normalized] = factory

    def create(self, name: str) -> ToolUseEnvironment:
        normalized = name.strip().lower()
        try:
            return self._factories[normalized]()
        except KeyError as exc:
            raise ValueError(
                f"Unknown tool environment {name!r}; available: {', '.join(self.names())}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def _create_vita_mini() -> ToolUseEnvironment:
    """Import the external package only when this environment is selected."""
    return import_module("vita_mini").MiniEnvironment()


def _create_bfcl_multi_turn_base() -> ToolUseEnvironment:
    """Load BFCL only when its explicit environment is selected."""
    return import_module("vita_rl.bfcl_environment").BFCLMultiTurnBaseEnvironment()


tool_environment_registry = EnvironmentRegistry()
tool_environment_registry.register("vita-mini", _create_vita_mini)
tool_environment_registry.register("bfcl-multi-turn-base", _create_bfcl_multi_turn_base)
