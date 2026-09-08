"""Environment-neutral localhost runtime used by Dressage whitebox rollouts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from vita_rl.environment_runner import OpenAICompatibleGenerator, run_tool_environment_episode
from vita_rl.environments import tool_environment_registry


class EpisodeValidationError(ValueError):
    """A client supplied an invalid environment episode request."""


def _required_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EpisodeValidationError(f"{key} must be a non-empty string")
    return value.strip()


def _positive_integer(data: Mapping[str, Any], key: str, default: int) -> int:
    try:
        value = int(data.get(key, default))
    except (TypeError, ValueError) as exc:
        raise EpisodeValidationError(f"{key} must be an integer") from exc
    if value <= 0:
        raise EpisodeValidationError(f"{key} must be greater than zero")
    return value


@dataclass(frozen=True)
class EpisodeRequest:
    """One model rollout request, independent of the selected environment."""

    environment: str
    task_id: str
    session_id: str
    instance_id: str
    agent_model: str
    dressage_proxy_url: str
    environment_args: dict[str, Any] = field(default_factory=dict)
    sampling_params: dict[str, Any] = field(default_factory=dict)
    max_steps: int = 300
    max_errors: int = 10

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeRequest":
        if not isinstance(data, Mapping):
            raise EpisodeValidationError("episode request must be an object")
        sampling_params = data.get("sampling_params", {})
        if not isinstance(sampling_params, Mapping):
            raise EpisodeValidationError("sampling_params must be an object")
        raw_environment_args = data.get("environment_args", {})
        if not isinstance(raw_environment_args, Mapping):
            raise EpisodeValidationError("environment_args must be an object")
        # Existing Vessl prompt records used domain/language fields directly.
        environment = str(data.get("environment") or "vitabench").strip().lower()
        environment_args = dict(raw_environment_args)
        for legacy_key in ("domain", "language"):
            if legacy_key in data and legacy_key not in environment_args:
                environment_args[legacy_key] = data[legacy_key]
        request = cls(
            environment=environment,
            task_id=_required_string(data, "task_id"),
            session_id=_required_string(data, "session_id"),
            instance_id=_required_string(data, "instance_id"),
            agent_model=_required_string(data, "agent_model"),
            dressage_proxy_url=_required_string(data, "dressage_proxy_url"),
            environment_args=environment_args,
            sampling_params=dict(sampling_params),
            max_steps=_positive_integer(data, "max_steps", 300),
            max_errors=_positive_integer(data, "max_errors", 10),
        )
        request.validate()
        return request

    @property
    def domain(self) -> str:
        """Compatibility view for the legacy VitaBench backend."""
        return str(self.environment_args.get("domain", "delivery"))

    @property
    def language(self) -> str:
        """Compatibility view for the legacy VitaBench backend."""
        return str(self.environment_args.get("language", "chinese"))

    def validate(self) -> None:
        supported = {"vitabench", *tool_environment_registry.names()}
        if self.environment not in supported:
            raise EpisodeValidationError(
                f"unsupported environment {self.environment!r}; supported: {', '.join(sorted(supported))}"
            )
        if self.environment == "vitabench" and not self.domain:
            raise EpisodeValidationError("vitabench domain must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "instance_id": self.instance_id,
            "agent_model": self.agent_model,
            "dressage_proxy_url": self.dressage_proxy_url.rstrip("/"),
            "environment_args": dict(self.environment_args),
            "sampling_params": dict(self.sampling_params),
            "max_steps": self.max_steps,
            "max_errors": self.max_errors,
        }


@dataclass(frozen=True)
class EpisodeResponse:
    task_id: str
    completed: bool
    reward: float
    termination_reason: str
    num_agent_turns: int
    proxy_turns: int
    simulation: dict[str, Any]
    final_assistant_response: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeResponse":
        if not isinstance(data, Mapping) or not isinstance(data.get("simulation", {}), Mapping):
            raise EpisodeValidationError("invalid episode response")
        return cls(
            task_id=_required_string(data, "task_id"),
            completed=bool(data.get("completed")),
            reward=float(data.get("reward", 0.0)),
            termination_reason=_required_string(data, "termination_reason"),
            num_agent_turns=int(data.get("num_agent_turns", 0)),
            proxy_turns=int(data.get("proxy_turns", 0)),
            simulation=dict(data["simulation"]),
            final_assistant_response=str(data.get("final_assistant_response") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "completed": self.completed,
            "reward": self.reward,
            "termination_reason": self.termination_reason,
            "num_agent_turns": self.num_agent_turns,
            "proxy_turns": self.proxy_turns,
            "simulation": dict(self.simulation),
            "final_assistant_response": self.final_assistant_response,
        }


class DressageProxyGenerator(OpenAICompatibleGenerator):
    """Forward each harness action to Dressage's recording model proxy."""

    def __init__(self, request: EpisodeRequest) -> None:
        super().__init__(f"{request.dressage_proxy_url.rstrip('/')}/v1/chat/completions")
        self.request = request
        self.turn_count = 0

    def __call__(self, **kwargs: Any) -> Any:
        self.turn_count += 1
        return super().__call__(
            extra_headers={
                "X-Session-Id": self.request.session_id,
                "X-Instance-Id": self.request.instance_id,
                "X-Turn-Id": f"{self.request.session_id}:agent:{self.turn_count:04d}",
            },
            **kwargs,
        )


def run_environment_episode(request: EpisodeRequest) -> EpisodeResponse:
    """Dispatch a neutral episode request to the selected environment backend."""
    request.validate()
    if request.environment == "vitabench":
        from vita_rl.vitabench_backend import run_vitabench_episode

        return run_vitabench_episode(request)
    generator = DressageProxyGenerator(request)
    result = run_tool_environment_episode(
        environment_name=request.environment,
        task_id=request.task_id,
        harness_name=str(request.environment_args.get("harness", "vita_rl_standard")),
        model=request.agent_model,
        generate_fn=generator,
        llm_args=request.sampling_params,
        max_steps=request.max_steps,
        max_errors=request.max_errors,
        enable_think=bool(request.environment_args.get("enable_think", False)),
    )
    return EpisodeResponse(
        task_id=result.task_id,
        completed=result.success,
        reward=result.reward,
        termination_reason=result.termination_reason,
        num_agent_turns=result.num_agent_turns,
        proxy_turns=generator.turn_count,
        simulation={
            "environment": result.environment,
            "num_tool_calls": result.num_tool_calls,
            "num_tool_errors": result.num_tool_errors,
            "user_events": result.user_events,
            "evaluation": result.evaluation,
            "messages": result.messages,
        },
        final_assistant_response=result.final_assistant_response,
    )


def create_app(runner: Callable[[EpisodeRequest], EpisodeResponse] | None = None):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError("FastAPI is required to host the environment runtime") from exc
    app = FastAPI(title="vita-rl environment runtime")
    run = runner or run_environment_episode

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "environments": ["vitabench", *tool_environment_registry.names()]}

    @app.post("/episode")
    def episode(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return run(EpisodeRequest.from_dict(payload)).to_dict()
        except EpisodeValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="environment episode execution failed") from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9010)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
