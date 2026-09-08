"""Dressage whitebox adapter for any registered environment runtime."""

from __future__ import annotations

import inspect
import os
from typing import Any, Callable

from vita_rl.runtime_server import EpisodeRequest, EpisodeResponse


try:
    from dressage.config import proxy_url as _proxy_url
    from dressage.rollout.generate.whitebox_agent import WhiteboxAgent, make_generate
except ImportError as _dressage_error:  # Keeps lightweight local tests importable.
    _DRESSAGE_IMPORT_ERROR = _dressage_error
    _proxy_url = None

    class WhiteboxAgent:  # type: ignore[no-redef]
        pass

    def make_generate(_cls):  # type: ignore[no-redef]
        async def unavailable(*_args, **_kwargs):
            raise RuntimeError("Dressage is required for this generate hook") from _DRESSAGE_IMPORT_ERROR

        return unavailable


class RuntimeClient:
    """Small async client for the common environment runtime endpoint."""

    def __init__(self, base_url: str, *, post: Callable | None = None):
        self.base_url = base_url.rstrip("/")
        self._post = post

    async def episode(self, request: EpisodeRequest) -> EpisodeResponse:
        if self._post is not None:
            data = self._post(f"{self.base_url}/episode", request.to_dict())
            if inspect.isawaitable(data):
                data = await data
        else:
            import httpx

            async with httpx.AsyncClient(timeout=900, trust_env=False) as client:
                response = await client.post(f"{self.base_url}/episode", json=request.to_dict())
                response.raise_for_status()
                data = response.json()
        return EpisodeResponse.from_dict(data)


class EnvironmentWhiteboxAgent(WhiteboxAgent):
    """Delegate agent/tool interaction to the requested environment runtime."""

    name = "environment_whitebox_agent"
    session_prefix = "environment"
    runtime_client_factory = RuntimeClient

    async def rollout(self, sample: Any, sampling_params: dict[str, Any]) -> str:
        if _proxy_url is None:
            raise RuntimeError("Dressage is required")
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            sample.metadata = metadata
        environment = str(metadata.get("environment", metadata.get("vita_environment", "vitabench")))
        environment_args = dict(metadata.get("environment_args") or {})
        # Backward compatibility for existing VitaBench prompt datasets.
        if environment == "vitabench":
            environment_args.setdefault("domain", metadata.get("vita_domain", "delivery"))
            environment_args.setdefault("language", metadata.get("vita_language", "chinese"))
        for key in ("harness", "enable_think"):
            if key in metadata and key not in environment_args:
                environment_args[key] = metadata[key]
        request = EpisodeRequest(
            environment=environment,
            task_id=str(metadata.get("task_id", metadata.get("vita_task_id", "10711001"))),
            session_id=str(self.session_id),
            instance_id=str(self.instance_id),
            agent_model=str(metadata.get("agent_model", metadata.get("vita_agent_model", "proxy-model"))),
            dressage_proxy_url=str(_proxy_url()),
            environment_args=environment_args,
            sampling_params=dict(sampling_params or {}),
            max_steps=int(metadata.get("max_steps", metadata.get("vita_max_steps", 300))),
            max_errors=int(metadata.get("max_errors", metadata.get("vita_max_errors", 10))),
        )
        result = await self.runtime_client_factory(
            str(metadata.get("runtime_url") or metadata.get("vita_runtime_url") or os.environ.get("ENVIRONMENT_RUNTIME_URL", os.environ.get("VITA_RUNTIME_URL", "http://127.0.0.1:9010")))
        ).episode(request)
        metadata.update(
            environment_name=environment,
            environment_reward=float(result.reward),
            environment_task_id=result.task_id,
            environment_termination_reason=result.termination_reason,
            environment_num_agent_turns=int(result.num_agent_turns),
            environment_proxy_turns=int(result.proxy_turns),
            environment_completed=bool(result.completed),
            environment_simulation=dict(result.simulation),
            # Preserve old reward hooks and diagnostics until Vessl job specs
            # migrate to the neutral metadata keys.
            vita_reward=float(result.reward),
            vita_task_id=result.task_id,
            vita_termination_reason=result.termination_reason,
            vita_num_agent_turns=int(result.num_agent_turns),
            vita_proxy_turns=int(result.proxy_turns),
            vita_completed=bool(result.completed),
            vita_simulation=dict(result.simulation),
        )
        return result.final_assistant_response


# Stable aliases avoid breaking existing Vessl configurations while all new
# configurations use EnvironmentWhiteboxAgent/RuntimeClient and `environment`.
VitaRuntimeClient = RuntimeClient
VitaWhiteboxAgent = EnvironmentWhiteboxAgent


def create_adapter():
    return EnvironmentWhiteboxAgent


generate = make_generate(EnvironmentWhiteboxAgent)
