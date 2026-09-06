"""Dressage whitebox adapter; Vita stays behind the localhost runtime API."""
from __future__ import annotations
import inspect
import os
from typing import Any, Callable
from vita_rl.vita_server import EpisodeRequest, EpisodeResponse

try:
    from dressage.config import proxy_url as _proxy_url
    from dressage.rollout.generate.whitebox_agent import WhiteboxAgent, make_generate
except ImportError as _dressage_error:  # Allows LAIR unit tests without Dressage.
    _DRESSAGE_IMPORT_ERROR = _dressage_error
    _proxy_url = None
    class WhiteboxAgent: pass
    def make_generate(_cls):
        async def unavailable(*_args, **_kwargs):
            raise RuntimeError("Dressage is required for this generate hook") from _DRESSAGE_IMPORT_ERROR
        return unavailable

class VitaRuntimeClient:
    def __init__(self, base_url: str, *, post: Callable | None = None):
        self.base_url, self._post = base_url.rstrip("/"), post
    async def episode(self, request: EpisodeRequest) -> EpisodeResponse:
        if self._post is not None:
            data = self._post(f"{self.base_url}/episode", request.to_dict())
            if inspect.isawaitable(data): data = await data
        else:
            import httpx
            async with httpx.AsyncClient(timeout=900, trust_env=False) as client:
                response = await client.post(f"{self.base_url}/episode", json=request.to_dict())
                response.raise_for_status(); data = response.json()
        return EpisodeResponse.from_dict(data)

class VitaWhiteboxAgent(WhiteboxAgent):
    """Let Vita drive tools/users while proxy records every agent turn."""
    name, session_prefix = "vita_whitebox_agent", "vita"
    runtime_client_factory = VitaRuntimeClient
    async def rollout(self, sample: Any, sampling_params: dict[str, Any]) -> str:
        if _proxy_url is None: raise RuntimeError("Dressage is required")
        meta = getattr(sample, "metadata", None)
        if not isinstance(meta, dict): meta = {}; sample.metadata = meta
        request = EpisodeRequest(
            domain=str(meta.get("vita_domain", "delivery")), task_id=str(meta.get("vita_task_id", "10711001")),
            language=str(meta.get("vita_language", "chinese")), session_id=str(self.session_id),
            instance_id=str(self.instance_id), agent_model=str(meta.get("vita_agent_model", "proxy-model")),
            dressage_proxy_url=str(_proxy_url()), sampling_params=dict(sampling_params or {}),
            max_steps=int(meta.get("vita_max_steps", 300)), max_errors=int(meta.get("vita_max_errors", 10)),
        )
        result = await self.runtime_client_factory(str(meta.get("vita_runtime_url") or os.environ.get("VITA_RUNTIME_URL", "http://127.0.0.1:9010"))).episode(request)
        meta.update(vita_reward=float(result.reward), vita_task_id=result.task_id,
                    vita_termination_reason=result.termination_reason,
                    vita_num_agent_turns=int(result.num_agent_turns), vita_proxy_turns=int(result.proxy_turns),
                    vita_completed=bool(result.completed), vita_simulation=dict(result.simulation))
        return result.final_assistant_response

def create_adapter(): return VitaWhiteboxAgent
generate = make_generate(VitaWhiteboxAgent)
