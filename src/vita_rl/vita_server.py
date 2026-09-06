"""Vita-only localhost service for Dressage whitebox rollouts.

Imports of FastAPI, requests, and VitaBench are delayed until runtime so this
module remains importable in LAIR tests and Dressage's system Python.
"""
from __future__ import annotations
import argparse, json, threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

SUPPORTED = {"domain": "delivery", "task_id": "10711001", "language": "chinese"}
_LOCK = threading.RLock()
class EpisodeValidationError(ValueError): pass
def _string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip(): raise EpisodeValidationError(f"{key} must be a non-empty string")
    return value.strip()
def _positive(data: Mapping[str, Any], key: str, default: int) -> int:
    try: value = int(data.get(key, default))
    except (TypeError, ValueError) as exc: raise EpisodeValidationError(f"{key} must be an integer") from exc
    if value <= 0: raise EpisodeValidationError(f"{key} must be greater than zero")
    return value

@dataclass(frozen=True)
class EpisodeRequest:
    domain: str; task_id: str; language: str; session_id: str; instance_id: str; agent_model: str; dressage_proxy_url: str
    sampling_params: dict[str, Any] = field(default_factory=dict); max_steps: int = 300; max_errors: int = 10
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        if not isinstance(data, Mapping): raise EpisodeValidationError("episode request must be an object")
        sampling = data.get("sampling_params", {})
        if not isinstance(sampling, Mapping): raise EpisodeValidationError("sampling_params must be an object")
        obj = cls(*[_string(data, k) for k in ("domain", "task_id", "language", "session_id", "instance_id", "agent_model", "dressage_proxy_url")], sampling_params=dict(sampling), max_steps=_positive(data,"max_steps",300), max_errors=_positive(data,"max_errors",10))
        obj.validate_supported(); return obj
    def validate_supported(self):
        for key, expected in SUPPORTED.items():
            if getattr(self, key) != expected: raise EpisodeValidationError(f"unsupported {key} {getattr(self,key)!r}; first milestone supports {expected!r}")
    def to_dict(self): return {"domain":self.domain,"task_id":self.task_id,"language":self.language,"session_id":self.session_id,"instance_id":self.instance_id,"agent_model":self.agent_model,"dressage_proxy_url":self.dressage_proxy_url.rstrip("/"),"sampling_params":dict(self.sampling_params),"max_steps":self.max_steps,"max_errors":self.max_errors}

@dataclass(frozen=True)
class EpisodeResponse:
    task_id: str; completed: bool; reward: float; termination_reason: str; num_agent_turns: int; proxy_turns: int; simulation: dict[str, Any]; final_assistant_response: str
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        if not isinstance(data, Mapping) or not isinstance(data.get("simulation", {}), Mapping): raise EpisodeValidationError("invalid episode response")
        return cls(_string(data,"task_id"),bool(data.get("completed")),float(data.get("reward",0)),_string(data,"termination_reason"),int(data.get("num_agent_turns",0)),int(data.get("proxy_turns",0)),dict(data.get("simulation",{})),str(data.get("final_assistant_response") or ""))
    def to_dict(self): return {"task_id":self.task_id,"completed":self.completed,"reward":self.reward,"termination_reason":self.termination_reason,"num_agent_turns":self.num_agent_turns,"proxy_turns":self.proxy_turns,"simulation":dict(self.simulation),"final_assistant_response":self.final_assistant_response}

class AgentProxyBridge:
    def __init__(self, request: EpisodeRequest): self.request, self.turn_count = request, 0
    def __call__(self, model, messages, tools=None, tool_choice=None, enable_think=False, **kwargs):
        del enable_think
        import requests
        from vita.data_model.message import AssistantMessage, ToolCall
        from vita.utils.llm_utils import format_messages
        self.turn_count += 1
        body = {"model": model or self.request.agent_model, "messages": format_messages(messages), "stream": False}
        if tools: body["tools"] = [tool.openai_schema for tool in tools]
        if tools: body["tool_choice"] = tool_choice or "auto"
        for key in ("temperature", "max_tokens", "top_p", "seed"):
            if kwargs.get(key) is not None: body[key] = kwargs[key]
        if "max_tokens" not in body and self.request.sampling_params.get("max_new_tokens"): body["max_tokens"] = int(self.request.sampling_params["max_new_tokens"])
        response = requests.post(f"{self.request.dressage_proxy_url.rstrip('/')}/v1/chat/completions", json=body, headers={"X-Session-Id":self.request.session_id,"X-Instance-Id":self.request.instance_id,"X-Turn-Id":f"{self.request.session_id}:agent:{self.turn_count:04d}"}, timeout=(10,900))
        response.raise_for_status(); payload = response.json(); choices = payload.get("choices") or []
        if not choices: raise RuntimeError("Dressage proxy response has no choices")
        choice = choices[0]; message = choice.get("message") or {}; calls=[]
        for call in message.get("tool_calls") or []:
            function=call.get("function") or {}
            try: arguments=json.loads(function.get("arguments") or "{}")
            except (TypeError,json.JSONDecodeError) as exc: raise RuntimeError("invalid proxy tool-call JSON") from exc
            calls.append(ToolCall(id=str(call.get("id") or ""),name=str(function.get("name") or ""),arguments=arguments,requestor="assistant"))
        if not message.get("content") and not calls: raise RuntimeError("Dressage proxy returned empty assistant message")
        return AssistantMessage(role="assistant",content=message.get("content"),tool_calls=calls or None,cost=0.0,usage=payload.get("usage") or {"prompt_tokens":0,"completion_tokens":0},raw_data=choice)

def run_vita_episode(request: EpisodeRequest) -> EpisodeResponse:
    request.validate_supported()
    from vita.agent import llm_agent
    from vita.config import models
    from vita.data_model.message import AssistantMessage
    from vita.run import get_tasks, run_task
    task = get_tasks(request.domain,[request.task_id],language=request.language)[0]; bridge=AgentProxyBridge(request)
    agent_args=dict(request.sampling_params)
    if "max_new_tokens" in agent_args and "max_tokens" not in agent_args: agent_args["max_tokens"]=agent_args["max_new_tokens"]
    with _LOCK:
        original=llm_agent.generate; llm_agent.generate=bridge
        try:
            simulation=run_task(domain=request.domain,task=task,agent="llm_agent",user="user_simulator",llm_agent=request.agent_model,llm_args_agent=agent_args,llm_user="gpt-4.1",llm_args_user=dict(models["gpt-4.1"]),llm_evaluator="gpt-4.1",llm_args_evaluator=dict(models["gpt-4.1"]),max_steps=request.max_steps,max_errors=request.max_errors,max_retries=0,language=request.language)
        finally: llm_agent.generate=original
    if simulation.reward_info is None: raise RuntimeError("Vita evaluator returned no reward")
    final=next((str(m.content) for m in reversed(simulation.messages) if isinstance(m,AssistantMessage) and m.content),"")
    return EpisodeResponse(simulation.task_id,True,float(simulation.reward_info.reward),str(simulation.termination_reason),bridge.turn_count,bridge.turn_count,{"simulation_id":simulation.id,"duration_seconds":round(float(simulation.duration),3),"message_count":len(simulation.messages),"agent_cost":simulation.agent_cost,"user_cost":simulation.user_cost},final)

def create_app(runner: Callable[[EpisodeRequest], EpisodeResponse] | None = None):
    try: from fastapi import FastAPI, HTTPException
    except ImportError as exc: raise RuntimeError("FastAPI is required to host vita_server") from exc
    app=FastAPI(title="vita-rl runtime"); run=runner or run_vita_episode
    @app.get("/health")
    def health(): return {"status":"ok","supported":dict(SUPPORTED)}
    @app.post("/episode")
    def episode(payload: dict[str, Any]):
        try: return run(EpisodeRequest.from_dict(payload)).to_dict()
        except EpisodeValidationError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc
        except Exception as exc: raise HTTPException(status_code=500,detail="Vita episode execution failed") from exc
    return app

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--host",default="127.0.0.1"); parser.add_argument("--port",type=int,default=9010); args=parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(),host=args.host,port=args.port)
if __name__ == "__main__": main()
