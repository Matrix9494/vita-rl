import asyncio
from vita_rl.dressage_adapter import VitaRuntimeClient, VitaWhiteboxAgent
from vita_rl.vita_server import EpisodeRequest, EpisodeResponse

def test_runtime_client_preserves_payload():
    seen={}
    def post(url,payload):
        seen.update(url=url,payload=payload)
        return {"task_id":"10711001","completed":True,"reward":0.5,"termination_reason":"agent_stop","num_agent_turns":2,"proxy_turns":2,"simulation":{},"final_assistant_response":"done"}
    request=EpisodeRequest.from_dict({"domain":"delivery","task_id":"10711001","language":"chinese","session_id":"s","instance_id":"i","agent_model":"proxy-model","dressage_proxy_url":"http://proxy"})
    assert asyncio.run(VitaRuntimeClient("http://runtime",post=post).episode(request)).reward == 0.5
    assert seen["url"] == "http://runtime/episode" and seen["payload"]["session_id"] == "s"
def test_adapter_propagates_metadata(monkeypatch):
    class Sample:
        metadata={"vita_runtime_url":"http://runtime"}
    class Client:
        def __init__(self,url): assert url == "http://runtime"
        async def episode(self,request):
            assert request.session_id == "s"
            return EpisodeResponse("10711001",True,1.0,"agent_stop",3,3,{"message_count":6},"final")
    monkeypatch.setattr("vita_rl.dressage_adapter._proxy_url",lambda:"http://proxy")
    agent=VitaWhiteboxAgent(); agent.session_id="s"; agent.instance_id="i"; agent.runtime_client_factory=Client
    assert asyncio.run(agent.rollout(Sample(),{})) == "final"
    assert Sample.metadata["vita_reward"] == 1.0 and Sample.metadata["vita_num_agent_turns"] == 3
