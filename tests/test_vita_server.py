import pytest
from vita_rl.vita_server import EpisodeRequest, EpisodeResponse, EpisodeValidationError

def payload(): return {"domain":"delivery","task_id":"10711001","language":"chinese","session_id":"s","instance_id":"i","agent_model":"proxy-model","dressage_proxy_url":"http://proxy/","sampling_params":{"temperature":0.2}}
def test_request_schema_round_trip():
    request=EpisodeRequest.from_dict(payload())
    assert request.dressage_proxy_url == "http://proxy/"
    assert request.to_dict()["sampling_params"]["temperature"] == 0.2
def test_environment_selection_is_validated():
    data=payload(); data["environment"]="vita-mini"; data["task_id"]="delivery_revision"
    request=EpisodeRequest.from_dict(data)
    assert request.environment == "vita-mini"
    assert request.environment_args == {"domain":"delivery", "language":"chinese"}
    data["environment"]="unknown"
    with pytest.raises(EpisodeValidationError): EpisodeRequest.from_dict(data)
def test_response_schema_round_trip():
    response=EpisodeResponse("10711001",True,1.0,"agent_stop",2,2,{"message_count":4},"done")
    assert EpisodeResponse.from_dict(response.to_dict()) == response
