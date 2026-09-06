from vita_rl.reward import compute_reward, compute_smoke_dense_reward
class Sample:
    def __init__(self,metadata): self.metadata=metadata
def test_reward_reads_vita_metadata():
    assert compute_reward(Sample({"vita_reward":"0.75"})) == 0.75
    assert compute_reward(Sample({})) == 0.0


def test_smoke_reward_uses_full_trajectory_assistant_content():
    short = Sample({
        "vita_num_agent_turns": 2,
        "messages": [{"role": "assistant", "content": "hi"}],
    })
    long = Sample({
        "vita_num_agent_turns": 2,
        "messages": [{"role": "assistant", "content": "hello world"}],
    })
    assert compute_smoke_dense_reward(long) > compute_smoke_dense_reward(short)
