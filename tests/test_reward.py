from vita_rl.reward import compute_reward
class Sample:
    def __init__(self,metadata): self.metadata=metadata
def test_reward_reads_vita_metadata():
    assert compute_reward(Sample({"vita_reward":"0.75"})) == 0.75
    assert compute_reward(Sample({})) == 0.0
