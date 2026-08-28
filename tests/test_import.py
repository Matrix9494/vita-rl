from advancewars import env, raw_env
from advancewars.engine import Action, ActionType, Game, load_map, load_ruleset


def test_public_imports():
    assert callable(env)
    assert callable(raw_env)
    assert Action(ActionType.END_TURN).type is ActionType.END_TURN
    assert Game.from_map("duel").reset().current_player == 0
    assert "0HQ" in load_map("duel")
    assert "0HQ" in load_map("duel.map")
    assert load_ruleset("defendpeace_awbw")["income_per_city"] == 1000
