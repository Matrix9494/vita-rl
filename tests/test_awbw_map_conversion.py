import json

from advancewars.engine import Game
from advancewars.utils.awbw_map import awbw_json_to_defendpeace_map


def test_awbw_map_json_converts_to_playable_defendpeace_text():
    payload = {
        "Name": "Tiny",
        "Size X": 3,
        "Size Y": 3,
        "Terrain Map": [
            [42, 39, 1],
            [15, 34, 15],
            [1, 44, 47],
        ],
        "Predeployed Units": [],
    }

    text = awbw_json_to_defendpeace_map(payload)
    state = Game(text, config="infantry_only", max_turns=2).reset(seed=0)

    assert state.map.width == 3
    assert state.map.height == 3
    assert sorted(state.players) == [0, 1]
    assert len(state.units) == 2
    assert state.map.tiles[0][0].terrain == "hq"
    assert state.map.tiles[2][2].terrain == "hq"


def test_awbw_map_json_seeds_on_factories_without_hq():
    payload = json.loads(
        json.dumps(
            {
                "Name": "Factory Only",
                "Size X": 2,
                "Size Y": 2,
                "Terrain Map": [[39, 1], [1, 44]],
                "Predeployed Units": [],
            }
        )
    )

    text = awbw_json_to_defendpeace_map(payload)
    state = Game(text, config="infantry_only", max_turns=2).reset(seed=0)

    assert sorted(state.players) == [0, 1]
    assert {(unit.owner, unit.coord.x, unit.coord.y) for unit in state.units.values()} == {
        (0, 0, 0),
        (1, 1, 1),
    }
