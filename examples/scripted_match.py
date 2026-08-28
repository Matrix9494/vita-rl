"""Small engine-level scripted match."""

from __future__ import annotations

from advancewars.engine import Action, ActionType, Game
from advancewars.engine.coordinates import Coord


def main() -> None:
    game = Game.from_map("duel")
    state = game.reset()
    infantry = state.units[1]

    infantry.coord = Coord(3, 0)
    state, events = game.step(Action(ActionType.CAPTURE, unit_id=infantry.id))
    print(events)

    game.step(Action.end_turn())
    game.step(Action.end_turn())
    state, events = game.step(Action(ActionType.CAPTURE, unit_id=infantry.id))
    print(events)
    print(state.map.tile_at(Coord(3, 0)))


if __name__ == "__main__":
    main()
