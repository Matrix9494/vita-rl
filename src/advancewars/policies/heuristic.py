"""Simple legal-action heuristic policy for PettingZoo-style rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from advancewars.engine.actions import Action, ActionType
from advancewars.engine.coordinates import Coord
from advancewars.engine.state import GameState, UnitState


@dataclass
class HeuristicPolicy:
    """A deterministic baseline that chooses one legal discrete action.

    The policy consumes PettingZoo observations and ``info`` entries. It uses the
    environment's current public game state only for scoring legal actions; it
    still commands the game through ``env.step(discrete_action)``.
    """

    build_priority: dict[str, int] = field(
        default_factory=lambda: {
            "tank": 95,
            "artillery": 80,
            "recon": 70,
            "mech": 65,
            "infantry": 60,
            "apc": 30,
        }
    )

    def choose_action(
        self,
        env: Any,
        agent: str,
        observation: dict[str, Any],
        info: dict[str, Any],
    ) -> int:
        del observation
        legal_actions = info.get("legal_actions", [])
        legal_indices = info.get("legal_action_indices")
        if legal_indices is None:
            legal_indices = [
                int(index)
                for index, value in enumerate(info["action_mask"])
                if int(value)
            ]
        if not legal_actions:
            return int(legal_indices[0])

        state = env.game.state
        if state is None:
            return int(legal_indices[0])
        player_id = int(agent.split("_", 1)[1])
        candidates = zip(legal_indices, legal_actions, strict=False)
        best_index, _best_action = max(
            candidates,
            key=lambda pair: (
                self._score_action(state, player_id, pair[1]),
                -int(pair[0]),
            ),
        )
        return int(best_index)

    def _score_action(
        self,
        state: GameState,
        player_id: int,
        action: Action,
    ) -> float:
        if action.type == ActionType.ATTACK:
            return self._score_attack(state, action)
        if action.type == ActionType.CAPTURE:
            return self._score_capture(state, player_id, action)
        if action.type == ActionType.BUILD:
            return 600 + self.build_priority.get(action.build_unit or "", 10)
        if action.type == ActionType.CO_ABILITY:
            return 800 if action.metadata.get("power") == "super" else 700
        if action.type == ActionType.LAUNCH:
            return 5200
        if action.type in {
            ActionType.REPAIR,
            ActionType.RESUPPLY,
            ActionType.JOIN,
            ActionType.UNLOAD,
            ActionType.LOAD,
        }:
            return 3500
        if action.type == ActionType.MOVE:
            return self._score_move(state, player_id, action)
        if action.type == ActionType.TRANSFORM:
            return 100
        if action.type == ActionType.WAIT:
            return -100
        if action.type == ActionType.END_TURN:
            return -1000
        if action.type == ActionType.DELETE:
            return -5000
        return 0

    def _score_attack(self, state: GameState, action: Action) -> float:
        if action.target is None:
            return 9000
        defender = state.unit_at(action.target)
        if defender is None:
            return 9000
        return 10000 + defender.definition.cost / 100 + (100 - defender.hp)

    def _score_capture(
        self,
        state: GameState,
        player_id: int,
        action: Action,
    ) -> float:
        unit = state.units[action.unit_id]
        tile = state.map.tile_at(unit.coord)
        score = 7600 + unit.hp / 10
        if tile.owner is None:
            score += 250
        elif tile.owner != player_id:
            score += 500
        if tile.terrain == "hq":
            score += 1500
        elif tile.terrain in {"factory", "airport", "seaport"}:
            score += 600
        elif tile.definition.profitable:
            score += 300
        return score

    def _score_move(
        self,
        state: GameState,
        player_id: int,
        action: Action,
    ) -> float:
        unit = state.units[action.unit_id]
        dest = action.path[-1]
        targets = self._targets_for_unit(state, player_id, unit)
        if not targets:
            return 1000 + len(action.path)

        score = max(
            bonus
            + (unit.coord.manhattan(target) - dest.manhattan(target)) * 120
            - dest.manhattan(target) * 12
            for target, bonus in targets
        )
        score -= len(action.path)
        if all(
            dest.manhattan(target) > unit.coord.manhattan(target)
            for target, _ in targets
        ):
            score -= 2000

        tile = state.map.tile_at(dest)
        if unit.definition.can_capture and tile.definition.capturable:
            if tile.owner != player_id:
                score += 2400
            if tile.terrain == "hq":
                score += 1200
            elif tile.terrain in {"factory", "airport", "seaport"}:
                score += 450

        if any(
            dest.manhattan(enemy.coord) == 1
            for enemy in state.map_units()
            if enemy.owner != player_id
        ):
            score += 100
        return score

    def _targets_for_unit(
        self,
        state: GameState,
        player_id: int,
        unit: UnitState,
    ) -> list[tuple[Coord, int]]:
        targets: list[tuple[Coord, int]] = []
        if unit.definition.can_capture:
            for y, row in enumerate(state.map.tiles):
                for x, tile in enumerate(row):
                    if not tile.definition.capturable or tile.owner == player_id:
                        continue
                    coord = Coord(x, y)
                    if tile.terrain == "hq" and tile.owner is not None:
                        targets.append((coord, 2800))
                    elif tile.owner is not None:
                        targets.append((coord, 1200))
                    elif tile.terrain in {"factory", "airport", "seaport"}:
                        targets.append((coord, 650))
                    elif tile.definition.profitable:
                        targets.append((coord, 250))
        for enemy in state.map_units():
            if enemy.owner != player_id:
                targets.append((enemy.coord, 900))
        return targets
