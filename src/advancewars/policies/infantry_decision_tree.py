"""Decision-tree baseline for infantry-only Advance Wars games."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from advancewars.engine.actions import ActionType
from advancewars.engine.coordinates import Coord
from advancewars.engine.semantic_actions import SemanticAction
from advancewars.engine.state import GameState, UnitState
from advancewars.env.structured_action_codec import ACTION_KINDS


@dataclass
class InfantryDecisionTreePolicy:
    """A stronger deterministic baseline for infantry-only self-play.

    The policy is intentionally written as a decision tree over legal semantic
    actions. It assumes infantry are the only producible unit, but it can still
    reason over existing map units.
    """

    min_funds_to_build: int = 1000
    attack_base: float = 9000
    attack_damage_weight: float = 14
    attack_cost_weight: float = 0.1
    attack_kill_bonus: float = 2600
    capture_threat_bonus: float = 900
    hq_threat_bonus: float = 2000
    capture_base: float = 7600
    capture_hp_weight: float = 1
    capture_hq_bonus: float = 4500
    capture_factory_bonus: float = 1800
    capture_property_bonus: float = 900
    capture_enemy_property_bonus: float = 1200
    build_base: float = 5200
    build_unit_deficit_weight: float = 120
    build_pressure_penalty: float = 10
    move_progress_weight: float = 180
    move_distance_penalty: float = 20
    move_capture_bonus: float = 1400
    move_hq_bonus: float = 2500
    move_factory_bonus: float = 900
    adjacent_enemy_penalty: float = 350
    nearby_enemy_penalty: float = 80
    wait_penalty: float = 50
    end_turn_score: float = -2000

    def choose_action(
        self,
        env: Any,
        agent: str,
        observation: dict[str, Any],
        info: dict[str, Any],
    ):
        del observation
        state = env.game.state
        if state is None:
            raise RuntimeError("Environment has not been reset.")
        player_id = int(agent.split("_", 1)[1])
        if "legal_semantic_actions" in info:
            semantic_actions = [
                self._semantic_from_info(action)
                for action in info["legal_semantic_actions"]
            ]
            structured_actions = info["legal_structured_actions"]
            best_position, _best_semantic = max(
                enumerate(semantic_actions),
                key=lambda item: (
                    self._score_semantic(state, player_id, item[1]),
                    -item[0],
                ),
            )
            return structured_actions[best_position]
        return self._choose_flat_action(state, player_id, info)

    def _choose_flat_action(
        self,
        state: GameState,
        player_id: int,
        info: dict[str, Any],
    ) -> int:
        legal_actions = info.get("legal_actions", [])
        legal_indices = info.get("legal_action_indices", [])
        semantic_actions = [
            self._legacy_action_to_semantic(state, action) for action in legal_actions
        ]
        best_position, _best_semantic = max(
            enumerate(semantic_actions),
            key=lambda item: (
                self._score_semantic(state, player_id, item[1]),
                -item[0],
            ),
        )
        return int(legal_indices[best_position])

    def _score_semantic(
        self,
        state: GameState,
        player_id: int,
        action: SemanticAction,
    ) -> float:
        if action.kind == ActionType.ATTACK:
            return self._score_attack(state, player_id, action)
        if action.kind == ActionType.CAPTURE:
            return self._score_capture(state, player_id, action)
        if action.kind == ActionType.BUILD:
            return self._score_build(state, player_id, action)
        if action.kind in {ActionType.WAIT, ActionType.MOVE}:
            return self._score_movement(state, player_id, action)
        if action.kind == ActionType.CO_ABILITY:
            return 600
        if action.kind == ActionType.END_TURN:
            return self.end_turn_score
        if action.kind in {ActionType.DELETE, ActionType.TRANSFORM}:
            return -5000
        return 0

    def _score_attack(
        self,
        state: GameState,
        player_id: int,
        action: SemanticAction,
    ) -> float:
        if action.target is None:
            return -1000
        defender = state.unit_at(action.target)
        if defender is None or defender.owner == player_id:
            return -1000
        attacker = self._unit_at_source(state, action)
        if attacker is None:
            return -1000

        damage_estimate = self._infantry_damage_estimate(attacker, defender)
        kill_bonus = self.attack_kill_bonus if damage_estimate >= defender.hp else 0
        capture_threat_bonus = 0
        defender_tile = state.map.tile_at(defender.coord)
        if defender.definition.can_capture and defender_tile.owner == player_id:
            capture_threat_bonus += self.capture_threat_bonus
        if defender_tile.terrain == "hq" and defender_tile.owner == player_id:
            capture_threat_bonus += self.hq_threat_bonus
        return (
            self.attack_base
            + kill_bonus
            + capture_threat_bonus
            + damage_estimate * self.attack_damage_weight
            + defender.definition.cost * self.attack_cost_weight
            - self._danger_penalty(state, player_id, action.destination)
        )

    def _score_capture(
        self,
        state: GameState,
        player_id: int,
        action: SemanticAction,
    ) -> float:
        destination = action.destination
        if destination is None:
            return -1000
        unit = self._unit_at_source(state, action)
        if unit is None:
            return -1000
        tile = state.map.tile_at(destination)
        score = self.capture_base + unit.hp * self.capture_hp_weight
        if tile.terrain == "hq":
            score += self.capture_hq_bonus
        elif tile.terrain == "factory":
            score += self.capture_factory_bonus
        elif tile.definition.profitable:
            score += self.capture_property_bonus
        if tile.owner is not None and tile.owner != player_id:
            score += self.capture_enemy_property_bonus
        score -= self._danger_penalty(state, player_id, destination)
        return score

    def _score_build(
        self,
        state: GameState,
        player_id: int,
        action: SemanticAction,
    ) -> float:
        if action.payload != "infantry" or action.target is None:
            return -1000
        player = state.players[player_id]
        if player.funds < self.min_funds_to_build:
            return -1000
        friendly_units = len(state.living_units(player_id))
        enemy_units = len(
            [unit for unit in state.living_units() if unit.owner != player_id]
        )
        target_pressure = self._nearest_enemy_distance(state, player_id, action.target)
        return (
            self.build_base
            + (enemy_units - friendly_units) * self.build_unit_deficit_weight
            - target_pressure * self.build_pressure_penalty
        )

    def _score_movement(
        self,
        state: GameState,
        player_id: int,
        action: SemanticAction,
    ) -> float:
        source = action.source
        destination = action.destination
        if source is None or destination is None:
            return -1000
        unit = state.unit_at(source)
        if unit is None:
            return -1000

        targets = self._strategic_targets(state, player_id, unit)
        if not targets:
            return -100
        best = max(
            bonus
            + (source.manhattan(target) - destination.manhattan(target))
            * self.move_progress_weight
            - destination.manhattan(target) * self.move_distance_penalty
            for target, bonus in targets
        )
        tile = state.map.tile_at(destination)
        if unit.definition.can_capture and tile.definition.capturable:
            if tile.owner != player_id:
                best += self.move_capture_bonus
            if tile.terrain == "hq":
                best += self.move_hq_bonus
            elif tile.terrain == "factory":
                best += self.move_factory_bonus
        if action.kind == ActionType.WAIT:
            best -= self.wait_penalty
        best -= self._danger_penalty(state, player_id, destination)
        return best

    def _strategic_targets(
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
                        targets.append((coord, 5000))
                    elif tile.terrain == "factory":
                        targets.append((coord, 2200))
                    elif tile.definition.profitable:
                        targets.append((coord, 1100))
        for enemy in state.map_units():
            if enemy.owner != player_id:
                targets.append((enemy.coord, 1500))
        return targets

    def _danger_penalty(
        self,
        state: GameState,
        player_id: int,
        coord: Coord | None,
    ) -> float:
        if coord is None:
            return 0
        adjacent_enemies = sum(
            1
            for enemy in state.map_units()
            if enemy.owner != player_id and enemy.coord.manhattan(coord) <= 1
        )
        nearby_enemies = sum(
            1
            for enemy in state.map_units()
            if enemy.owner != player_id and enemy.coord.manhattan(coord) == 2
        )
        return (
            adjacent_enemies * self.adjacent_enemy_penalty
            + nearby_enemies * self.nearby_enemy_penalty
        )

    def _nearest_enemy_distance(
        self,
        state: GameState,
        player_id: int,
        coord: Coord,
    ) -> int:
        enemies = [unit for unit in state.map_units() if unit.owner != player_id]
        if not enemies:
            return 0
        return min(coord.manhattan(enemy.coord) for enemy in enemies)

    def _infantry_damage_estimate(
        self,
        attacker: UnitState,
        defender: UnitState,
    ) -> int:
        weapon = next(
            (
                weapon
                for weapon in attacker.definition.weapons
                if defender.unit_type in weapon.damage
            ),
            None,
        )
        if weapon is None:
            return 0
        return weapon.damage[defender.unit_type] * attacker.hp // 100

    def _unit_at_source(
        self,
        state: GameState,
        action: SemanticAction,
    ) -> UnitState | None:
        if action.source is None:
            return None
        return state.unit_at(action.source)

    def _semantic_from_info(self, payload: dict[str, Any]) -> SemanticAction:
        return SemanticAction(
            kind=ActionType(payload["kind"]),
            source=self._coord_from_payload(payload["source"]),
            destination=self._coord_from_payload(payload["destination"]),
            target=self._coord_from_payload(payload["target"]),
            payload=payload["payload"],
            metadata=dict(payload.get("metadata", {})),
        )

    def _legacy_action_to_semantic(
        self,
        state: GameState,
        action,
    ) -> SemanticAction:
        source = None
        destination = None
        if action.unit_id is not None:
            unit = state.units[action.unit_id]
            source = unit.coord
            destination = unit.coord
        if action.type == ActionType.MOVE and action.path:
            destination = action.path[-1]
        payload = None
        if action.type == ActionType.BUILD:
            payload = action.build_unit
        elif action.type == ActionType.CO_ABILITY:
            payload = action.metadata.get("power", "power")
        return SemanticAction(action.type, source, destination, action.target, payload)

    @staticmethod
    def _coord_from_payload(payload: dict[str, int] | None) -> Coord | None:
        if payload is None:
            return None
        return Coord(payload["x"], payload["y"])
