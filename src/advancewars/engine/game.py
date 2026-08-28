"""Core deterministic game engine."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import replace
import random
from collections.abc import Mapping, Sequence

from advancewars.engine.actions import Action, ActionType
from advancewars.engine.commanders import (
    CommanderDef,
    attack_bonus,
    can_build_from_city,
    counter_damage_multiplier,
    capture_multiplier,
    commander_for,
    defense_bonus,
    has_perfect_movement,
    idle_fuel_burn,
    income_per_property,
    luck_range,
    max_power_cost,
    move_bonus,
    power_cost,
    power_heal,
    power_name,
    preemptive_counter,
    repair_power,
    unit_cost,
    vision_bonus,
    weapon_range,
    weather_for_movement,
)
from advancewars.engine.config import GameConfig, load_config
from advancewars.engine.coordinates import Coord
from advancewars.engine.data import (
    BUILD_LISTS,
    MOVEMENT_COSTS_BY_WEATHER,
    RULESETS,
    UNITS,
    WEATHERS,
    UnitDef,
    WeaponDef,
)
from advancewars.engine.state import Event, GameState, PlayerState, UnitState
from advancewars.engine.loaders import load_map
from advancewars.engine.semantic_actions import SemanticAction, action_to_semantic
from advancewars.utils.defendpeace_map import parse_defendpeace_map
from advancewars.utils.serialization import game_state_from_dict, game_state_to_dict


HIDDEN_UNIT_TYPES = {"stealth_hide", "sub_sub"}


class Game:
    """Advance Wars-style rules engine for one match."""

    def __init__(
        self,
        map_text: str,
        ruleset: str = "defendpeace_awbw",
        max_turns: int = 100,
        fog: bool = False,
        weather: str = "clear",
        commanders: (
            str
            | Sequence[str | Sequence[str]]
            | Mapping[int | str, str | Sequence[str]]
            | None
        ) = None,
        tag_mode: bool = False,
        luck: bool = False,
        seed: int | None = None,
        config: str | Mapping[str, object] | GameConfig | None = None,
        enabled_units: Sequence[str] | None = None,
        strict_units: bool | None = None,
    ):
        if ruleset not in RULESETS:
            raise ValueError(f"Unknown ruleset: {ruleset}")
        if weather not in WEATHERS:
            raise ValueError(f"Unknown weather: {weather}")
        self.map_text = map_text
        self.ruleset_name = ruleset
        self.rules = RULESETS[ruleset]
        self.max_turns = max_turns
        self.fog = fog
        self.initial_weather = weather
        self.commanders = commanders
        self.tag_mode = tag_mode
        self.luck = luck
        self.seed = seed
        self.config = load_config(config).with_overrides(
            enabled_units=enabled_units,
            strict_units=strict_units,
        )
        self._rng = random.Random(seed)
        self.state: GameState | None = None
        self._last_events: list[Event] = []

    @classmethod
    def from_map(
        cls,
        map_name: str = "duel",
        ruleset: str = "defendpeace_awbw",
        **options,
    ) -> Game:
        return cls(load_map(map_name), ruleset=ruleset, **options)

    def reset(self, seed: int | None = None) -> GameState:
        if seed is not None:
            self.seed = seed
            self._rng.seed(seed)
        parsed = parse_defendpeace_map(self.map_text)
        commander_teams = self._resolve_commanders(parsed.player_ids)
        players = {
            player_id: PlayerState(
                id=player_id,
                funds=int(self.rules["starting_funds"]),
                commander=commander_teams[player_id][0],
                commanders=commander_teams[player_id],
            )
            for player_id in parsed.player_ids
        }
        for unit in parsed.units.values():
            unit_def = unit.definition
            unit.fuel = unit_def.max_fuel
            unit.ammo = unit_def.max_ammo_by_weapon
        self._validate_initial_unit_pool(parsed.units.values())
        self.state = GameState(
            map=parsed.map_state,
            players=players,
            units=parsed.units,
            current_player=min(players),
            turn=1,
            weather=self.initial_weather,
            weather_base=self.initial_weather,
        )
        self._start_turn(self.state.current_player, collect_income=False)
        return self.state

    @property
    def last_events(self) -> list[Event]:
        return list(self._last_events)

    def _unit_enabled(self, unit_type: str) -> bool:
        return self.config.unit_enabled(unit_type)

    def _validate_initial_unit_pool(self, units: Sequence[UnitState]) -> None:
        if not self.config.strict_units:
            return
        disabled = sorted(
            unit.unit_type
            for unit in units
            if not self._unit_enabled(unit.unit_type)
        )
        if disabled:
            raise ValueError(
                "Map contains unit type(s) disabled by config: "
                + ", ".join(disabled)
            )

    def _encoded_rng_state(self):
        return self._rng.getstate()

    @staticmethod
    def _decoded_rng_state(payload):
        if isinstance(payload, list):
            return tuple(Game._decoded_rng_state(item) for item in payload)
        return payload

    def clone(self) -> Game:
        other = Game(
            self.map_text,
            self.ruleset_name,
            self.max_turns,
            fog=self.fog,
            weather=self.initial_weather,
            commanders=self.commanders,
            tag_mode=self.tag_mode,
            luck=self.luck,
            seed=self.seed,
            config=self.config,
        )
        other.state = deepcopy(self.state)
        other._last_events = deepcopy(self._last_events)
        other._rng.setstate(self._rng.getstate())
        return other

    def to_json(self) -> dict:
        return {
            "map_text": self.map_text,
            "ruleset": self.ruleset_name,
            "max_turns": self.max_turns,
            "fog": self.fog,
            "weather": self.initial_weather,
            "commanders": self.commanders,
            "tag_mode": self.tag_mode,
            "luck": self.luck,
            "seed": self.seed,
            "config": self.config.to_json(),
            "rng_state": self._encoded_rng_state(),
            "state": None if self.state is None else game_state_to_dict(self.state),
        }

    @classmethod
    def from_json(cls, payload: dict) -> Game:
        game = cls(
            payload["map_text"],
            ruleset=payload["ruleset"],
            max_turns=int(payload["max_turns"]),
            fog=bool(payload["fog"]),
            weather=payload.get("weather", "clear"),
            commanders=payload.get("commanders"),
            tag_mode=bool(payload.get("tag_mode", False)),
            luck=bool(payload.get("luck", False)),
            seed=payload.get("seed"),
            config=payload.get("config"),
        )
        if payload.get("rng_state") is not None:
            game._rng.setstate(cls._decoded_rng_state(payload["rng_state"]))
        if payload.get("state") is not None:
            game.state = game_state_from_dict(payload["state"])
        return game

    def step(self, action: Action) -> tuple[GameState, list[Event]]:
        state = self._require_state()
        if state.done or state.truncated:
            self._last_events = []
            return state, []
        events: list[Event] = []
        if action.type == ActionType.END_TURN:
            events.extend(self._end_turn())
        elif action.type == ActionType.WAIT:
            events.extend(self._wait(action))
        elif action.type == ActionType.CAPTURE:
            events.extend(self._capture(action))
        elif action.type == ActionType.BUILD:
            events.extend(self._build(action))
        elif action.type == ActionType.MOVE:
            events.extend(self._move(action))
        elif action.type == ActionType.ATTACK:
            events.extend(self._attack(action))
        elif action.type == ActionType.JOIN:
            events.extend(self._join(action))
        elif action.type == ActionType.LOAD:
            events.extend(self._load(action))
        elif action.type == ActionType.UNLOAD:
            events.extend(self._unload(action))
        elif action.type == ActionType.LAUNCH:
            events.extend(self._launch(action))
        elif action.type == ActionType.RESUPPLY:
            events.extend(self._resupply(action))
        elif action.type == ActionType.REPAIR:
            events.extend(self._repair(action))
        elif action.type == ActionType.DELETE:
            events.extend(self._delete(action))
        elif action.type == ActionType.TRANSFORM:
            events.extend(self._transform(action))
        elif action.type == ActionType.CO_ABILITY:
            events.extend(self._co_ability(action))
        elif action.type == ActionType.SWAP_CO:
            events.extend(self._swap_co(action))
        else:
            raise NotImplementedError(f"Action not implemented yet: {action.type}")
        return self._finish_step(events)

    def step_semantic(self, action: SemanticAction) -> tuple[GameState, list[Event]]:
        """Execute a structured complete action.

        This is the engine entry point for the structured training interface.
        It supports moving to a destination and then applying the terminal
        action for core unit actions.
        """
        state = self._require_state()
        if state.done or state.truncated:
            self._last_events = []
            return state, []
        if action not in self.legal_semantic_actions():
            raise ValueError(f"Illegal semantic action: {action}")

        if action.kind == ActionType.END_TURN:
            return self.step(Action.end_turn())
        if action.kind == ActionType.CO_ABILITY:
            metadata = {}
            if action.payload == "super":
                metadata["power"] = "super"
            return self.step(Action(ActionType.CO_ABILITY, metadata=metadata))
        if action.kind == ActionType.SWAP_CO:
            return self.step(Action(ActionType.SWAP_CO))
        if action.kind == ActionType.BUILD:
            return self.step(
                Action(
                    ActionType.BUILD,
                    target=action.target,
                    build_unit=str(action.payload),
                )
            )

        if action.source is None:
            raise ValueError("Unit semantic action requires source.")
        unit = state.unit_at(action.source)
        if unit is None:
            raise ValueError(f"No unit at semantic action source: {action.source}")
        destination = action.destination or action.source
        events: list[Event] = []
        if destination != unit.coord:
            events.extend(self._relocate_unit(unit, (destination,)))

        terminal = Action(
            action.kind,
            unit_id=unit.id,
            target=action.target,
            metadata=dict(action.metadata),
        )
        if action.kind == ActionType.MOVE:
            unit.can_act = False
        elif action.kind == ActionType.WAIT:
            events.extend(self._wait(terminal))
        elif action.kind == ActionType.CAPTURE:
            events.extend(self._capture(terminal))
        elif action.kind == ActionType.ATTACK:
            terminal.metadata["after_moving"] = destination != action.source
            events.extend(self._attack(terminal))
        elif action.kind == ActionType.DELETE:
            events.extend(self._delete(terminal))
        elif action.kind == ActionType.TRANSFORM:
            events.extend(self._transform(terminal))
        elif action.kind == ActionType.LOAD:
            events.extend(self._load(terminal))
        elif action.kind == ActionType.JOIN:
            events.extend(self._join(terminal))
        elif action.kind == ActionType.RESUPPLY:
            events.extend(self._resupply(terminal))
        elif action.kind == ActionType.REPAIR:
            events.extend(self._repair(terminal))
        elif action.kind == ActionType.LAUNCH:
            if isinstance(action.payload, int):
                terminal.metadata["cargo_unit_id"] = action.payload
            events.extend(self._launch(terminal))
        elif action.kind == ActionType.UNLOAD:
            if isinstance(action.payload, int):
                terminal.metadata["cargo_unit_id"] = action.payload
            events.extend(self._unload(terminal))
        else:
            raise NotImplementedError(f"Semantic action not implemented: {action.kind}")
        return self._finish_step(events)

    def _finish_step(self, events: list[Event]) -> tuple[GameState, list[Event]]:
        state = self._require_state()
        self._award_power_charge(events)
        self._check_victory()
        if not state.done and state.turn > self.max_turns:
            state.truncated = True
        if not state.done and not state.truncated:
            self._update_fog_memory(state.current_player)
        self._last_events = events
        return state, events

    def legal_semantic_actions(
        self,
        player_id: int | None = None,
    ) -> list[SemanticAction]:
        state = self._require_state()
        if state.done or state.truncated:
            return []
        player_id = state.current_player if player_id is None else player_id
        if player_id != state.current_player:
            return []

        semantic_actions: list[SemanticAction] = []
        current_legal = self.legal_actions(player_id)
        for action in current_legal:
            if action.unit_id is None or action.type == ActionType.BUILD:
                semantic_actions.append(action_to_semantic(action, state))

        for unit in state.map_units(player_id):
            if not unit.can_act:
                continue
            source = unit.coord
            for destination in self.reachable_destinations(unit):
                after_moving = destination != source
                semantic_actions.append(
                    SemanticAction(ActionType.WAIT, source, destination)
                )
                if after_moving:
                    semantic_actions.append(
                        SemanticAction(ActionType.MOVE, source, destination)
                    )
                tile = state.map.tile_at(destination)
                if (
                    unit.definition.can_capture
                    and tile.definition.capturable
                    and tile.owner != player_id
                ):
                    semantic_actions.append(
                        SemanticAction(ActionType.CAPTURE, source, destination)
                    )
                for target in self._attack_targets_from(
                    unit,
                    destination,
                    after_moving=after_moving,
                ):
                    semantic_actions.append(
                        SemanticAction(
                            ActionType.ATTACK,
                            source,
                            destination,
                            target.coord,
                        )
                    )
                semantic_actions.extend(
                    self._destination_terminal_semantic_actions(
                        unit,
                        source,
                        destination,
                    )
                )
        return semantic_actions

    def _destination_terminal_semantic_actions(
        self,
        unit: UnitState,
        source: Coord,
        destination: Coord,
    ) -> list[SemanticAction]:
        actions: list[SemanticAction] = []
        original = unit.coord
        unit.coord = destination
        try:
            if self._can_delete(unit):
                actions.append(SemanticAction(ActionType.DELETE, source, destination))
            if unit.definition.transform_to is not None:
                actions.append(
                    SemanticAction(ActionType.TRANSFORM, source, destination)
                )
            if self._can_launch_silo(unit):
                for target in self._launch_targets():
                    actions.append(
                        SemanticAction(
                            ActionType.LAUNCH,
                            source,
                            destination,
                            target,
                        )
                    )
            for cargo_id, target in self.launch_options(unit):
                actions.append(
                    SemanticAction(
                        ActionType.LAUNCH,
                        source,
                        destination,
                        target,
                        payload=cargo_id,
                    )
                )
            for target in self.load_targets(unit):
                actions.append(
                    SemanticAction(
                        ActionType.LOAD,
                        source,
                        destination,
                        target.coord,
                    )
                )
            for target in self.join_targets(unit):
                actions.append(
                    SemanticAction(
                        ActionType.JOIN,
                        source,
                        destination,
                        target.coord,
                    )
                )
            for target in self.resupply_targets(unit):
                actions.append(
                    SemanticAction(
                        ActionType.RESUPPLY,
                        source,
                        destination,
                        target.coord,
                    )
                )
            for target in self.repair_targets(unit):
                actions.append(
                    SemanticAction(
                        ActionType.REPAIR,
                        source,
                        destination,
                        target.coord,
                    )
                )
            for cargo_id, target in self.unload_options(unit):
                actions.append(
                    SemanticAction(
                        ActionType.UNLOAD,
                        source,
                        destination,
                        target,
                        payload=cargo_id,
                    )
                )
        finally:
            unit.coord = original
        return actions

    def legal_actions(self, player_id: int | None = None) -> list[Action]:
        state = self._require_state()
        if state.done or state.truncated:
            return []
        player_id = state.current_player if player_id is None else player_id
        if player_id != state.current_player:
            return []
        actions = [Action.end_turn()]
        if self._can_activate_power(player_id, "power"):
            actions.append(Action(ActionType.CO_ABILITY))
        if self._can_activate_power(player_id, "super"):
            actions.append(Action(ActionType.CO_ABILITY, metadata={"power": "super"}))
        if self._can_swap_co(player_id):
            actions.append(Action(ActionType.SWAP_CO))
        for unit in state.map_units(player_id):
            if not unit.can_act:
                continue
            actions.append(Action(ActionType.WAIT, unit_id=unit.id))
            if self._can_delete(unit):
                actions.append(Action(ActionType.DELETE, unit_id=unit.id))
            if unit.definition.transform_to is not None:
                actions.append(Action(ActionType.TRANSFORM, unit_id=unit.id))
            if self._can_launch_silo(unit):
                for target in self._launch_targets():
                    actions.append(Action(ActionType.LAUNCH, unit_id=unit.id, target=target))
            for cargo_id, dest in self.launch_options(unit):
                actions.append(
                    Action(
                        ActionType.LAUNCH,
                        unit_id=unit.id,
                        target=dest,
                        metadata={"cargo_unit_id": cargo_id},
                    )
                )
            tile = state.map.tile_at(unit.coord)
            if unit.definition.can_capture and tile.definition.capturable:
                if tile.owner != player_id:
                    actions.append(Action(ActionType.CAPTURE, unit_id=unit.id))
            for transport in self.load_targets(unit):
                actions.append(Action(ActionType.LOAD, unit_id=unit.id, target=transport.coord))
            for target in self.join_targets(unit):
                actions.append(Action(ActionType.JOIN, unit_id=unit.id, target=target.coord))
            for target in self.resupply_targets(unit):
                actions.append(Action(ActionType.RESUPPLY, unit_id=unit.id, target=target.coord))
            for target in self.repair_targets(unit):
                actions.append(Action(ActionType.REPAIR, unit_id=unit.id, target=target.coord))
            for cargo_id, dest in self.unload_options(unit):
                actions.append(
                    Action(
                        ActionType.UNLOAD,
                        unit_id=unit.id,
                        target=dest,
                        metadata={"cargo_unit_id": cargo_id},
                    )
                )
            for dest, path in self.reachable_paths(unit).items():
                if dest != unit.coord:
                    actions.append(Action(ActionType.MOVE, unit_id=unit.id, path=path))
            for target in self.attack_targets(unit, after_moving=False):
                actions.append(Action(ActionType.ATTACK, unit_id=unit.id, target=target.coord))
        actions.extend(self._legal_build_actions(player_id))
        return actions

    def reachable_destinations(self, unit: UnitState) -> set[Coord]:
        return set(self._reachable_costs(unit))

    def reachable_paths(self, unit: UnitState) -> dict[Coord, tuple[Coord, ...]]:
        _costs, paths = self._reachable_costs_and_paths(unit)
        return paths

    def _reachable_costs(self, unit: UnitState) -> dict[Coord, int]:
        costs, _paths = self._reachable_costs_and_paths(unit)
        return costs

    def _reachable_costs_and_paths(
        self, unit: UnitState
    ) -> tuple[dict[Coord, int], dict[Coord, tuple[Coord, ...]]]:
        state = self._require_state()
        costs = self._movement_costs(unit)
        start = unit.coord
        seen = {start: 0}
        paths: dict[Coord, tuple[Coord, ...]] = {start: ()}
        queue = deque([start])
        fuel_budget = unit.fuel if unit.fuel is not None else unit.definition.max_fuel
        while queue:
            coord = queue.popleft()
            spent = seen[coord]
            for nxt in coord.neighbors():
                if not state.map.in_bounds(nxt):
                    continue
                occupant = state.unit_at(nxt)
                if occupant is not None and occupant.owner != unit.owner:
                    continue
                terrain = state.map.tile_at(nxt).terrain
                cost = costs.get(terrain)
                if cost is None:
                    continue
                new_spent = spent + int(cost)
                if new_spent > self._move_power(unit):
                    continue
                if new_spent * unit.definition.fuel_burn_per_tile > fuel_budget:
                    continue
                if occupant is not None and occupant.owner == unit.owner and nxt != start:
                    continue
                if nxt not in seen or new_spent < seen[nxt]:
                    seen[nxt] = new_spent
                    paths[nxt] = (*paths[coord], nxt)
                    queue.append(nxt)
        return seen, paths

    def attack_targets(self, unit: UnitState, after_moving: bool) -> list[UnitState]:
        return self._attack_targets_from(unit, unit.coord, after_moving)

    def _attack_targets_from(
        self,
        unit: UnitState,
        coord: Coord,
        after_moving: bool,
    ) -> list[UnitState]:
        state = self._require_state()
        targets: list[UnitState] = []
        for other in state.map_units():
            if other.owner == unit.owner:
                continue
            if self.fog and not self.is_unit_visible(unit.owner, other):
                continue
            distance = coord.manhattan(other.coord)
            if self._choose_weapon(unit, other.definition, distance, after_moving):
                targets.append(other)
        return targets

    def visible_coords(self, player_id: int) -> set[Coord]:
        state = self._require_state()
        if not self.fog:
            return {
                Coord(x, y)
                for y in range(state.map.height)
                for x in range(state.map.width)
            }

        return self._current_visible_coords(player_id) | state.fog_visible_coords.get(
            player_id, set()
        )

    def _current_visible_coords(self, player_id: int) -> set[Coord]:
        state = self._require_state()
        visible: set[Coord] = set()
        for unit in state.map_units(player_id):
            tile = state.map.tile_at(unit.coord)
            radius = self._vision_range(unit, tile.definition.vision_boost)
            for y in range(unit.coord.y - radius, unit.coord.y + radius + 1):
                for x in range(unit.coord.x - radius, unit.coord.x + radius + 1):
                    coord = Coord(x, y)
                    if not state.map.in_bounds(coord):
                        continue
                    if unit.coord.manhattan(coord) <= radius:
                        visible.add(coord)
        return visible

    def is_unit_visible(self, player_id: int, unit: UnitState) -> bool:
        if unit.owner == player_id:
            return unit.hp > 0 and unit.carried_by is None
        if unit.hp <= 0 or unit.carried_by is not None:
            return False
        if not self.fog:
            return True
        if unit.id in self._require_state().fog_visible_unit_ids.get(player_id, set()):
            return True
        return unit.id in self._current_visible_unit_ids(player_id)

    def _current_visible_unit_ids(self, player_id: int) -> set[int]:
        state = self._require_state()
        visible_coords = self._current_visible_coords(player_id)
        visible_unit_ids: set[int] = set()
        for unit in state.map_units():
            if unit.owner == player_id:
                visible_unit_ids.add(unit.id)
                continue
            if unit.hp <= 0 or unit.carried_by is not None:
                continue
            if unit.unit_type in HIDDEN_UNIT_TYPES:
                if self._has_adjacent_viewer(player_id, unit):
                    visible_unit_ids.add(unit.id)
                continue
            tile = state.map.tile_at(unit.coord)
            if tile.definition.provides_cover and unit.definition.move_type != "flight":
                if self._has_adjacent_viewer(player_id, unit):
                    visible_unit_ids.add(unit.id)
                continue
            if unit.coord in visible_coords:
                visible_unit_ids.add(unit.id)
        return visible_unit_ids

    def _update_fog_memory(self, player_id: int) -> None:
        if not self.fog:
            return
        state = self._require_state()
        state.fog_visible_coords.setdefault(player_id, set()).update(
            self._current_visible_coords(player_id)
        )
        state.fog_visible_unit_ids.setdefault(player_id, set()).update(
            self._current_visible_unit_ids(player_id)
        )

    def _reset_fog_memory(self, player_id: int) -> None:
        if not self.fog:
            return
        state = self._require_state()
        state.fog_visible_coords[player_id] = set()
        state.fog_visible_unit_ids[player_id] = set()
        self._update_fog_memory(player_id)

    def observed_units(self, player_id: int) -> list[UnitState]:
        state = self._require_state()
        return [
            unit
            for unit in state.map_units()
            if self.is_unit_visible(player_id, unit)
        ]

    def _has_adjacent_viewer(self, player_id: int, unit: UnitState) -> bool:
        return any(
            viewer.coord.manhattan(unit.coord) <= 1
            for viewer in self._require_state().map_units(player_id)
        )

    def load_targets(self, unit: UnitState) -> list[UnitState]:
        state = self._require_state()
        if unit.carried_by is not None:
            return []
        targets = []
        for transport in state.map_units(unit.owner):
            if transport.id == unit.id:
                continue
            if unit.coord.manhattan(transport.coord) > 1:
                continue
            if self._can_load(transport, unit):
                targets.append(transport)
        return targets

    def join_targets(self, unit: UnitState) -> list[UnitState]:
        state = self._require_state()
        if unit.carried_by is not None or unit.cargo:
            return []
        targets = []
        for other in state.map_units(unit.owner):
            if other.id == unit.id:
                continue
            if other.carried_by is not None or other.cargo:
                continue
            if other.unit_type != unit.unit_type:
                continue
            if other.hp >= other.definition.max_hp:
                continue
            if unit.coord.manhattan(other.coord) <= 1:
                targets.append(other)
        return targets

    def resupply_targets(self, supplier: UnitState) -> list[UnitState]:
        if not self._can_resupply_adjacent(supplier):
            return []
        return [
            unit
            for unit in self._adjacent_allied_units(supplier)
            if self._needs_resupply(unit)
        ]

    def repair_targets(self, repairer: UnitState) -> list[UnitState]:
        if repairer.unit_type != "bboat":
            return []
        state = self._require_state()
        player = state.players[repairer.owner]
        targets = []
        for unit in self._adjacent_allied_units(repairer):
            if unit.hp >= unit.definition.max_hp:
                continue
            if player.funds <= 0:
                continue
            targets.append(unit)
        return targets

    def unload_options(self, transport: UnitState) -> list[tuple[int, Coord]]:
        state = self._require_state()
        if not transport.cargo:
            return []
        options: list[tuple[int, Coord]] = []
        for cargo_id in transport.cargo:
            cargo = state.units[cargo_id]
            for dest in transport.coord.neighbors():
                if not state.map.in_bounds(dest) or state.unit_at(dest) is not None:
                    continue
                if self._can_stand_on(cargo, dest):
                    options.append((cargo_id, dest))
        return options

    def launch_options(self, transport: UnitState) -> list[tuple[int, Coord]]:
        state = self._require_state()
        if not self._can_launch_cargo(transport):
            return []
        options: list[tuple[int, Coord]] = []
        for cargo_id in transport.cargo:
            cargo = state.units[cargo_id]
            if not cargo.can_act:
                continue
            carried_by = cargo.carried_by
            coord = cargo.coord
            cargo.carried_by = None
            cargo.coord = transport.coord
            try:
                for dest in self.reachable_destinations(cargo):
                    if dest == transport.coord or state.unit_at(dest) is not None:
                        continue
                    options.append((cargo_id, dest))
            finally:
                cargo.carried_by = carried_by
                cargo.coord = coord
        return options

    def _can_launch_silo(self, unit: UnitState) -> bool:
        state = self._require_state()
        if unit.unit_type not in {"infantry", "mech"}:
            return False
        return state.map.tile_at(unit.coord).terrain == "missile_silo"

    @staticmethod
    def _can_launch_cargo(unit: UnitState) -> bool:
        return unit.unit_type == "carrier" and bool(unit.cargo)

    def _launch_targets(self) -> list[Coord]:
        state = self._require_state()
        return [
            Coord(x, y)
            for y in range(state.map.height)
            for x in range(state.map.width)
        ]

    def _require_state(self) -> GameState:
        if self.state is None:
            raise RuntimeError("Game has not been reset.")
        return self.state

    def _start_turn(self, player_id: int, collect_income: bool = True) -> list[Event]:
        state = self._require_state()
        events: list[Event] = []
        events.extend(self._advance_weather())
        state.players[player_id].swapped_this_turn = False
        if collect_income:
            commander = self._active_commander(state.players[player_id])
            base_income = int(self.rules["income_per_city"])
            income = sum(
                income_per_property(commander, base_income)
                for row in state.map.tiles
                for tile in row
                if tile.owner == player_id and tile.definition.profitable
            )
            state.players[player_id].funds += income
        for unit in list(state.map_units(player_id)):
            if unit.id not in state.units:
                continue
            events.extend(self._burn_idle_fuel(unit))
        for unit in state.map_units(player_id):
            self._repair_and_resupply(unit)
        for unit in state.living_units(player_id):
            unit.can_act = True
            if unit.stunned_turns > 0:
                unit.stunned_turns -= 1
                unit.can_act = False
                events.append(Event("stun_skip", {"unit_id": unit.id}))
        self._reset_fog_memory(player_id)
        return events

    def _end_turn(self) -> list[Event]:
        state = self._require_state()
        current = state.current_player
        for unit in state.living_units(current):
            unit.can_act = False
        player = state.players[current]
        if player.active_power_turns > 0:
            player.active_power_turns -= 1
            if player.active_power_turns <= 0:
                player.active_power_kind = "power"
        player_ids = sorted(state.players)
        idx = player_ids.index(current)
        next_idx = (idx + 1) % len(player_ids)
        if next_idx == 0:
            state.turn += 1
        state.current_player = player_ids[next_idx]
        events = [Event("end_turn", {"from": current, "to": state.current_player})]
        events.extend(self._start_turn(state.current_player, collect_income=True))
        return events

    def _set_temporary_weather(
        self, weather: str, duration_rounds: int = 1
    ) -> Event:
        state = self._require_state()
        state.weather_base = (
            state.weather_base
            if state.weather_turns_remaining > 0
            else state.weather
        )
        state.weather = weather
        state.weather_turns_remaining = max(
            0, len(state.players) * duration_rounds - 1
        )
        return Event(
            "weather",
            {
                "weather": weather,
                "duration_rounds": duration_rounds,
                "turns_remaining": state.weather_turns_remaining,
            },
        )

    def _advance_weather(self) -> list[Event]:
        state = self._require_state()
        if state.weather_turns_remaining > 0:
            state.weather_turns_remaining -= 1
            return []
        if state.weather != state.weather_base:
            previous = state.weather
            state.weather = state.weather_base
            return [
                Event(
                    "weather",
                    {
                        "weather": state.weather,
                        "previous": previous,
                        "duration_rounds": 0,
                        "turns_remaining": 0,
                    },
                )
            ]
        return []

    def _wait(self, action: Action) -> list[Event]:
        unit = self._active_unit(action.unit_id)
        unit.can_act = False
        return [Event("wait", {"unit_id": unit.id})]

    def _move(self, action: Action) -> list[Event]:
        unit = self._active_unit(action.unit_id)
        if not action.path:
            raise ValueError("MOVE requires a destination in path.")
        events = self._relocate_unit(unit, action.path)
        unit.can_act = False
        return events

    def _relocate_unit(
        self,
        unit: UnitState,
        path: tuple[Coord, ...],
    ) -> list[Event]:
        state = self._require_state()
        dest, move_cost, resolved_path = self._validated_move_path(unit, path)
        fuel_before = unit.fuel
        fuel_cost = move_cost * unit.definition.fuel_burn_per_tile
        if unit.fuel is not None:
            unit.fuel = max(0, unit.fuel - fuel_cost)
        for coord in resolved_path:
            if state.map.in_bounds(coord):
                unit.coord = coord
                self._update_fog_memory(unit.owner)
        unit.coord = dest
        unit.capture_progress = 0
        return [
            Event(
                "move",
                {
                    "unit_id": unit.id,
                    "to": (dest.x, dest.y),
                    "fuel_before": fuel_before,
                    "fuel_after": unit.fuel,
                    "fuel_cost": fuel_cost,
                },
            )
        ]

    def _validated_move_path(
        self, unit: UnitState, path: tuple[Coord, ...]
    ) -> tuple[Coord, int, tuple[Coord, ...]]:
        state = self._require_state()
        dest = path[-1]
        reachable, reachable_paths = self._reachable_costs_and_paths(unit)
        if len(path) == 1 and dest in reachable:
            return dest, reachable[dest], reachable_paths[dest]
        costs = self._movement_costs(unit)
        spent = 0
        current = unit.coord
        fuel_budget = unit.fuel if unit.fuel is not None else unit.definition.max_fuel
        for step in path:
            if current.manhattan(step) != 1:
                raise ValueError(f"Non-contiguous move path step: {step}")
            if not state.map.in_bounds(step):
                raise ValueError(f"Illegal move destination: {step}")
            occupant = state.unit_at(step)
            if occupant is not None and occupant.id != unit.id:
                raise ValueError(f"Illegal occupied move destination: {step}")
            terrain = state.map.tile_at(step).terrain
            terrain_cost = costs.get(terrain)
            if terrain_cost is None:
                raise ValueError(f"Illegal terrain for move destination: {step}")
            spent += int(terrain_cost)
            if spent > self._move_power(unit):
                raise ValueError(f"Illegal move destination: {dest}")
            if spent * unit.definition.fuel_burn_per_tile > fuel_budget:
                raise ValueError(f"Insufficient fuel for move destination: {dest}")
            current = step
        return dest, spent, path

    def _capture(self, action: Action) -> list[Event]:
        state = self._require_state()
        unit = self._active_unit(action.unit_id)
        if not unit.definition.can_capture:
            raise ValueError("Unit cannot capture.")
        tile = state.map.tile_at(unit.coord)
        if not tile.definition.capturable or tile.owner == unit.owner:
            raise ValueError("Tile cannot be captured by this unit.")
        commander = self._active_commander(state.players[unit.owner])
        progress = unit.hp // 10
        player = state.players[unit.owner]
        progress = progress * capture_multiplier(
            commander,
            unit,
            player.active_power_turns > 0,
            player.active_power_kind,
        ) // 100
        unit.capture_progress += max(1, progress)
        events = [
            Event(
                "capture_progress",
                {
                    "unit_id": unit.id,
                    "owner": unit.owner,
                    "coord": (unit.coord.x, unit.coord.y),
                    "progress": unit.capture_progress,
                },
            )
        ]
        if unit.capture_progress >= int(self.rules["capture_threshold"]):
            old_owner = tile.owner
            tile.owner = unit.owner
            unit.capture_progress = 0
            events.append(
                Event(
                    "capture",
                    {
                        "coord": (unit.coord.x, unit.coord.y),
                        "old_owner": old_owner,
                        "new_owner": unit.owner,
                    },
                )
            )
        unit.can_act = False
        return events

    def _co_ability(self, action: Action) -> list[Event]:
        state = self._require_state()
        player = state.players[state.current_player]
        commander = self._active_commander(player)
        requested_kind = str(action.metadata.get("power", "power"))
        if requested_kind not in {"power", "super"}:
            raise ValueError(f"Unknown CO power kind: {requested_kind}")
        if not self._can_activate_power(player.id, requested_kind):
            raise ValueError("CO power is not charged.")
        player.power_charge -= power_cost(commander, requested_kind)
        player.active_power_turns = max(
            player.active_power_turns,
            commander.power_duration_turns,
        )
        player.active_power_kind = requested_kind
        healed: list[dict] = []
        heal_amount = power_heal(commander, requested_kind)
        if heal_amount:
            for unit in state.living_units(player.id):
                if unit.carried_by is not None or unit.hp >= unit.definition.max_hp:
                    continue
                before = unit.hp
                unit.hp = min(unit.definition.max_hp, unit.hp + heal_amount)
                if unit.hp != before:
                    healed.append({"unit_id": unit.id, "from": before, "to": unit.hp})
        funds_before = player.funds
        if commander.power_funds_multiplier != 100:
            player.funds = player.funds * commander.power_funds_multiplier // 100
        power_events: list[Event] = []
        if commander.key == "olaf":
            power_events.append(self._set_temporary_weather("snow", duration_rounds=1))
            if requested_kind == "super":
                for unit in state.living_units():
                    if unit.owner != player.id:
                        unit.hp = max(1, unit.hp - 20)
        elif commander.key == "drake":
            if requested_kind == "super":
                power_events.append(
                    self._set_temporary_weather("rain", duration_rounds=1)
                )
            amount = 20 if requested_kind == "super" else 10
            for unit in state.living_units():
                if unit.owner != player.id:
                    unit.hp = max(1, unit.hp - amount)
                    if unit.fuel is not None:
                        unit.fuel //= 2
        elif commander.key == "sasha":
            drain_percent = player.funds * 10 // 5000
            for other in state.players.values():
                if other.id == player.id:
                    continue
                other_commander = self._active_commander(other)
                drain = drain_percent * other_commander.power_cost // 100
                other.power_charge = max(0, other.power_charge - drain)
        elif commander.key == "hawke":
            amount = 20 if requested_kind == "super" else 10
            for unit in state.living_units():
                if unit.owner == player.id:
                    unit.hp = min(unit.definition.max_hp, unit.hp + amount)
                else:
                    unit.hp = max(1, unit.hp - amount)
        elif commander.key == "kindle":
            for unit in state.living_units():
                if unit.owner == player.id:
                    continue
                if state.map.tile_at(unit.coord).definition.capturable:
                    unit.hp = max(1, unit.hp - 30)
        elif commander.key == "von_bolt":
            power_events.extend(
                self._apply_meteor_power(
                    player.id,
                    power=30,
                    value_kind="cost",
                    center_on_enemy=False,
                    self_harm=False,
                    stun=True,
                    label="ex_machina",
                )
            )
        elif commander.key == "sturm":
            power_events.extend(
                self._apply_meteor_power(
                    player.id,
                    power=80 if requested_kind == "super" else 40,
                    value_kind="cost",
                    center_on_enemy=True,
                    self_harm=False,
                    stun=False,
                    label="meteor_strike",
                )
            )
        elif commander.key == "rachel" and requested_kind == "super":
            covering_fire_targets = [
                (
                    label,
                    self._plan_meteor_target(
                        player.id,
                        radius=2,
                        power=30,
                        value_kind=value_kind,
                        center_on_enemy=False,
                        self_harm=True,
                    )
                    or Coord(0, 0),
                )
                for value_kind, label in (
                    ("capture", "covering_fire_capture"),
                    ("cost", "covering_fire_cost"),
                    ("health", "covering_fire_health"),
                )
            ]
            for label, target in covering_fire_targets:
                power_events.append(
                    self._apply_radius_damage(
                        owner=player.id,
                        center=target,
                        radius=2,
                        power=30,
                        self_harm=True,
                        stun=False,
                        label=label,
                    )
                )
        elif commander.key == "sensei":
            spawn_type = "mech" if requested_kind == "super" else "infantry"
            power_events.extend(self._spawn_on_owned_cities(player.id, spawn_type, 90))
        if commander.key == "jess":
            for unit in state.living_units(player.id):
                self._resupply_unit(unit)
        return [
            Event(
                "co_power",
                {
                    "player_id": player.id,
                    "commander": commander.key,
                    "power_kind": requested_kind,
                    "power_name": power_name(commander, requested_kind),
                    "healed": healed,
                    "funds_before": funds_before,
                    "funds_after": player.funds,
                    "active_turns": player.active_power_turns,
                },
            )
        ] + power_events

    def _swap_co(self, _action: Action) -> list[Event]:
        state = self._require_state()
        player = state.players[state.current_player]
        if not self._can_swap_co(player.id):
            raise ValueError("CO swap is not available.")
        previous = player.commander
        next_index = (player.active_commander_index + 1) % len(player.commanders)
        player.set_active_commander_index(next_index)
        player.swapped_this_turn = True
        return [
            Event(
                "swap_co",
                {
                    "player_id": player.id,
                    "from": previous,
                    "to": player.commander,
                    "active_commander_index": player.active_commander_index,
                },
            )
        ]

    def _build(self, action: Action) -> list[Event]:
        state = self._require_state()
        if action.target is None or action.build_unit is None:
            raise ValueError("BUILD requires target and build_unit.")
        tile = state.map.tile_at(action.target)
        if tile.owner != state.current_player:
            raise ValueError("Cannot build on unowned property.")
        if state.unit_at(action.target) is not None:
            raise ValueError("Cannot build on occupied property.")
        if not self._unit_enabled(action.build_unit):
            raise ValueError(f"Unit type is disabled by config: {action.build_unit}.")
        build_options = self._build_options_for_tile(state.current_player, tile.terrain)
        if action.build_unit not in build_options:
            raise ValueError(f"Cannot build {action.build_unit} from {tile.terrain}.")
        unit_def = UNITS[action.build_unit]
        player = state.players[state.current_player]
        cost = self._unit_cost_for_player(state.current_player, action.build_unit)
        if player.funds < cost:
            raise ValueError("Insufficient funds.")
        unit_id = max(state.units, default=0) + 1
        player.funds -= cost
        state.units[unit_id] = UnitState(
            id=unit_id,
            owner=state.current_player,
            unit_type=action.build_unit,
            coord=action.target,
            fuel=unit_def.max_fuel,
            ammo=unit_def.max_ammo_by_weapon,
            can_act=False,
        )
        return [
            Event(
                "build",
                {
                    "unit_id": unit_id,
                    "unit_type": action.build_unit,
                    "coord": (action.target.x, action.target.y),
                },
            )
        ]

    def _attack(self, action: Action) -> list[Event]:
        state = self._require_state()
        attacker = self._active_unit(action.unit_id)
        if action.target is None:
            raise ValueError("ATTACK requires target.")
        defender = state.unit_at(action.target)
        if defender is None or defender.owner == attacker.owner:
            raise ValueError("No enemy target at coordinate.")
        if self.fog and not self.is_unit_visible(attacker.owner, defender):
            raise ValueError("No visible enemy target at coordinate.")
        distance = attacker.coord.manhattan(defender.coord)
        weapon = self._choose_weapon(
            attacker,
            defender.definition,
            distance,
            bool(action.metadata.get("after_moving", False)),
        )
        if weapon is None:
            raise ValueError("No weapon can attack this target.")
        counter = self._counter_weapon(defender, attacker, distance)
        defender_player = state.players[defender.owner]
        defender_commander = self._active_commander(defender_player)
        defender_preempts = (
            counter is not None
            and preemptive_counter(
                defender_commander,
                defender_player.active_power_turns > 0,
                defender_player.active_power_kind,
            )
        )
        events: list[Event] = []
        if defender_preempts and counter is not None:
            events.extend(
                self._apply_strike(
                    defender,
                    attacker,
                    counter,
                    event_type="counterattack",
                    is_counter=True,
                )
            )
        if attacker.id in state.units:
            events.extend(
                self._apply_strike(
                    attacker,
                    defender,
                    weapon,
                    event_type="attack",
                    is_counter=False,
                )
            )
        if (
            not defender_preempts
            and attacker.id in state.units
            and defender.id in state.units
            and counter is not None
        ):
            events.extend(
                self._apply_strike(
                    defender,
                    attacker,
                    counter,
                    event_type="counterattack",
                    is_counter=True,
                )
            )
        if attacker.id in state.units:
            attacker.can_act = False
        return events

    def _counter_weapon(
        self,
        defender: UnitState,
        attacker: UnitState,
        distance: int,
    ) -> WeaponDef | None:
        counter = self._choose_weapon(defender, attacker.definition, distance, False)
        if counter is None or counter.min_range != 1:
            return None
        return counter

    def _apply_strike(
        self,
        attacker: UnitState,
        defender: UnitState,
        weapon: WeaponDef,
        event_type: str,
        is_counter: bool,
    ) -> list[Event]:
        state = self._require_state()
        damage, luck_roll = self._calculate_damage_with_luck(
            attacker,
            defender,
            weapon,
            is_counter=is_counter,
        )
        defender_hp_before = defender.hp
        defender.hp = max(0, defender.hp - damage)
        events = [
            Event(
                event_type,
                {
                    "attacker": attacker.id,
                    "attacker_owner": attacker.owner,
                    "defender": defender.id,
                    "defender_owner": defender.owner,
                    "defender_unit_type": defender.unit_type,
                    "defender_unit_cost": defender.definition.cost,
                    "defender_hp_before": defender_hp_before,
                    "defender_hp_after": defender.hp,
                    "damage": damage,
                    "luck": luck_roll,
                    "weapon": weapon.name,
                },
            )
        ]
        if weapon.max_ammo is not None:
            attacker.ammo[weapon.name] -= 1
        if defender.hp <= 0:
            del state.units[defender.id]
            events.append(Event("unit_die", {"unit_id": defender.id}))
        return events

    def _join(self, action: Action) -> list[Event]:
        state = self._require_state()
        source = self._active_unit(action.unit_id)
        if action.target is None:
            raise ValueError("JOIN requires target.")
        target = state.unit_at(action.target)
        if target is None or target.owner != source.owner:
            raise ValueError("No allied join target at coordinate.")
        if target.id == source.id:
            raise ValueError("Unit cannot join itself.")
        if target.unit_type != source.unit_type:
            raise ValueError("Joined units must have the same unit type.")
        if source.cargo or target.cargo:
            raise ValueError("Loaded transports cannot join.")
        if source.coord.manhattan(target.coord) > 1:
            raise ValueError("Join target must be adjacent.")
        combined_hp = source.hp + target.hp
        overflow_hp = max(0, combined_hp - target.definition.max_hp)
        before_hp = target.hp
        before_fuel = target.fuel
        before_ammo = dict(target.ammo)
        target.hp = min(target.definition.max_hp, combined_hp)
        if source.fuel is not None or target.fuel is not None:
            target.fuel = min(
                target.definition.max_fuel,
                (target.fuel or 0) + (source.fuel or 0),
            )
        for weapon, max_ammo in target.definition.max_ammo_by_weapon.items():
            target.ammo[weapon] = min(
                max_ammo,
                target.ammo.get(weapon, 0) + source.ammo.get(weapon, 0),
            )
        refund = target.definition.cost * overflow_hp // target.definition.max_hp
        if refund:
            state.players[target.owner].funds += refund
        del state.units[source.id]
        target.can_act = False
        target.capture_progress = 0
        return [
            Event(
                "join",
                {
                    "source_unit_id": source.id,
                    "target_unit_id": target.id,
                    "hp_before": before_hp,
                    "hp_after": target.hp,
                    "fuel_before": before_fuel,
                    "fuel_after": target.fuel,
                    "ammo_before": before_ammo,
                    "ammo_after": dict(target.ammo),
                    "refund": refund,
                },
            )
        ]

    def _load(self, action: Action) -> list[Event]:
        state = self._require_state()
        cargo = self._active_unit(action.unit_id)
        if action.target is None:
            raise ValueError("LOAD requires target transport coordinate.")
        transport = state.unit_at(action.target)
        if transport is None or transport.owner != cargo.owner:
            raise ValueError("No allied transport at target.")
        if cargo.coord.manhattan(transport.coord) > 1:
            raise ValueError("Cargo must be adjacent to the transport.")
        if not self._can_load(transport, cargo):
            raise ValueError("Transport cannot load this unit.")
        cargo.carried_by = transport.id
        cargo.coord = transport.coord
        cargo.can_act = False
        cargo.capture_progress = 0
        transport.cargo.append(cargo.id)
        return [
            Event(
                "load",
                {
                    "cargo_unit_id": cargo.id,
                    "transport_unit_id": transport.id,
                },
            )
        ]

    def _unload(self, action: Action) -> list[Event]:
        state = self._require_state()
        transport = self._active_unit(action.unit_id)
        cargo_id = action.metadata.get("cargo_unit_id")
        if cargo_id is None or cargo_id not in transport.cargo:
            raise ValueError("UNLOAD requires cargo_unit_id in transport cargo.")
        if action.target is None:
            raise ValueError("UNLOAD requires target coordinate.")
        if action.target not in [dest for cid, dest in self.unload_options(transport) if cid == cargo_id]:
            raise ValueError("Illegal unload destination.")
        cargo = state.units[cargo_id]
        cargo.carried_by = None
        cargo.coord = action.target
        cargo.can_act = False
        cargo.capture_progress = 0
        transport.cargo.remove(cargo_id)
        transport.can_act = False
        return [
            Event(
                "unload",
                {
                    "cargo_unit_id": cargo.id,
                    "transport_unit_id": transport.id,
                    "coord": (action.target.x, action.target.y),
                },
            )
        ]

    def _launch(self, action: Action) -> list[Event]:
        state = self._require_state()
        unit = self._active_unit(action.unit_id)
        if action.metadata.get("cargo_unit_id") is not None:
            return self._launch_cargo(unit, action)
        return self._launch_silo(unit, action)

    def _launch_cargo(self, transport: UnitState, action: Action) -> list[Event]:
        state = self._require_state()
        cargo_id = action.metadata.get("cargo_unit_id")
        if cargo_id is None or cargo_id not in transport.cargo:
            raise ValueError("Cargo LAUNCH requires cargo_unit_id in transport cargo.")
        if not self._can_launch_cargo(transport):
            raise ValueError("Unit cannot launch cargo.")
        if action.target is None:
            raise ValueError("Cargo LAUNCH requires target coordinate.")
        if action.target not in [
            dest for cid, dest in self.launch_options(transport) if cid == cargo_id
        ]:
            raise ValueError("Illegal cargo launch destination.")
        cargo = state.units[cargo_id]
        cargo.carried_by = None
        cargo.coord = action.target
        cargo.capture_progress = 0
        transport.cargo.remove(cargo_id)
        transport.can_act = False
        return [
            Event(
                "launch_cargo",
                {
                    "transport_unit_id": transport.id,
                    "cargo_unit_id": cargo.id,
                    "coord": (action.target.x, action.target.y),
                    "cargo_can_act": cargo.can_act,
                },
            )
        ]

    def _launch_silo(self, unit: UnitState, action: Action) -> list[Event]:
        state = self._require_state()
        if not self._can_launch_silo(unit):
            raise ValueError("Unit cannot launch from this tile.")
        if action.target is None or not state.map.in_bounds(action.target):
            raise ValueError("LAUNCH requires an in-bounds target.")
        affected: list[dict] = []
        for target in state.map_units():
            if target.coord.manhattan(action.target) > 2:
                continue
            hp_before = target.hp
            target.hp = max(10, target.hp - 30)
            if target.hp != hp_before:
                affected.append(
                    {
                        "unit_id": target.id,
                        "owner": target.owner,
                        "hp_before": hp_before,
                        "hp_after": target.hp,
                    }
                )
        silo_coord = unit.coord
        state.map.tile_at(silo_coord).terrain = "spent_missile_silo"
        unit.can_act = False
        unit.capture_progress = 0
        return [
            Event(
                "launch",
                {
                    "unit_id": unit.id,
                    "silo": (silo_coord.x, silo_coord.y),
                    "target": (action.target.x, action.target.y),
                    "affected": affected,
                },
            )
        ]

    def _apply_meteor_power(
        self,
        player_id: int,
        power: int,
        value_kind: str,
        center_on_enemy: bool,
        self_harm: bool,
        stun: bool,
        label: str,
    ) -> list[Event]:
        target = self._plan_meteor_target(
            player_id,
            radius=2,
            power=power,
            value_kind=value_kind,
            center_on_enemy=center_on_enemy,
            self_harm=self_harm,
        )
        if target is None:
            target = Coord(0, 0)
        return [
            self._apply_radius_damage(
                owner=player_id,
                center=target,
                radius=2,
                power=power,
                self_harm=self_harm,
                stun=stun,
                label=label,
            )
        ]

    def _plan_meteor_target(
        self,
        player_id: int,
        radius: int,
        power: int,
        value_kind: str,
        center_on_enemy: bool,
        self_harm: bool,
    ) -> Coord | None:
        state = self._require_state()
        best_target: Coord | None = None
        best_value = 0
        for y in range(state.map.height):
            for x in range(state.map.width):
                coord = Coord(x, y)
                if center_on_enemy:
                    unit = state.unit_at(coord)
                    if unit is None or unit.owner == player_id:
                        continue
                value = sum(
                    self._meteor_unit_value(player_id, unit, power, value_kind, self_harm)
                    for unit in state.map_units()
                    if unit.coord.manhattan(coord) <= radius
                )
                if value > best_value:
                    best_value = value
                    best_target = coord
        return best_target

    def _meteor_unit_value(
        self,
        player_id: int,
        unit: UnitState,
        power: int,
        value_kind: str,
        self_harm: bool,
    ) -> int:
        hp_value = 1 if unit.hp < 10 else min(unit.hp, power)
        if value_kind == "cost":
            value = 2 if unit.hp < 10 else unit.definition.cost * hp_value // 100
        elif value_kind == "health":
            value = hp_value
        elif value_kind == "capture":
            value = hp_value
            if unit.definition.can_capture:
                value *= 4
                if unit.capture_progress > 0:
                    value *= 2
        else:
            raise ValueError(f"Unknown meteor value kind: {value_kind}")
        if unit.owner != player_id:
            return value
        return -value if self_harm else 0

    def _apply_radius_damage(
        self,
        owner: int,
        center: Coord,
        radius: int,
        power: int,
        self_harm: bool,
        stun: bool,
        label: str,
    ) -> Event:
        state = self._require_state()
        affected: list[dict] = []
        for unit in state.map_units():
            if unit.coord.manhattan(center) > radius:
                continue
            if unit.owner == owner and not self_harm:
                continue
            hp_before = unit.hp
            can_act_before = unit.can_act
            unit.hp = max(10, unit.hp - power)
            if stun and unit.owner != owner:
                unit.stunned_turns = max(unit.stunned_turns, 1)
                unit.can_act = False
            if unit.hp != hp_before or unit.can_act != can_act_before or stun:
                affected.append(
                    {
                        "unit_id": unit.id,
                        "owner": unit.owner,
                        "hp_before": hp_before,
                        "hp_after": unit.hp,
                        "stunned_turns": unit.stunned_turns,
                    }
                )
        return Event(
            "meteor",
            {
                "label": label,
                "owner": owner,
                "target": (center.x, center.y),
                "radius": radius,
                "damage": power,
                "stun": stun,
                "affected": affected,
            },
        )

    def _spawn_on_owned_cities(
        self,
        player_id: int,
        unit_type: str,
        hp: int,
    ) -> list[Event]:
        state = self._require_state()
        if not self._unit_enabled(unit_type):
            return []
        events: list[Event] = []
        unit_def = UNITS[unit_type]
        for y, row in enumerate(state.map.tiles):
            for x, tile in enumerate(row):
                coord = Coord(x, y)
                if tile.terrain != "city" or tile.owner != player_id:
                    continue
                if state.unit_at(coord) is not None:
                    continue
                unit_id = max(state.units, default=0) + 1
                state.units[unit_id] = UnitState(
                    id=unit_id,
                    owner=player_id,
                    unit_type=unit_type,
                    coord=coord,
                    hp=hp,
                    fuel=unit_def.max_fuel,
                    ammo=unit_def.max_ammo_by_weapon,
                    can_act=True,
                )
                events.append(
                    Event(
                        "spawn",
                        {
                            "unit_id": unit_id,
                            "owner": player_id,
                            "unit_type": unit_type,
                            "coord": (coord.x, coord.y),
                            "hp": hp,
                        },
                    )
                )
        return events

    def _resupply(self, action: Action) -> list[Event]:
        supplier = self._active_unit(action.unit_id)
        if not self._can_resupply_adjacent(supplier):
            raise ValueError("Unit cannot resupply adjacent allies.")
        target = self._target_adjacent_ally(supplier, action.target, "RESUPPLY")
        if not self._needs_resupply(target):
            raise ValueError("Target does not need resupply.")
        before_fuel = target.fuel
        before_ammo = dict(target.ammo)
        self._resupply_unit(target)
        supplier.can_act = False
        return [
            Event(
                "resupply",
                {
                    "supplier_unit_id": supplier.id,
                    "target_unit_id": target.id,
                    "fuel_before": before_fuel,
                    "fuel_after": target.fuel,
                    "ammo_before": before_ammo,
                    "ammo_after": dict(target.ammo),
                },
            )
        ]

    def _repair(self, action: Action) -> list[Event]:
        state = self._require_state()
        repairer = self._active_unit(action.unit_id)
        if repairer.unit_type != "bboat":
            raise ValueError("Only BBoat can use manual REPAIR in this ruleset.")
        target = self._target_adjacent_ally(repairer, action.target, "REPAIR")
        if target.hp >= target.definition.max_hp:
            raise ValueError("Target is already at full HP.")
        player = state.players[repairer.owner]
        desired_heal = min(10, target.definition.max_hp - target.hp)
        max_affordable_heal = (
            player.funds * target.definition.max_hp // target.definition.cost
            if target.definition.cost > 0
            else desired_heal
        )
        actual_heal = min(desired_heal, max_affordable_heal)
        if actual_heal <= 0:
            raise ValueError("Insufficient funds to repair target.")
        repair_cost = target.definition.cost * actual_heal // target.definition.max_hp
        hp_before = target.hp
        fuel_before = target.fuel
        ammo_before = dict(target.ammo)
        player.funds -= repair_cost
        target.hp += actual_heal
        self._resupply_unit(target)
        repairer.can_act = False
        return [
            Event(
                "repair",
                {
                    "repairer_unit_id": repairer.id,
                    "target_unit_id": target.id,
                    "hp_before": hp_before,
                    "hp_after": target.hp,
                    "repair_cost": repair_cost,
                    "fuel_before": fuel_before,
                    "fuel_after": target.fuel,
                    "ammo_before": ammo_before,
                    "ammo_after": dict(target.ammo),
                },
            )
        ]

    def _delete(self, action: Action) -> list[Event]:
        state = self._require_state()
        unit = self._active_unit(action.unit_id)
        if not self._can_delete(unit):
            raise ValueError("Cannot delete the player's final map unit.")
        deleted_ids = [unit.id, *unit.cargo]
        for cargo_id in list(unit.cargo):
            if cargo_id in state.units:
                del state.units[cargo_id]
        del state.units[unit.id]
        return [
            Event(
                "delete",
                {
                    "unit_id": unit.id,
                    "deleted_unit_ids": deleted_ids,
                },
            )
        ]

    def _transform(self, action: Action) -> list[Event]:
        unit = self._active_unit(action.unit_id)
        target_type = unit.definition.transform_to
        if target_type is None:
            raise ValueError("Unit cannot transform.")
        old_type = unit.unit_type
        unit.unit_type = target_type
        unit.ammo = {
            weapon: min(unit.ammo.get(weapon, max_ammo), max_ammo)
            for weapon, max_ammo in unit.definition.max_ammo_by_weapon.items()
        }
        unit.can_act = False
        return [
            Event(
                "transform",
                {
                    "unit_id": unit.id,
                    "old_type": old_type,
                    "new_type": target_type,
                },
            )
        ]

    def _active_unit(self, unit_id: int | None) -> UnitState:
        state = self._require_state()
        if unit_id is None or unit_id not in state.units:
            raise ValueError("Missing or unknown unit_id.")
        unit = state.units[unit_id]
        if unit.owner != state.current_player:
            raise ValueError("Unit does not belong to current player.")
        if unit.carried_by is not None:
            raise ValueError("Carried unit cannot act.")
        if not unit.can_act:
            raise ValueError("Unit has already acted.")
        return unit

    def _choose_weapon(
        self,
        attacker: UnitState,
        defender_def: UnitDef,
        distance: int,
        after_moving: bool,
    ) -> WeaponDef | None:
        state = self._require_state()
        player = state.players[attacker.owner]
        commander = self._active_commander(player)
        active = player.active_power_turns > 0
        best_weapon: WeaponDef | None = None
        best_damage = 0
        for weapon in attacker.definition.weapons:
            if after_moving and not weapon.can_fire_after_moving:
                continue
            min_range, max_range = weapon_range(
                commander,
                active,
                weapon,
                player.active_power_kind,
            )
            if not (min_range <= distance <= max_range):
                continue
            if weapon.max_ammo is not None and attacker.ammo.get(weapon.name, 0) <= 0:
                continue
            base_damage = weapon.damage.get(defender_def.key, 0)
            if base_damage <= 0:
                continue
            if best_weapon is None or base_damage > best_damage:
                best_weapon = weapon
                best_damage = base_damage
        return best_weapon

    def _calculate_damage(
        self, attacker: UnitState, defender: UnitState, weapon: WeaponDef
    ) -> int:
        damage, _luck_roll = self._calculate_damage_with_luck(attacker, defender, weapon)
        return damage

    def _calculate_damage_with_luck(
        self,
        attacker: UnitState,
        defender: UnitState,
        weapon: WeaponDef,
        is_counter: bool = False,
    ) -> tuple[int, int]:
        state = self._require_state()
        base = weapon.damage.get(defender.unit_type, 0)
        attacker_player = state.players[attacker.owner]
        defender_player = state.players[defender.owner]
        attacker_commander = self._active_commander(attacker_player)
        defender_commander = self._active_commander(defender_player)
        attacker_active = attacker_player.active_power_turns > 0
        defender_active = defender_player.active_power_turns > 0
        attacker_tile = state.map.tile_at(attacker.coord)
        attack_power = 100 + attack_bonus(
            attacker_commander,
            attacker_active,
            attacker,
            weapon,
            attacker_tile.terrain,
            attacker_tile.definition.defense,
            self._owned_property_count(attacker.owner),
            attacker_player.active_power_kind,
            is_counter,
        )
        raw_damage = base * attack_power // 100
        luck_roll = self._roll_luck(
            attacker_commander,
            attacker_active,
            attacker_player.active_power_kind,
        )
        raw_damage += luck_roll
        overall_power = raw_damage * attacker.hp // 100
        if is_counter:
            overall_power = (
                overall_power * counter_damage_multiplier(attacker_commander) // 100
            )
        defender_bonus = defense_bonus(
            defender_commander,
            defender_active,
            defender,
            attacker.coord.manhattan(defender.coord),
            defender_player.active_power_kind,
        )
        terrain_subtraction = (
            self._terrain_stars_for_defender(defender) * defender.hp // 10
        )
        subtraction_multiplier = 100 - defender_bonus - terrain_subtraction
        overall_power = overall_power * subtraction_multiplier // 100
        return max(0, overall_power), luck_roll

    def _roll_luck(
        self,
        commander: CommanderDef,
        active: bool,
        power_kind: str = "power",
    ) -> int:
        if not self.luck:
            return 0
        low, high = luck_range(commander, active, power_kind)
        return self._rng.randint(low, high)

    def _terrain_stars_for_defender(self, defender: UnitState) -> int:
        if defender.definition.move_type == "flight":
            return 0
        state = self._require_state()
        return state.map.tile_at(defender.coord).definition.defense

    def _legal_build_actions(self, player_id: int) -> list[Action]:
        state = self._require_state()
        actions: list[Action] = []
        player = state.players[player_id]
        for y, row in enumerate(state.map.tiles):
            for x, tile in enumerate(row):
                if tile.owner != player_id or state.unit_at(Coord(x, y)) is not None:
                    continue
                for unit_type in self._build_options_for_tile(player_id, tile.terrain):
                    if player.funds >= self._unit_cost_for_player(player_id, unit_type):
                        actions.append(
                            Action(
                                ActionType.BUILD,
                                target=Coord(x, y),
                                build_unit=unit_type,
                            )
                        )
        return actions

    def _build_options_for_tile(self, player_id: int, terrain: str) -> tuple[str, ...]:
        state = self._require_state()
        options = tuple(BUILD_LISTS.get(terrain, ()))
        if terrain != "city":
            return self._filter_enabled_units(options)
        player = state.players[player_id]
        commander = self._active_commander(player)
        if player.active_power_turns <= 0:
            return self._filter_enabled_units(options)
        hachi_options = tuple(
            unit_type
            for unit_type in BUILD_LISTS.get("factory", ())
            if can_build_from_city(
                commander,
                True,
                unit_type,
                player.active_power_kind,
            )
        )
        return self._filter_enabled_units(
            tuple(dict.fromkeys((*options, *hachi_options)))
        )

    def _filter_enabled_units(self, unit_types: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            unit_type for unit_type in unit_types if self._unit_enabled(unit_type)
        )

    def _can_load(self, transport: UnitState, cargo: UnitState) -> bool:
        if transport.definition.cargo_capacity <= len(transport.cargo):
            return False
        return cargo.definition.unit_class in transport.definition.carry_classes

    def _can_delete(self, unit: UnitState) -> bool:
        state = self._require_state()
        return len(state.map_units(unit.owner)) > 1

    def _can_stand_on(self, unit: UnitState, coord: Coord) -> bool:
        state = self._require_state()
        terrain = state.map.tile_at(coord).terrain
        return terrain in self._movement_costs(unit)

    def _movement_costs(self, unit: UnitState) -> dict[str, int | None]:
        state = self._require_state()
        weather = state.weather
        commander = self._active_commander(state.players[unit.owner])
        if has_perfect_movement(
            commander,
            state.players[unit.owner].active_power_turns > 0,
            weather,
            state.players[unit.owner].active_power_kind,
        ):
            return {
                terrain: None if cost is None else 1
                for terrain, cost in MOVEMENT_COSTS_BY_WEATHER[weather][
                    unit.definition.move_type
                ].items()
            }
        weather = weather_for_movement(commander, weather)
        weather_costs = MOVEMENT_COSTS_BY_WEATHER[weather]
        return weather_costs[unit.definition.move_type]

    def _move_power(self, unit: UnitState) -> int:
        state = self._require_state()
        player = state.players[unit.owner]
        commander = self._active_commander(player)
        return unit.definition.move + move_bonus(
            commander,
            player.active_power_turns > 0,
            unit,
            player.active_power_kind,
        )

    def _unit_cost_for_player(self, player_id: int, unit_type: str) -> int:
        state = self._require_state()
        commander = self._active_commander(state.players[player_id])
        return unit_cost(
            commander,
            state.players[player_id].active_power_turns > 0,
            UNITS[unit_type].cost,
        )

    def _owned_property_count(self, player_id: int) -> int:
        state = self._require_state()
        return sum(
            1
            for row in state.map.tiles
            for tile in row
            if tile.owner == player_id and tile.definition.capturable
        )

    def _vision_range(self, unit: UnitState, terrain_boost: int = 0) -> int:
        state = self._require_state()
        player = state.players[unit.owner]
        commander = self._active_commander(player)
        radius = (
            unit.definition.vision
            + terrain_boost
            + vision_bonus(
                commander,
                player.active_power_turns > 0,
                player.active_power_kind,
            )
        )
        if state.weather == "rain":
            radius -= 1
        return max(1, radius)

    def _active_commander(self, player: PlayerState) -> CommanderDef:
        player.set_active_commander_index(player.active_commander_index)
        return commander_for(player.commander)

    def _normalize_commander_team(
        self, value: str | Sequence[str] | None
    ) -> list[str]:
        if value is None:
            value = "andy"
        if isinstance(value, str):
            team = [value]
        elif isinstance(value, Sequence):
            team = [str(key) for key in value]
        else:
            raise TypeError("commander team must be a string or sequence of strings.")
        if not team:
            raise ValueError("commander team cannot be empty.")
        for key in team:
            commander_for(key)
        return team

    def _resolve_commanders(self, player_ids: list[int]) -> dict[int, list[str]]:
        if self.commanders is None:
            return {player_id: ["andy"] for player_id in player_ids}
        if isinstance(self.commanders, str):
            team = self._normalize_commander_team(self.commanders)
            return {player_id: list(team) for player_id in player_ids}
        if isinstance(self.commanders, Mapping):
            resolved = {}
            for player_id in player_ids:
                value = self.commanders.get(
                    player_id,
                    self.commanders.get(str(player_id), "andy"),
                )
                resolved[player_id] = self._normalize_commander_team(value)
            return resolved
        if isinstance(self.commanders, Sequence):
            if len(self.commanders) < len(player_ids):
                raise ValueError("Not enough commanders for all players.")
            resolved = {}
            for player_id, value in zip(sorted(player_ids), self.commanders):
                resolved[player_id] = self._normalize_commander_team(value)
            return resolved
        raise TypeError("commanders must be None, str, sequence, or mapping.")

    def _can_activate_power(self, player_id: int, power_kind: str = "power") -> bool:
        state = self._require_state()
        player = state.players[player_id]
        commander = self._active_commander(player)
        try:
            cost = power_cost(commander, power_kind)
        except ValueError:
            return False
        return (
            player.active_power_turns <= 0
            and player.power_charge >= cost
        )

    def _can_swap_co(self, player_id: int) -> bool:
        state = self._require_state()
        player = state.players[player_id]
        return (
            (self.tag_mode or len(player.commanders) > 1)
            and len(player.commanders) > 1
            and player.active_power_turns <= 0
            and not player.swapped_this_turn
        )

    def _award_power_charge(self, events: list[Event]) -> None:
        state = self._require_state()
        for event in events:
            if event.type not in {"attack", "counterattack"}:
                continue
            damage = int(event.payload["damage"])
            attacker_id = int(event.payload["attacker_owner"])
            defender_id = int(event.payload["defender_owner"])
            attacker = state.players[attacker_id]
            defender = state.players[defender_id]
            attacker_def = self._active_commander(attacker)
            defender_def = self._active_commander(defender)
            attacker.power_charge = min(
                max_power_cost(attacker_def),
                attacker.power_charge + damage // 2,
            )
            defender.power_charge = min(
                max_power_cost(defender_def),
                defender.power_charge + damage,
            )

    def _adjacent_allied_units(self, unit: UnitState) -> list[UnitState]:
        state = self._require_state()
        return [
            other
            for other in state.map_units(unit.owner)
            if other.id != unit.id and unit.coord.manhattan(other.coord) <= 1
        ]

    def _target_adjacent_ally(
        self, unit: UnitState, target_coord: Coord | None, action_name: str
    ) -> UnitState:
        state = self._require_state()
        if target_coord is None:
            raise ValueError(f"{action_name} requires target.")
        target = state.unit_at(target_coord)
        if target is None or target.owner != unit.owner:
            raise ValueError(f"No allied {action_name} target at coordinate.")
        if target.id == unit.id:
            raise ValueError(f"{action_name} target cannot be the acting unit.")
        if unit.coord.manhattan(target.coord) > 1:
            raise ValueError(f"{action_name} target must be adjacent.")
        return target

    @staticmethod
    def _can_resupply_adjacent(unit: UnitState) -> bool:
        return unit.unit_type in {"apc", "bboat"}

    @staticmethod
    def _needs_resupply(unit: UnitState) -> bool:
        if unit.fuel is not None and unit.fuel < unit.definition.max_fuel:
            return True
        for weapon, max_ammo in unit.definition.max_ammo_by_weapon.items():
            if unit.ammo.get(weapon, 0) < max_ammo:
                return True
        return False

    @staticmethod
    def _resupply_unit(unit: UnitState) -> None:
        unit.fuel = unit.definition.max_fuel
        unit.ammo = unit.definition.max_ammo_by_weapon

    def _burn_idle_fuel(self, unit: UnitState) -> list[Event]:
        state = self._require_state()
        commander = self._active_commander(state.players[unit.owner])
        burn = idle_fuel_burn(commander, unit, unit.definition.fuel_burn_idle)
        if burn <= 0 or unit.fuel is None or self._on_friendly_supply_tile(unit):
            return []
        before = unit.fuel
        unit.fuel = max(0, unit.fuel - burn)
        events = [
            Event(
                "fuel_burn",
                {
                    "unit_id": unit.id,
                    "owner": unit.owner,
                    "fuel_before": before,
                    "fuel_after": unit.fuel,
                    "fuel_burn": burn,
                },
            )
        ]
        if unit.fuel <= 0 and self._unit_domain(unit) in {"air", "sea"}:
            del state.units[unit.id]
            events.append(Event("unit_die", {"unit_id": unit.id, "reason": "fuel"}))
        return events

    def _on_friendly_supply_tile(self, unit: UnitState) -> bool:
        state = self._require_state()
        tile = state.map.tile_at(unit.coord)
        if tile.owner != unit.owner:
            return False
        terrain = tile.definition
        unit_domain = self._unit_domain(unit)
        return (
            (unit_domain == "land" and terrain.heals_land)
            or (unit_domain == "air" and terrain.heals_air)
            or (unit_domain == "sea" and terrain.heals_sea)
        )

    def _check_victory(self) -> None:
        state = self._require_state()
        alive_players = []
        for player_id in state.players:
            has_hq = any(
                tile.terrain == "hq" and tile.owner == player_id
                for row in state.map.tiles
                for tile in row
            )
            has_units = bool(state.living_units(player_id))
            if has_hq and has_units:
                alive_players.append(player_id)
            else:
                state.players[player_id].defeated = True
        if len(alive_players) == 1:
            state.done = True
            state.truncated = False
            state.winner = alive_players[0]

    def _repair_and_resupply(self, unit: UnitState) -> None:
        state = self._require_state()
        tile = state.map.tile_at(unit.coord)
        if not self._on_friendly_supply_tile(unit):
            return

        self._resupply_unit(unit)

        if unit.hp >= unit.definition.max_hp:
            return
        player = state.players[unit.owner]
        commander = self._active_commander(player)
        desired_heal = min(repair_power(commander), unit.definition.max_hp - unit.hp)
        max_affordable_heal = (
            player.funds * unit.definition.max_hp // unit.definition.cost
            if unit.definition.cost > 0
            else desired_heal
        )
        actual_heal = min(desired_heal, max_affordable_heal)
        if actual_heal <= 0:
            return
        repair_cost = unit.definition.cost * actual_heal // unit.definition.max_hp
        player.funds -= repair_cost
        unit.hp += actual_heal

    @staticmethod
    def _unit_domain(unit: UnitState) -> str:
        move_type = unit.definition.move_type
        if move_type == "flight":
            return "air"
        if move_type in {"float_light", "float_heavy"}:
            return "sea"
        return "land"
