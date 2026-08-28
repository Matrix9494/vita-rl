import pytest

from advancewars.engine import Action, ActionType, Game
from advancewars.engine.coordinates import Coord


def test_end_turn_collects_income():
    game = Game.from_map("duel")
    state = game.reset()

    assert state.players[1].funds == 0
    game.step(Action.end_turn())

    # Player 1 owns HQ + factory at reset.
    assert state.current_player == 1
    assert state.players[1].funds == 2000


def test_infantry_can_capture_neutral_city_in_two_turns():
    game = Game.from_map("duel")
    state = game.reset()
    unit = state.units[1]
    unit.coord = Coord(3, 0)

    game.step(Action(ActionType.CAPTURE, unit_id=unit.id))
    assert state.map.tile_at(Coord(3, 0)).owner is None
    game.step(Action.end_turn())
    game.step(Action.end_turn())
    unit.can_act = True
    game.step(Action(ActionType.CAPTURE, unit_id=unit.id))

    assert state.map.tile_at(Coord(3, 0)).owner == 0


def test_tank_attack_kills_weakened_infantry():
    game = Game.from_map("duel")
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(5, 1)
    defender.coord = Coord(6, 1)
    defender.hp = 30
    attacker.ammo = {"cannon": 9}

    _, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )

    assert any(event.type == "unit_die" and event.payload["unit_id"] == defender.id for event in events)
    assert defender.id not in state.units


def test_city_terrain_reduces_ground_unit_damage():
    game = Game.from_map("duel")
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(1, 1)
    attacker.ammo = {"TankCannon": 9}
    defender.unit_type = "infantry"
    defender.coord = Coord(1, 2)

    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )

    attack = next(event for event in events if event.type == "attack")
    assert attack.payload["weapon"] == "TankMGun"
    assert attack.payload["damage"] == 52
    assert defender.hp == 48


def test_air_units_do_not_receive_terrain_defense():
    game = Game.from_map("duel")
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "fighter"
    attacker.coord = Coord(3, 1)
    attacker.ammo = {"FighterMissiles": 9}
    defender.unit_type = "bomber"
    defender.coord = Coord(3, 2)

    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )

    attack = next(event for event in events if event.type == "attack")
    assert attack.payload["damage"] == 100
    assert defender.id not in state.units


def test_seeded_luck_is_reproducible_and_default_combat_stays_no_luck():
    def first_attack(game: Game) -> tuple[int, int]:
        state = game.reset()
        attacker = state.units[1]
        defender = state.units[2]
        attacker.coord = Coord(2, 1)
        defender.coord = Coord(3, 1)
        _state, events = game.step(
            Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
        )
        event = next(event for event in events if event.type == "attack")
        return event.payload["damage"], event.payload["luck"]

    assert first_attack(Game.from_map("duel")) == (55, 0)

    lucky_damage, lucky_roll = first_attack(Game.from_map("duel", luck=True, seed=7))
    assert (lucky_damage, lucky_roll) == first_attack(
        Game.from_map("duel", luck=True, seed=7)
    )
    assert 0 <= lucky_roll <= 9
    assert lucky_damage == 55 + lucky_roll


def test_move_consumes_action_and_blocks_enemy_tile():
    game = Game.from_map("duel")
    state = game.reset()
    unit = state.units[1]
    enemy = state.units[2]
    unit.coord = Coord(4, 1)
    enemy.coord = Coord(6, 1)

    destinations = game.reachable_destinations(unit)
    assert Coord(5, 1) in destinations
    assert Coord(6, 1) not in destinations

    game.step(Action(ActionType.MOVE, unit_id=unit.id, path=(Coord(5, 1),)))
    assert unit.coord == Coord(5, 1)
    assert not unit.can_act


def test_movement_consumes_fuel_and_fuel_limits_reachability():
    game = Game.from_map("duel")
    state = game.reset()
    unit = state.units[1]
    enemy = state.units[2]
    unit.unit_type = "tank"
    unit.coord = Coord(0, 4)
    unit.fuel = 1
    enemy.coord = Coord(6, 4)

    destinations = game.reachable_destinations(unit)
    assert Coord(1, 4) in destinations
    assert Coord(2, 4) not in destinations

    _state, events = game.step(Action(ActionType.MOVE, unit_id=unit.id, path=(Coord(1, 4),)))

    assert unit.fuel == 0
    assert events[0].payload["fuel_before"] == 1
    assert events[0].payload["fuel_after"] == 0
    assert events[0].payload["fuel_cost"] == 1


def test_move_with_destination_only_uses_inferred_path_cost():
    game = Game.from_map("duel")
    state = game.reset()
    unit = state.units[1]
    unit.coord = Coord(0, 1)
    unit.fuel = 99

    _state, events = game.step(
        Action(ActionType.MOVE, unit_id=unit.id, path=(Coord(3, 1),))
    )

    assert unit.coord == Coord(3, 1)
    assert unit.fuel == 96
    assert events[0].payload["fuel_cost"] == 3


def test_move_rejects_non_contiguous_explicit_path():
    game = Game.from_map("duel")
    state = game.reset()
    unit = state.units[1]
    unit.coord = Coord(0, 1)

    with pytest.raises(ValueError, match="Non-contiguous"):
        game.step(
            Action(
                ActionType.MOVE,
                unit_id=unit.id,
                path=(Coord(2, 1), Coord(3, 1)),
            )
        )


def test_air_and_sea_units_burn_idle_fuel_on_turn_start_and_can_die():
    game = Game.from_map("duel")
    state = game.reset()
    air_id = max(state.units) + 1
    state.units[air_id] = state.units[1].__class__(
        id=air_id,
        owner=1,
        unit_type="fighter",
        coord=Coord(3, 4),
        fuel=5,
        ammo={"FighterMissiles": 9},
        can_act=False,
    )

    _state, events = game.step(Action.end_turn())

    assert air_id not in state.units
    assert any(event.type == "fuel_burn" and event.payload["unit_id"] == air_id for event in events)
    assert any(
        event.type == "unit_die"
        and event.payload["unit_id"] == air_id
        and event.payload["reason"] == "fuel"
        for event in events
    )


def test_friendly_supply_tile_prevents_idle_fuel_burn_and_resupplies():
    game = Game.from_map("duel")
    state = game.reset()
    air_id = max(state.units) + 1
    state.map.tile_at(Coord(5, 1)).terrain = "airport"
    state.map.tile_at(Coord(5, 1)).owner = 1
    state.units[air_id] = state.units[1].__class__(
        id=air_id,
        owner=1,
        unit_type="fighter",
        coord=Coord(5, 1),
        fuel=4,
        ammo={"FighterMissiles": 0},
        can_act=False,
    )
    fighter = state.units[air_id]

    _state, events = game.step(Action.end_turn())

    assert fighter.fuel == fighter.definition.max_fuel
    assert fighter.ammo == {"FighterMissiles": 9}
    assert not any(event.type == "fuel_burn" and event.payload["unit_id"] == air_id for event in events)


def test_build_spends_funds_and_creates_unit():
    game = Game.from_map("duel")
    state = game.reset()
    state.players[0].funds = 1000
    target = Coord(1, 0)

    _, events = game.step(
        Action(ActionType.BUILD, target=target, build_unit="infantry")
    )

    built = state.unit_at(target)
    assert built is not None
    assert built.unit_type == "infantry"
    assert state.players[0].funds == 0
    assert events[0].type == "build"


def test_owned_property_repairs_and_resupplies_on_turn_start():
    game = Game.from_map("duel")
    state = game.reset()
    unit = state.units[1]
    unit.unit_type = "tank"
    unit.coord = Coord(0, 0)
    unit.hp = 50
    unit.fuel = 1
    unit.ammo = {"TankCannon": 0}
    state.players[0].funds = 10_000

    game.step(Action.end_turn())
    game.step(Action.end_turn())

    assert unit.hp == 70
    assert unit.fuel == unit.definition.max_fuel
    assert unit.ammo == {"TankCannon": 9}
    assert state.players[0].funds == 10_000 + 2000 - 1400


def test_apc_loads_and_unloads_infantry():
    game = Game.from_map("duel")
    state = game.reset()
    infantry = state.units[1]
    infantry.coord = Coord(0, 1)
    apc_id = max(state.units) + 1
    state.units[apc_id] = infantry.__class__(
        id=apc_id,
        owner=0,
        unit_type="apc",
        coord=Coord(1, 1),
        fuel=70,
        can_act=True,
    )
    apc = state.units[apc_id]

    legal_loads = [
        action for action in game.legal_actions() if action.type == ActionType.LOAD
    ]
    assert any(action.unit_id == infantry.id and action.target == apc.coord for action in legal_loads)

    game.step(Action(ActionType.LOAD, unit_id=infantry.id, target=apc.coord))
    assert infantry.carried_by == apc.id
    assert infantry.id in apc.cargo
    assert state.unit_at(infantry.coord) == apc

    game.step(Action.end_turn())
    game.step(Action.end_turn())
    unload_target = Coord(2, 1)
    game.step(
        Action(
            ActionType.UNLOAD,
            unit_id=apc.id,
            target=unload_target,
            metadata={"cargo_unit_id": infantry.id},
        )
    )

    assert infantry.carried_by is None
    assert infantry.coord == unload_target
    assert infantry.id not in apc.cargo
    assert not infantry.can_act
    assert not apc.can_act


def test_join_combines_same_unit_and_refunds_overflow_hp():
    game = Game.from_map("duel")
    state = game.reset()
    source = state.units[1]
    source.unit_type = "tank"
    source.coord = Coord(0, 4)
    source.hp = 50
    source.fuel = 30
    source.ammo = {"TankCannon": 8}
    target_id = max(state.units) + 1
    state.units[target_id] = source.__class__(
        id=target_id,
        owner=0,
        unit_type="tank",
        coord=Coord(1, 4),
        hp=60,
        fuel=20,
        ammo={"TankCannon": 2},
        can_act=False,
    )
    target = state.units[target_id]

    legal_joins = [
        action for action in game.legal_actions() if action.type == ActionType.JOIN
    ]
    assert any(action.unit_id == source.id and action.target == target.coord for action in legal_joins)

    _state, events = game.step(
        Action(ActionType.JOIN, unit_id=source.id, target=target.coord)
    )

    assert source.id not in state.units
    assert target.hp == 100
    assert target.fuel == 50
    assert target.ammo == {"TankCannon": 9}
    assert target.can_act is False
    assert state.players[0].funds == 700
    assert events[0].type == "join"
    assert events[0].payload["refund"] == 700


def test_delete_removes_unit_and_carried_cargo_but_not_final_unit():
    game = Game.from_map("duel")
    state = game.reset()
    cargo = state.units[1]
    apc_id = max(state.units) + 1
    state.units[apc_id] = cargo.__class__(
        id=apc_id,
        owner=0,
        unit_type="apc",
        coord=Coord(1, 1),
        fuel=70,
        can_act=True,
        cargo=[cargo.id],
    )
    cargo.carried_by = apc_id
    cargo.coord = Coord(1, 1)
    reserve_id = apc_id + 1
    state.units[reserve_id] = cargo.__class__(
        id=reserve_id,
        owner=0,
        unit_type="infantry",
        coord=Coord(0, 4),
        fuel=99,
        can_act=True,
    )

    _state, events = game.step(Action(ActionType.DELETE, unit_id=apc_id))

    assert apc_id not in state.units
    assert cargo.id not in state.units
    assert events[0].type == "delete"
    assert events[0].payload["deleted_unit_ids"] == [apc_id, cargo.id]

    remaining = [unit for unit in state.map_units(0)]
    for unit in remaining[1:]:
        del state.units[unit.id]
    final_unit = remaining[0]
    final_unit.can_act = True

    assert not any(
        action.type == ActionType.DELETE and action.unit_id == final_unit.id
        for action in game.legal_actions()
    )


def test_apc_resupplies_adjacent_ally():
    game = Game.from_map("duel")
    state = game.reset()
    tank = state.units[1]
    tank.unit_type = "tank"
    tank.coord = Coord(0, 4)
    tank.fuel = 1
    tank.ammo = {"TankCannon": 0}
    apc_id = max(state.units) + 1
    state.units[apc_id] = tank.__class__(
        id=apc_id,
        owner=0,
        unit_type="apc",
        coord=Coord(1, 4),
        fuel=70,
        can_act=True,
    )
    apc = state.units[apc_id]

    legal_resupplies = [
        action for action in game.legal_actions() if action.type == ActionType.RESUPPLY
    ]
    assert any(action.unit_id == apc.id and action.target == tank.coord for action in legal_resupplies)

    _state, events = game.step(
        Action(ActionType.RESUPPLY, unit_id=apc.id, target=tank.coord)
    )

    assert tank.fuel == tank.definition.max_fuel
    assert tank.ammo == {"TankCannon": 9}
    assert apc.can_act is False
    assert events[0].type == "resupply"


def test_bboat_repairs_and_resupplies_adjacent_ally():
    game = Game.from_map("duel")
    state = game.reset()
    tank = state.units[1]
    tank.unit_type = "tank"
    tank.coord = Coord(0, 4)
    tank.hp = 80
    tank.fuel = 1
    tank.ammo = {"TankCannon": 0}
    state.players[0].funds = 1000
    bboat_id = max(state.units) + 1
    state.units[bboat_id] = tank.__class__(
        id=bboat_id,
        owner=0,
        unit_type="bboat",
        coord=Coord(1, 4),
        fuel=60,
        can_act=True,
    )
    bboat = state.units[bboat_id]

    legal_repairs = [
        action for action in game.legal_actions() if action.type == ActionType.REPAIR
    ]
    assert any(action.unit_id == bboat.id and action.target == tank.coord for action in legal_repairs)

    _state, events = game.step(
        Action(ActionType.REPAIR, unit_id=bboat.id, target=tank.coord)
    )

    assert tank.hp == 90
    assert tank.fuel == tank.definition.max_fuel
    assert tank.ammo == {"TankCannon": 9}
    assert state.players[0].funds == 300
    assert bboat.can_act is False
    assert events[0].type == "repair"
    assert events[0].payload["repair_cost"] == 700


def test_infantry_launches_missile_silo_once_without_killing_or_charging_power():
    game = Game.from_map("duel")
    state = game.reset()
    launcher = state.units[1]
    launcher.coord = Coord(3, 0)
    state.map.tile_at(launcher.coord).terrain = "missile_silo"
    enemy = state.units[2]
    enemy.coord = Coord(4, 4)
    enemy.hp = 100
    ally_id = max(state.units) + 1
    state.units[ally_id] = launcher.__class__(
        id=ally_id,
        owner=0,
        unit_type="tank",
        coord=Coord(3, 4),
        hp=20,
        fuel=70,
        ammo={"TankCannon": 9},
        can_act=True,
    )
    ally = state.units[ally_id]
    state.players[0].power_charge = 5
    state.players[1].power_charge = 7

    legal_launches = [
        action for action in game.legal_actions() if action.type == ActionType.LAUNCH
    ]
    assert any(action.unit_id == launcher.id and action.target == Coord(4, 4) for action in legal_launches)

    _state, events = game.step(
        Action(ActionType.LAUNCH, unit_id=launcher.id, target=Coord(4, 4))
    )

    assert enemy.hp == 70
    assert ally.hp == 10
    assert launcher.hp == 100
    assert launcher.can_act is False
    assert state.map.tile_at(Coord(3, 0)).terrain == "spent_missile_silo"
    assert state.players[0].power_charge == 5
    assert state.players[1].power_charge == 7
    assert events[0].type == "launch"
    assert events[0].payload["target"] == (4, 4)
    assert not any(action.type == ActionType.LAUNCH for action in game.legal_actions())


def test_carried_cargo_readies_on_owner_turn_start():
    game = Game.from_map("duel")
    state = game.reset()
    carrier_id = max(state.units) + 1
    cargo_id = carrier_id + 1
    state.units[carrier_id] = state.units[1].__class__(
        id=carrier_id,
        owner=0,
        unit_type="carrier",
        coord=Coord(3, 2),
        fuel=99,
        ammo={"CarrierMissiles": 9},
        can_act=True,
        cargo=[cargo_id],
    )
    state.units[cargo_id] = state.units[1].__class__(
        id=cargo_id,
        owner=0,
        unit_type="b_copter",
        coord=Coord(3, 2),
        fuel=99,
        ammo={"CopterRockets": 6},
        can_act=False,
        carried_by=carrier_id,
    )

    game.step(Action.end_turn())
    game.step(Action.end_turn())

    assert state.current_player == 0
    assert state.units[cargo_id].can_act is True


def test_carrier_launches_cargo_to_reachable_destination_and_cargo_can_act():
    game = Game.from_map("duel")
    state = game.reset()
    carrier_id = max(state.units) + 1
    cargo_id = carrier_id + 1
    state.units[carrier_id] = state.units[1].__class__(
        id=carrier_id,
        owner=0,
        unit_type="carrier",
        coord=Coord(3, 2),
        fuel=99,
        ammo={"CarrierMissiles": 9},
        can_act=True,
        cargo=[cargo_id],
    )
    state.units[cargo_id] = state.units[1].__class__(
        id=cargo_id,
        owner=0,
        unit_type="b_copter",
        coord=Coord(3, 2),
        fuel=99,
        ammo={"CopterRockets": 6},
        can_act=True,
        carried_by=carrier_id,
    )
    carrier = state.units[carrier_id]
    cargo = state.units[cargo_id]
    target = Coord(5, 4)

    legal_launches = [
        action for action in game.legal_actions() if action.type == ActionType.LAUNCH
    ]
    assert any(
        action.unit_id == carrier.id
        and action.target == target
        and action.metadata.get("cargo_unit_id") == cargo.id
        for action in legal_launches
    )

    _state, events = game.step(
        Action(
            ActionType.LAUNCH,
            unit_id=carrier.id,
            target=target,
            metadata={"cargo_unit_id": cargo.id},
        )
    )

    assert cargo.carried_by is None
    assert cargo.coord == target
    assert cargo.can_act is True
    assert cargo.id not in carrier.cargo
    assert carrier.can_act is False
    assert events[0].type == "launch_cargo"
    assert any(
        action.type == ActionType.WAIT and action.unit_id == cargo.id
        for action in game.legal_actions()
    )


def test_transform_sub_and_stealth_states():
    game = Game.from_map("duel")
    state = game.reset()
    unit = state.units[1]
    unit.unit_type = "sub"
    unit.ammo = {"SubTorpedoes": 6}

    _, events = game.step(Action(ActionType.TRANSFORM, unit_id=unit.id))
    assert unit.unit_type == "sub_sub"
    assert events[0].payload["old_type"] == "sub"

    game.step(Action.end_turn())
    game.step(Action.end_turn())
    _, events = game.step(Action(ActionType.TRANSFORM, unit_id=unit.id))
    assert unit.unit_type == "sub"
    assert events[0].payload["new_type"] == "sub"
