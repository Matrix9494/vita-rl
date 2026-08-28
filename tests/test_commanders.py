import json

from advancewars import raw_env
from advancewars.engine import Action, ActionType, Game
from advancewars.engine.coordinates import Coord


def test_default_commander_is_andy():
    game = Game.from_map("duel")
    state = game.reset()

    assert state.players[0].commander == "andy"
    assert state.players[1].commander == "andy"


def test_co_power_becomes_legal_when_charged_and_heals_units():
    game = Game.from_map("duel")
    state = game.reset()
    unit = state.units[1]
    unit.hp = 50
    state.players[0].power_charge = 30

    assert any(action.type == ActionType.CO_ABILITY for action in game.legal_actions())

    _state, events = game.step(Action(ActionType.CO_ABILITY))

    assert unit.hp == 70
    assert state.players[0].power_charge == 0
    assert state.players[0].active_power_turns == 1
    assert events[0].type == "co_power"
    assert events[0].payload["power_name"] == "Hyper Repair"


def test_super_power_action_becomes_legal_when_fully_charged():
    game = Game.from_map("duel", commanders={0: "andy", 1: "andy"})
    state = game.reset()
    state.players[0].power_charge = 60

    super_actions = [
        action
        for action in game.legal_actions()
        if action.type == ActionType.CO_ABILITY
        and action.metadata.get("power") == "super"
    ]

    assert super_actions


def test_andy_hyper_upgrade_heals_and_adds_movement():
    game = Game.from_map("duel", commanders={0: "andy", 1: "andy"})
    state = game.reset()
    infantry = state.units[1]
    infantry.hp = 40
    infantry.coord = Coord(0, 4)
    state.players[0].power_charge = 60

    _state, events = game.step(
        Action(ActionType.CO_ABILITY, metadata={"power": "super"})
    )

    assert infantry.hp == 90
    assert Coord(4, 4) in game.reachable_destinations(infantry)
    assert events[0].payload["power_name"] == "Hyper Upgrade"
    assert events[0].payload["power_kind"] == "super"


def test_andy_power_boosts_attack_and_defense_for_current_turn():
    clear_game = Game.from_map("duel")
    clear_state = clear_game.reset()
    clear_attacker = clear_state.units[1]
    clear_defender = clear_state.units[2]
    clear_attacker.unit_type = "tank"
    clear_attacker.coord = Coord(4, 1)
    clear_defender.coord = Coord(5, 1)

    attack = Action(
        ActionType.ATTACK,
        unit_id=clear_attacker.id,
        target=clear_defender.coord,
    )
    _state, clear_events = clear_game.step(attack)
    clear_damage = next(
        event for event in clear_events if event.type == "attack"
    ).payload["damage"]

    powered_game = Game.from_map("duel")
    powered_state = powered_game.reset()
    powered_attacker = powered_state.units[1]
    powered_defender = powered_state.units[2]
    powered_attacker.unit_type = "tank"
    powered_attacker.coord = Coord(4, 1)
    powered_defender.coord = Coord(5, 1)
    powered_state.players[0].power_charge = 30
    powered_game.step(Action(ActionType.CO_ABILITY))

    _state, powered_events = powered_game.step(
        Action(
            ActionType.ATTACK,
            unit_id=powered_attacker.id,
            target=powered_defender.coord,
        )
    )
    powered_damage = next(
        event for event in powered_events if event.type == "attack"
    ).payload["damage"]

    assert clear_damage == 75
    assert powered_damage == 82


def test_co_power_expires_on_end_turn():
    game = Game.from_map("duel")
    state = game.reset()
    state.players[0].power_charge = 30

    game.step(Action(ActionType.CO_ABILITY))
    game.step(Action.end_turn())

    assert state.players[0].active_power_turns == 0


def test_combat_awards_power_charge():
    game = Game.from_map("duel")
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(4, 1)
    defender.coord = Coord(5, 1)

    game.step(Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord))

    assert state.players[0].power_charge > 0
    assert state.players[1].power_charge > 0


def test_commander_state_serializes_round_trip():
    game = Game.from_map("duel", commanders={0: "andy", 1: "andy"})
    state = game.reset()
    state.players[0].power_charge = 30
    game.step(Action(ActionType.CO_ABILITY))

    restored = Game.from_json(json.loads(json.dumps(game.to_json())))
    assert restored.state is not None
    assert restored.state.players[0].commander == "andy"
    assert restored.state.players[0].active_power_turns == 1


def test_env_accepts_commander_configuration():
    env = raw_env(commanders={0: "andy", 1: "andy"})
    env.reset()

    assert env.game.state is not None
    assert env.game.state.players[0].commander == "andy"


def test_observation_includes_current_player_power_state():
    env = raw_env()
    env.reset()
    assert env.game.state is not None
    env.game.state.players[0].power_charge = 15
    obs = env.observe("player_0")["observation"]

    assert obs[-2].min() == 0.5
    assert obs[-2].max() == 0.5
    assert obs[-1].max() == 0.0

    env.game.state.players[0].active_power_turns = 1
    obs = env.observe("player_0")["observation"]
    assert obs[-1].min() == 1.0
    assert obs[-1].max() == 1.0


def test_max_direct_units_hit_harder_and_indirect_range_is_shorter():
    game = Game.from_map("duel", commanders={0: "max", 1: "andy"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)

    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 90

    game = Game.from_map("duel", commanders={0: "max", 1: "andy"})
    state = game.reset()
    artillery = state.units[1]
    target = state.units[2]
    artillery.unit_type = "artillery"
    artillery.ammo = {"ArtilleryCannon": 9}
    artillery.coord = Coord(0, 1)
    target.coord = Coord(3, 1)

    assert target not in game.attack_targets(artillery, after_moving=False)


def test_grit_indirect_units_gain_range_and_damage():
    game = Game.from_map("duel", commanders={0: "grit", 1: "andy"})
    state = game.reset()
    artillery = state.units[1]
    target = state.units[2]
    artillery.unit_type = "artillery"
    artillery.ammo = {"ArtilleryCannon": 9}
    artillery.coord = Coord(0, 1)
    target.coord = Coord(4, 1)

    assert target in game.attack_targets(artillery, after_moving=False)
    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=artillery.id, target=target.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 108


def test_sami_footsoldiers_attack_and_capture_better():
    game = Game.from_map("duel", commanders={0: "sami", 1: "andy"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)

    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 71

    game = Game.from_map("duel", commanders={0: "sami", 1: "andy"})
    state = game.reset()
    infantry = state.units[1]
    infantry.coord = Coord(3, 0)
    game.step(Action(ActionType.CAPTURE, unit_id=infantry.id))
    assert infantry.capture_progress == 15


def test_sami_transports_gain_move_and_power_boosts_troop_move():
    game = Game.from_map("duel", commanders={0: "sami", 1: "andy"})
    state = game.reset()
    infantry = state.units[1]
    infantry.coord = Coord(0, 4)
    apc_id = max(state.units) + 1
    state.units[apc_id] = infantry.__class__(
        id=apc_id,
        owner=0,
        unit_type="apc",
        coord=Coord(0, 1),
        fuel=70,
        can_act=True,
    )
    apc = state.units[apc_id]

    assert Coord(6, 0) in game.reachable_destinations(apc)

    state.players[0].power_charge = 30
    game.step(Action(ActionType.CO_ABILITY))
    assert Coord(4, 4) in game.reachable_destinations(infantry)


def test_sami_victory_march_instant_captures_and_adds_two_troop_move():
    game = Game.from_map("duel", commanders={0: "sami", 1: "andy"})
    state = game.reset()
    infantry = state.units[1]
    infantry.hp = 10
    infantry.coord = Coord(3, 0)
    state.players[0].power_charge = 80

    game.step(Action(ActionType.CO_ABILITY, metadata={"power": "super"}))
    game.step(Action(ActionType.CAPTURE, unit_id=infantry.id))

    assert state.map.tile_at(Coord(3, 0)).owner == 0

    game = Game.from_map("duel", commanders={0: "sami", 1: "andy"})
    state = game.reset()
    infantry = state.units[1]
    infantry.coord = Coord(0, 4)
    state.players[0].power_charge = 80
    game.step(Action(ActionType.CO_ABILITY, metadata={"power": "super"}))
    assert Coord(5, 4) in game.reachable_destinations(infantry)


def test_colin_builds_discounted_units_and_hits_weaker():
    game = Game.from_map("duel", commanders={0: "colin", 1: "andy"})
    state = game.reset()
    state.players[0].funds = 800
    game.step(Action(ActionType.BUILD, target=Coord(1, 0), build_unit="infantry"))
    assert state.players[0].funds == 0

    game = Game.from_map("duel", commanders={0: "colin", 1: "andy"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)

    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 67


def test_colin_gold_rush_increases_funds():
    game = Game.from_map("duel", commanders={0: "colin", 1: "andy"})
    state = game.reset()
    state.players[0].funds = 10_000
    state.players[0].power_charge = 20

    _state, events = game.step(Action(ActionType.CO_ABILITY))

    assert state.players[0].funds == 15_000
    assert events[0].payload["power_name"] == "Gold Rush"


def test_kanbei_units_cost_more_hit_harder_and_defend_better():
    game = Game.from_map("duel", commanders={0: "kanbei", 1: "andy"})
    state = game.reset()
    state.players[0].funds = 1200
    game.step(Action(ActionType.BUILD, target=Coord(1, 0), build_unit="infantry"))
    assert state.players[0].funds == 0

    game = Game.from_map("duel", commanders={0: "kanbei", 1: "andy"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)
    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 97

    game = Game.from_map("duel", commanders={0: "andy", 1: "kanbei"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)
    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 52


def test_kanbei_powers_use_awbw_stats_and_samurai_spirit_counter_bonus():
    game = Game.from_map("duel", commanders={0: "kanbei", 1: "andy"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)
    state.players[0].power_charge = 40
    game.step(Action(ActionType.CO_ABILITY))

    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )

    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 112

    game = Game.from_map("duel", commanders={0: "andy", 1: "kanbei"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)
    state.players[1].active_power_turns = 1
    state.players[1].active_power_kind = "super"

    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )

    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 30

    game = Game.from_map("duel", commanders={0: "andy", 1: "kanbei"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.coord = Coord(2, 1)
    defender.unit_type = "tank"
    defender.coord = Coord(3, 1)
    state.players[1].active_power_turns = 1
    state.players[1].active_power_kind = "super"

    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )

    counter = next(event for event in events if event.type == "counterattack")
    assert counter.payload["damage"] == 157
    assert attacker.id not in state.units


def test_grimm_hits_harder_and_defends_worse():
    game = Game.from_map("duel", commanders={0: "grimm", 1: "andy"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)

    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 97

    game = Game.from_map("duel", commanders={0: "andy", 1: "grimm"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)

    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 90


def test_jake_gains_attack_on_plains_and_power_extends_indirect_range():
    game = Game.from_map("duel", commanders={0: "jake", 1: "andy"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(2, 2)
    defender.coord = Coord(2, 3)

    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 82

    game = Game.from_map("duel", commanders={0: "jake", 1: "andy"})
    state = game.reset()
    artillery = state.units[1]
    target = state.units[2]
    artillery.unit_type = "artillery"
    artillery.ammo = {"ArtilleryCannon": 9}
    artillery.coord = Coord(0, 1)
    target.coord = Coord(4, 1)
    assert target not in game.attack_targets(artillery, after_moving=False)

    state.players[0].power_charge = 30
    game.step(Action(ActionType.CO_ABILITY))
    assert target in game.attack_targets(artillery, after_moving=False)


def test_eagle_air_units_are_stronger_and_burn_less_fuel():
    game = Game.from_map("duel", commanders={0: "eagle", 1: "andy"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "fighter"
    attacker.ammo = {"FighterMissiles": 9}
    attacker.coord = Coord(2, 1)
    defender.unit_type = "bomber"
    defender.coord = Coord(3, 1)

    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 115

    game = Game.from_map("duel", commanders={0: "andy", 1: "eagle"})
    state = game.reset()
    air_id = max(state.units) + 1
    state.units[air_id] = state.units[1].__class__(
        id=air_id,
        owner=1,
        unit_type="fighter",
        coord=Coord(3, 4),
        fuel=5,
        ammo={"FighterMissiles": 9},
    )
    game.step(Action.end_turn())
    assert state.units[air_id].fuel == 2


def test_drake_boosts_sea_units_and_ignores_rain_movement_penalty():
    game = Game.from_map("duel", commanders={0: "drake", 1: "andy"}, weather="rain")
    state = game.reset()
    for x in range(state.map.width):
        state.map.tile_at(Coord(x, 4)).terrain = "sea"
    lander_id = max(state.units) + 1
    state.units[lander_id] = state.units[1].__class__(
        id=lander_id,
        owner=0,
        unit_type="lander",
        coord=Coord(0, 4),
        fuel=99,
        can_act=True,
    )
    lander = state.units[lander_id]
    assert Coord(6, 4) in game.reachable_destinations(lander)

    game = Game.from_map("duel", commanders={0: "drake", 1: "andy"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "b_copter"
    attacker.ammo = {"CopterMGun": 6}
    attacker.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)
    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 52


def test_drake_powers_damage_enemy_fuel_and_typhoon_sets_rain():
    game = Game.from_map("duel", commanders={0: "drake", 1: "andy"})
    state = game.reset()
    enemy = state.units[2]
    enemy.hp = 80
    enemy.fuel = 99
    state.players[0].power_charge = 40

    game.step(Action(ActionType.CO_ABILITY))

    assert enemy.hp == 70
    assert enemy.fuel == 49
    assert state.weather == "clear"

    game = Game.from_map("duel", commanders={0: "drake", 1: "andy"})
    state = game.reset()
    enemy = state.units[2]
    enemy.hp = 80
    enemy.fuel = 99
    state.players[0].power_charge = 70

    _state, events = game.step(
        Action(ActionType.CO_ABILITY, metadata={"power": "super"})
    )
    weather = next(event for event in events if event.type == "weather")

    assert enemy.hp == 60
    assert enemy.fuel == 49
    assert state.weather == "rain"
    assert weather.payload["weather"] == "rain"


def test_jess_vehicles_hit_harder_others_weaker_and_power_resupplies():
    game = Game.from_map("duel", commanders={0: "jess", 1: "andy"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)
    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 82

    game = Game.from_map("duel", commanders={0: "jess", 1: "andy"})
    state = game.reset()
    infantry = state.units[1]
    defender = state.units[2]
    infantry.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)
    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=infantry.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 49

    game = Game.from_map("duel", commanders={0: "jess", 1: "andy"})
    state = game.reset()
    tank = state.units[1]
    tank.unit_type = "tank"
    tank.fuel = 1
    tank.ammo = {"TankCannon": 0}
    state.players[0].power_charge = 30
    game.step(Action(ActionType.CO_ABILITY))
    assert tank.fuel == tank.definition.max_fuel
    assert tank.ammo == {"TankCannon": 9}


def test_olaf_ignores_snow_but_rain_uses_snow_movement_costs():
    game = Game.from_map("duel", commanders={0: "olaf", 1: "andy"}, weather="snow")
    state = game.reset()
    tank = state.units[1]
    tank.unit_type = "tank"
    tank.coord = Coord(0, 4)

    assert Coord(6, 4) in game.reachable_destinations(tank)

    game = Game.from_map("duel", commanders={0: "olaf", 1: "andy"}, weather="rain")
    state = game.reset()
    infantry = state.units[1]
    infantry.coord = Coord(0, 4)

    assert Coord(3, 4) not in game.reachable_destinations(infantry)


def test_sasha_gets_extra_income_and_market_crash_drains_power():
    game = Game.from_map("duel", commanders={0: "sasha", 1: "andy"})
    state = game.reset()

    game.step(Action.end_turn())
    game.step(Action.end_turn())
    assert state.players[0].funds == 2200

    state.players[0].funds = 10_000
    state.players[0].power_charge = 20
    state.players[1].power_charge = 30
    game.step(Action(ActionType.CO_ABILITY))
    assert state.players[1].power_charge == 24


def test_rachel_repairs_one_extra_hp_on_properties():
    game = Game.from_map("duel", commanders={0: "rachel", 1: "andy"})
    state = game.reset()
    infantry = state.units[1]
    infantry.coord = Coord(0, 0)
    infantry.hp = 50
    state.players[0].funds = 10_000

    game.step(Action.end_turn())
    game.step(Action.end_turn())

    assert infantry.hp == 80


def test_hachi_discounts_units_and_power_builds_ground_units_from_cities():
    game = Game.from_map("duel", commanders={0: "hachi", 1: "andy"})
    state = game.reset()
    state.players[0].funds = 900
    game.step(Action(ActionType.BUILD, target=Coord(1, 0), build_unit="infantry"))
    assert state.players[0].funds == 0

    game = Game.from_map("duel", commanders={0: "hachi", 1: "andy"})
    state = game.reset()
    state.map.tile_at(Coord(3, 0)).owner = 0
    state.players[0].funds = 5000
    assert not any(
        action.type == ActionType.BUILD
        and action.target == Coord(3, 0)
        and action.build_unit == "tank"
        for action in game.legal_actions()
    )

    state.players[0].power_charge = 30
    game.step(Action(ActionType.CO_ABILITY))
    assert not any(
        action.type == ActionType.BUILD
        and action.target == Coord(3, 0)
        and action.build_unit == "tank"
        for action in game.legal_actions()
    )

    game = Game.from_map("duel", commanders={0: "hachi", 1: "andy"})
    state = game.reset()
    state.map.tile_at(Coord(3, 0)).owner = 0
    state.players[0].funds = 5000
    state.players[0].power_charge = 50
    game.step(Action(ActionType.CO_ABILITY, metadata={"power": "super"}))
    game.step(Action(ActionType.BUILD, target=Coord(3, 0), build_unit="tank"))

    built_tank = state.unit_at(Coord(3, 0))
    assert built_tank is not None
    assert built_tank.unit_type == "tank"
    assert state.players[0].funds == 1500


def test_sensei_boosts_troops_copters_and_transports():
    game = Game.from_map("duel", commanders={0: "sensei", 1: "andy"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)

    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 77

    game = Game.from_map("duel", commanders={0: "sensei", 1: "andy"})
    state = game.reset()
    copter = state.units[1]
    defender = state.units[2]
    copter.unit_type = "b_copter"
    copter.ammo = {"CopterMGun": 6}
    copter.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)
    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=copter.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 112

    game = Game.from_map("duel", commanders={0: "sensei", 1: "andy"})
    state = game.reset()
    apc_id = max(state.units) + 1
    state.units[apc_id] = state.units[1].__class__(
        id=apc_id,
        owner=0,
        unit_type="apc",
        coord=Coord(0, 4),
        fuel=70,
        can_act=True,
    )
    assert Coord(6, 4) in game.reachable_destinations(state.units[apc_id])


def test_hawke_and_von_bolt_gain_global_stats_and_hawke_power_drains_hp():
    game = Game.from_map("duel", commanders={0: "hawke", 1: "andy"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)
    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 82

    game = Game.from_map("duel", commanders={0: "andy", 1: "von_bolt"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)
    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 67

    game = Game.from_map("duel", commanders={0: "hawke", 1: "andy"})
    state = game.reset()
    ally = state.units[1]
    enemy = state.units[2]
    ally.hp = 80
    enemy.hp = 80
    state.players[0].power_charge = 50
    game.step(Action(ActionType.CO_ABILITY))
    assert ally.hp == 90
    assert enemy.hp == 70


def test_olaf_and_hawke_super_powers_apply_stronger_mass_effects():
    game = Game.from_map("duel", commanders={0: "olaf", 1: "andy"})
    state = game.reset()
    enemy = state.units[2]
    enemy.hp = 80
    state.players[0].power_charge = 70

    game.step(Action(ActionType.CO_ABILITY, metadata={"power": "super"}))

    assert state.weather == "snow"
    assert enemy.hp == 60

    game = Game.from_map("duel", commanders={0: "hawke", 1: "andy"})
    state = game.reset()
    ally = state.units[1]
    enemy = state.units[2]
    ally.hp = 70
    enemy.hp = 80
    state.players[0].power_charge = 90

    game.step(Action(ActionType.CO_ABILITY, metadata={"power": "super"}))

    assert ally.hp == 90
    assert enemy.hp == 60


def test_adder_power_increases_all_unit_movement():
    game = Game.from_map("duel", commanders={0: "adder", 1: "andy"})
    state = game.reset()
    infantry = state.units[1]
    infantry.coord = Coord(0, 4)

    assert Coord(4, 4) not in game.reachable_destinations(infantry)
    state.players[0].power_charge = 20
    game.step(Action(ActionType.CO_ABILITY))
    assert Coord(4, 4) in game.reachable_destinations(infantry)


def test_adder_sidewinder_adds_two_movement_and_nell_lady_luck_range():
    game = Game.from_map("duel", commanders={0: "adder", 1: "andy"})
    state = game.reset()
    infantry = state.units[1]
    infantry.coord = Coord(0, 4)
    state.players[0].power_charge = 50

    game.step(Action(ActionType.CO_ABILITY, metadata={"power": "super"}))

    assert Coord(5, 4) in game.reachable_destinations(infantry)

    game = Game.from_map("duel", commanders={0: "nell", 1: "andy"}, luck=True)
    state = game.reset(seed=4)
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)
    state.players[0].power_charge = 60
    game.step(Action(ActionType.CO_ABILITY, metadata={"power": "super"}))
    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    luck = next(event for event in events if event.type == "attack").payload["luck"]
    assert 0 <= luck <= 99


def test_lash_koal_and_kindle_gain_terrain_based_attack():
    game = Game.from_map("duel", commanders={0: "lash", 1: "andy"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(3, 0)
    defender.coord = Coord(3, 1)
    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 97

    game = Game.from_map("duel", commanders={0: "koal", 1: "andy"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)
    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 82

    game = Game.from_map("duel", commanders={0: "kindle", 1: "andy"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(3, 0)
    defender.coord = Coord(3, 1)
    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 105


def test_koal_lash_sturm_power_movement_hooks():
    game = Game.from_map("duel", commanders={0: "koal", 1: "andy"})
    state = game.reset()
    infantry = state.units[1]
    infantry.coord = Coord(0, 4)
    assert Coord(4, 4) not in game.reachable_destinations(infantry)
    state.players[0].power_charge = 30
    game.step(Action(ActionType.CO_ABILITY))
    assert Coord(4, 4) in game.reachable_destinations(infantry)

    game = Game.from_map("duel", commanders={0: "lash", 1: "andy"})
    state = game.reset()
    tank = state.units[1]
    tank.unit_type = "tank"
    tank.coord = Coord(0, 4)
    state.map.tile_at(Coord(1, 4)).terrain = "forest"
    assert Coord(6, 4) not in game.reachable_destinations(tank)
    state.players[0].power_charge = 40
    game.step(Action(ActionType.CO_ABILITY))
    assert Coord(6, 4) in game.reachable_destinations(tank)

    game = Game.from_map("duel", commanders={0: "sturm", 1: "andy"})
    state = game.reset()
    tank = state.units[1]
    tank.unit_type = "tank"
    tank.coord = Coord(0, 4)
    state.map.tile_at(Coord(1, 4)).terrain = "forest"
    assert Coord(6, 4) in game.reachable_destinations(tank)


def test_javier_sturm_and_sonja_defensive_hooks():
    game = Game.from_map("duel", commanders={0: "andy", 1: "javier"})
    state = game.reset()
    artillery = state.units[1]
    defender = state.units[2]
    artillery.unit_type = "artillery"
    artillery.ammo = {"ArtilleryCannon": 9}
    artillery.coord = Coord(0, 1)
    defender.coord = Coord(3, 1)
    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=artillery.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 72

    game = Game.from_map("duel", commanders={0: "andy", 1: "sturm"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)
    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert damage == 60

    game = Game.from_map("duel", commanders={0: "andy", 1: "sonja"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.coord = Coord(2, 1)
    defender.unit_type = "tank"
    defender.coord = Coord(3, 1)
    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )
    counter = next(event for event in events if event.type == "counterattack")
    assert counter.payload["damage"] == 106


def test_sonja_counter_break_counterattacks_before_incoming_attack():
    game = Game.from_map("duel", commanders={0: "andy", 1: "sonja"})
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.coord = Coord(2, 1)
    attacker.hp = 10
    defender.unit_type = "tank"
    defender.coord = Coord(3, 1)
    state.players[1].active_power_turns = 1
    state.players[1].active_power_kind = "super"

    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )

    assert events[0].type == "counterattack"
    assert not any(event.type == "attack" for event in events)
    assert attacker.id not in state.units
    assert defender.hp == 100


def test_sonja_vision_and_luck_commanders_use_extended_luck_ranges():
    game = Game.from_map("duel", commanders={0: "sonja", 1: "andy"}, fog=True)
    state = game.reset()
    viewer = state.units[1]
    enemy = state.units[2]
    viewer.coord = Coord(0, 4)
    enemy.coord = Coord(3, 4)
    assert game.is_unit_visible(0, enemy)

    for commander, low, high in (
        ("nell", 0, 19),
        ("flak", -9, 24),
        ("jugger", -14, 29),
        ("sonja", -9, 9),
    ):
        game = Game.from_map("duel", commanders={0: commander, 1: "andy"}, luck=True)
        state = game.reset(seed=3)
        attacker = state.units[1]
        defender = state.units[2]
        attacker.unit_type = "tank"
        attacker.coord = Coord(2, 1)
        defender.coord = Coord(3, 1)
        _state, events = game.step(
            Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
        )
        luck = next(event for event in events if event.type == "attack").payload["luck"]
        assert low <= luck <= high


def test_kindle_power_damages_enemy_units_on_urban_terrain():
    game = Game.from_map("duel", commanders={0: "kindle", 1: "andy"})
    state = game.reset()
    urban_enemy = state.units[2]
    urban_enemy.coord = Coord(5, 0)
    urban_enemy.hp = 80
    rural_id = max(state.units) + 1
    state.units[rural_id] = state.units[1].__class__(
        id=rural_id,
        owner=1,
        unit_type="infantry",
        coord=Coord(4, 4),
        fuel=99,
    )
    state.players[0].power_charge = 30

    game.step(Action(ActionType.CO_ABILITY))

    assert urban_enemy.hp == 50
    assert state.units[rural_id].hp == 100


def test_von_bolt_ex_machina_damages_and_stuns_next_turn():
    game = Game.from_map("duel", commanders={0: "von_bolt", 1: "andy"})
    state = game.reset()
    enemy = state.units[2]
    enemy.coord = Coord(4, 1)
    enemy.hp = 100
    friend = state.units[1]
    friend.coord = Coord(3, 1)
    state.players[0].power_charge = 100

    _state, events = game.step(Action(ActionType.CO_ABILITY))
    meteor = next(event for event in events if event.type == "meteor")

    assert meteor.payload["label"] == "ex_machina"
    assert enemy.hp == 70
    assert enemy.stunned_turns == 1
    assert friend.hp == 100

    game.step(Action.end_turn())
    assert enemy.can_act is False
    assert enemy.stunned_turns == 0


def test_sturm_meteor_and_rachel_covering_fire_create_meteor_events():
    game = Game.from_map("duel", commanders={0: "sturm", 1: "andy"})
    state = game.reset()
    enemy = state.units[2]
    enemy.unit_type = "megatank"
    enemy.coord = Coord(4, 4)
    enemy.hp = 100
    state.players[0].power_charge = 60

    _state, events = game.step(Action(ActionType.CO_ABILITY))
    meteor = next(event for event in events if event.type == "meteor")

    assert meteor.payload["label"] == "meteor_strike"
    assert meteor.payload["target"] == (4, 4)
    assert enemy.hp == 60

    game = Game.from_map("duel", commanders={0: "rachel", 1: "andy"})
    state = game.reset()
    state.units[2].coord = Coord(4, 1)
    state.players[0].power_charge = 60

    _state, events = game.step(
        Action(ActionType.CO_ABILITY, metadata={"power": "super"})
    )
    meteors = [event for event in events if event.type == "meteor"]

    assert [event.payload["label"] for event in meteors] == [
        "covering_fire_capture",
        "covering_fire_cost",
        "covering_fire_health",
    ]
    assert state.units[2].hp < 100


def test_rachel_covering_fire_preplans_all_targets_before_damage():
    game = Game.from_map("duel", commanders={0: "rachel", 1: "andy"})
    state = game.reset()
    enemy = state.units[2]
    enemy.coord = Coord(4, 1)
    enemy.hp = 100
    state.players[0].power_charge = 60
    planned_hp: list[int] = []

    def fake_plan_meteor_target(
        player_id: int,
        radius: int,
        power: int,
        value_kind: str,
        center_on_enemy: bool,
        self_harm: bool,
    ) -> Coord:
        planned_hp.append(enemy.hp)
        return enemy.coord

    game._plan_meteor_target = fake_plan_meteor_target  # type: ignore[method-assign]

    _state, events = game.step(
        Action(ActionType.CO_ABILITY, metadata={"power": "super"})
    )
    meteors = [event for event in events if event.type == "meteor"]

    assert planned_hp == [100, 100, 100]
    assert [event.payload["target"] for event in meteors] == [
        (4, 1),
        (4, 1),
        (4, 1),
    ]
    assert enemy.hp == 10


def test_sensei_powers_spawn_ready_footsoldiers_on_owned_empty_cities():
    game = Game.from_map("duel", commanders={0: "sensei", 1: "andy"})
    state = game.reset()
    state.map.tile_at(Coord(3, 0)).owner = 0
    state.players[0].power_charge = 20

    _state, events = game.step(Action(ActionType.CO_ABILITY))
    spawn = next(event for event in events if event.type == "spawn")
    infantry = state.units[spawn.payload["unit_id"]]

    assert infantry.unit_type == "infantry"
    assert infantry.hp == 90
    assert infantry.can_act is True

    game = Game.from_map("duel", commanders={0: "sensei", 1: "andy"})
    state = game.reset()
    state.map.tile_at(Coord(3, 0)).owner = 0
    state.players[0].power_charge = 60

    _state, events = game.step(
        Action(ActionType.CO_ABILITY, metadata={"power": "super"})
    )
    spawn = next(event for event in events if event.type == "spawn")
    mech = state.units[spawn.payload["unit_id"]]

    assert mech.unit_type == "mech"
    assert mech.hp == 90
    assert mech.can_act is True


def test_tag_mode_initializes_team_and_swaps_active_commander():
    game = Game.from_map(
        "duel",
        commanders={0: ["andy", "max"], 1: ["sami", "grit"]},
        tag_mode=True,
    )
    state = game.reset()

    assert state.players[0].commanders == ["andy", "max"]
    assert state.players[0].commander == "andy"
    assert any(action.type == ActionType.SWAP_CO for action in game.legal_actions())

    _state, events = game.step(Action(ActionType.SWAP_CO))

    assert state.players[0].commander == "max"
    assert state.players[0].active_commander_index == 1
    assert state.players[0].swapped_this_turn is True
    assert events[0].type == "swap_co"
    assert events[0].payload["from"] == "andy"
    assert events[0].payload["to"] == "max"


def test_swap_co_changes_active_commander_combat_modifiers():
    game = Game.from_map(
        "duel",
        commanders={0: ["andy", "max"], 1: "andy"},
        tag_mode=True,
    )
    state = game.reset()
    attacker = state.units[1]
    defender = state.units[2]
    attacker.unit_type = "tank"
    attacker.coord = Coord(2, 1)
    defender.coord = Coord(3, 1)

    game.step(Action(ActionType.SWAP_CO))
    _state, events = game.step(
        Action(ActionType.ATTACK, unit_id=attacker.id, target=defender.coord)
    )

    damage = next(event for event in events if event.type == "attack").payload["damage"]
    assert state.players[0].commander == "max"
    assert damage == 90


def test_swap_co_only_once_per_turn_and_resets_next_turn():
    game = Game.from_map(
        "duel",
        commanders={0: ["andy", "max"], 1: ["sami", "grit"]},
        tag_mode=True,
    )
    state = game.reset()

    game.step(Action(ActionType.SWAP_CO))
    assert not any(action.type == ActionType.SWAP_CO for action in game.legal_actions())

    game.step(Action.end_turn())
    game.step(Action.end_turn())

    assert state.current_player == 0
    assert any(action.type == ActionType.SWAP_CO for action in game.legal_actions())


def test_tag_mode_state_serializes_round_trip():
    game = Game.from_map(
        "duel",
        commanders={0: ["andy", "max"], 1: ["sami", "grit"]},
        tag_mode=True,
    )
    state = game.reset()
    game.step(Action(ActionType.SWAP_CO))

    restored = Game.from_json(json.loads(json.dumps(game.to_json())))

    assert restored.tag_mode is True
    assert restored.state is not None
    assert restored.state.players[0].commanders == ["andy", "max"]
    assert restored.state.players[0].commander == "max"
    assert restored.state.players[0].active_commander_index == 1
    assert restored.state.players[0].swapped_this_turn is True


def test_env_accepts_tag_mode_commander_configuration():
    env = raw_env(
        commanders={0: ["andy", "max"], 1: ["sami", "grit"]},
        tag_mode=True,
    )
    env.reset()

    assert env.game.state is not None
    assert env.game.state.players[0].commanders == ["andy", "max"]
    assert any(
        action.type == ActionType.SWAP_CO
        for action in env.infos["player_0"]["legal_actions"]
    )
