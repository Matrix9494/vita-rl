"""Commander data and lightweight combat modifiers."""

from __future__ import annotations

from dataclasses import dataclass

from advancewars.engine.data import WeaponDef
from advancewars.engine.state import UnitState


@dataclass(frozen=True)
class CommanderDef:
    key: str
    name: str
    power_name: str
    power_cost: int
    super_power_name: str | None = None
    super_power_cost: int | None = None
    attack_bonus: int = 0
    defense_bonus: int = 0
    capture_multiplier: int = 100
    power_attack_bonus: int = 10
    power_defense_bonus: int = 10
    power_heal: int = 0
    power_funds_multiplier: int = 100
    power_duration_turns: int = 1


COMMANDERS: dict[str, CommanderDef] = {
    "andy": CommanderDef(
        key="andy",
        name="Andy",
        power_name="Hyper Repair",
        power_cost=30,
        super_power_name="Hyper Upgrade",
        super_power_cost=60,
        power_heal=20,
    ),
    "max": CommanderDef(
        key="max",
        name="Max",
        power_name="Max Force",
        power_cost=30,
        super_power_name="Max Blast",
        super_power_cost=60,
    ),
    "sami": CommanderDef(
        key="sami",
        name="Sami",
        power_name="Double Time",
        power_cost=30,
        super_power_name="Victory March",
        super_power_cost=80,
        capture_multiplier=150,
    ),
    "grit": CommanderDef(
        key="grit",
        name="Grit",
        power_name="Snipe Attack",
        power_cost=30,
        super_power_name="Super Snipe",
        super_power_cost=60,
    ),
    "colin": CommanderDef(
        key="colin",
        name="Colin",
        power_name="Gold Rush",
        power_cost=20,
        super_power_name="Power of Money",
        super_power_cost=60,
        attack_bonus=-10,
        power_funds_multiplier=150,
    ),
    "kanbei": CommanderDef(
        key="kanbei",
        name="Kanbei",
        power_name="Morale Boost",
        power_cost=40,
        super_power_name="Samurai Spirit",
        super_power_cost=70,
        attack_bonus=30,
        defense_bonus=30,
    ),
    "grimm": CommanderDef(
        key="grimm",
        name="Grimm",
        power_name="Knuckleduster",
        power_cost=30,
        super_power_name="Haymaker",
        super_power_cost=60,
        attack_bonus=30,
        defense_bonus=-20,
        power_attack_bonus=20,
        power_defense_bonus=10,
    ),
    "jake": CommanderDef(
        key="jake",
        name="Jake",
        power_name="Beat Down",
        power_cost=30,
        super_power_name="Block Rock",
        super_power_cost=60,
    ),
    "eagle": CommanderDef(
        key="eagle",
        name="Eagle",
        power_name="Lightning Drive",
        power_cost=30,
        super_power_name="Lightning Strike",
        super_power_cost=90,
    ),
    "drake": CommanderDef(
        key="drake",
        name="Drake",
        power_name="Tsunami",
        power_cost=40,
        super_power_name="Typhoon",
        super_power_cost=70,
    ),
    "jess": CommanderDef(
        key="jess",
        name="Jess",
        power_name="Turbo Charge",
        power_cost=30,
        super_power_name="Overdrive",
        super_power_cost=60,
    ),
    "olaf": CommanderDef(
        key="olaf",
        name="Olaf",
        power_name="Blizzard",
        power_cost=30,
        super_power_name="Winter Fury",
        super_power_cost=70,
    ),
    "sasha": CommanderDef(
        key="sasha",
        name="Sasha",
        power_name="Market Crash",
        power_cost=20,
        super_power_name="War Bonds",
        super_power_cost=60,
    ),
    "rachel": CommanderDef(
        key="rachel",
        name="Rachel",
        power_name="Lucky Lass",
        power_cost=30,
        super_power_name="Covering Fire",
        super_power_cost=60,
    ),
    "hachi": CommanderDef(
        key="hachi",
        name="Hachi",
        power_name="Barter",
        power_cost=30,
        super_power_name="Merchant Union",
        super_power_cost=50,
    ),
    "sensei": CommanderDef(
        key="sensei",
        name="Sensei",
        power_name="Copter Command",
        power_cost=20,
        super_power_name="Airborne Assault",
        super_power_cost=60,
    ),
    "hawke": CommanderDef(
        key="hawke",
        name="Hawke",
        power_name="Black Wave",
        power_cost=50,
        super_power_name="Black Storm",
        super_power_cost=90,
        attack_bonus=10,
    ),
    "adder": CommanderDef(
        key="adder",
        name="Adder",
        power_name="Sideslip",
        power_cost=20,
        super_power_name="Sidewinder",
        super_power_cost=50,
    ),
    "von_bolt": CommanderDef(
        key="von_bolt",
        name="Von Bolt",
        power_name="Ex Machina",
        power_cost=100,
        attack_bonus=10,
        defense_bonus=10,
    ),
    "sonja": CommanderDef(
        key="sonja",
        name="Sonja",
        power_name="Enhanced Vision",
        power_cost=30,
        super_power_name="Counter Break",
        super_power_cost=50,
    ),
    "lash": CommanderDef(
        key="lash",
        name="Lash",
        power_name="Terrain Tactics",
        power_cost=40,
        super_power_name="Prime Tactics",
        super_power_cost=70,
    ),
    "koal": CommanderDef(
        key="koal",
        name="Koal",
        power_name="Forced March",
        power_cost=30,
        super_power_name="Trail of Woe",
        super_power_cost=50,
    ),
    "kindle": CommanderDef(
        key="kindle",
        name="Kindle",
        power_name="Urban Blight",
        power_cost=30,
        super_power_name="High Society",
        super_power_cost=60,
    ),
    "javier": CommanderDef(
        key="javier",
        name="Javier",
        power_name="Tower Shield",
        power_cost=30,
        super_power_name="Tower of Power",
        super_power_cost=60,
    ),
    "flak": CommanderDef(
        key="flak",
        name="Flak",
        power_name="Brute Force",
        power_cost=30,
        super_power_name="Barbaric Blow",
        super_power_cost=60,
    ),
    "jugger": CommanderDef(
        key="jugger",
        name="Jugger",
        power_name="Overclock",
        power_cost=30,
        super_power_name="System Crash",
        super_power_cost=70,
    ),
    "sturm": CommanderDef(
        key="sturm",
        name="Sturm",
        power_name="Meteor Strike",
        power_cost=60,
        attack_bonus=-20,
        defense_bonus=20,
    ),
    "nell": CommanderDef(
        key="nell",
        name="Nell",
        power_name="Lucky Star",
        power_cost=30,
        super_power_name="Lady Luck",
        super_power_cost=60,
    ),
}


def commander_for(key: str) -> CommanderDef:
    if key not in COMMANDERS:
        raise ValueError(f"Unknown commander: {key}")
    return COMMANDERS[key]


def power_cost(commander: CommanderDef, power_kind: str = "power") -> int:
    if power_kind == "power":
        return commander.power_cost
    if power_kind == "super" and commander.super_power_cost is not None:
        return commander.super_power_cost
    raise ValueError(f"{commander.name} does not have power kind: {power_kind}")


def max_power_cost(commander: CommanderDef) -> int:
    return max(commander.power_cost, commander.super_power_cost or commander.power_cost)


def power_name(commander: CommanderDef, power_kind: str = "power") -> str:
    if power_kind == "power":
        return commander.power_name
    if power_kind == "super" and commander.super_power_name is not None:
        return commander.super_power_name
    raise ValueError(f"{commander.name} does not have power kind: {power_kind}")


def power_heal(commander: CommanderDef, power_kind: str = "power") -> int:
    if commander.key == "andy" and power_kind == "super":
        return 50
    return commander.power_heal


def is_super(power_kind: str | None) -> bool:
    return power_kind == "super"


def is_troop(unit: UnitState) -> bool:
    return unit.definition.unit_class == "troop"


def is_transport(unit: UnitState) -> bool:
    return unit.definition.cargo_capacity > 0 and not unit.definition.weapons


def is_indirect(weapon: WeaponDef) -> bool:
    return weapon.max_range > 1


def is_air(unit: UnitState) -> bool:
    return unit.definition.move_type == "flight"


def is_sea(unit: UnitState) -> bool:
    return unit.definition.move_type in {"float_light", "float_heavy"}


def is_land_vehicle(unit: UnitState) -> bool:
    return unit.definition.unit_class == "land"


def is_copter(unit: UnitState) -> bool:
    return unit.unit_type in {"b_copter", "t_copter"}


def attack_bonus(
    commander: CommanderDef,
    active: bool,
    unit: UnitState,
    weapon: WeaponDef,
    terrain: str | None = None,
    terrain_stars: int = 0,
    owned_properties: int = 0,
    power_kind: str = "power",
    is_counter: bool = False,
) -> int:
    bonus = commander.attack_bonus
    if commander.key == "max" and not is_troop(unit):
        bonus += -10 if is_indirect(weapon) else 20
        if active and not is_indirect(weapon):
            bonus += 10
    elif commander.key == "kanbei":
        if active:
            bonus += 10
        if active and is_super(power_kind) and is_counter:
            bonus += 65
    elif commander.key == "sami":
        if is_troop(unit):
            bonus += 30
            if active:
                bonus += 40 if is_super(power_kind) else 20
        elif not is_indirect(weapon):
            bonus -= 10
    elif commander.key == "grit":
        if is_indirect(weapon):
            bonus += 20
            if active:
                bonus += 40 if is_super(power_kind) else 20
        elif not is_troop(unit):
            bonus -= 20
    elif commander.key == "jake":
        if terrain == "grass":
            bonus += 10
            if active:
                bonus += 20 if is_super(power_kind) else 10
    elif commander.key == "eagle":
        if is_air(unit):
            bonus += 15
            if active:
                bonus += 5
        elif is_sea(unit):
            bonus -= 30
    elif commander.key == "drake":
        if is_air(unit):
            bonus -= 30
    elif commander.key == "jess":
        if is_land_vehicle(unit):
            bonus += 10
            if active:
                bonus += 20 if is_super(power_kind) else 10
        else:
            bonus -= 10
    elif commander.key == "sensei":
        if is_copter(unit) or is_troop(unit):
            bonus += 50
        if not is_air(unit):
            bonus -= 10
    elif commander.key == "lash":
        if not is_air(unit):
            bonus += terrain_stars * 10
    elif commander.key == "koal":
        if terrain == "road":
            bonus += 10
            if active:
                bonus += 20 if is_super(power_kind) else 10
    elif commander.key == "kindle":
        if terrain in URBAN_TERRAINS:
            bonus += 40
            if active:
                bonus += 90 if is_super(power_kind) else 40
        if active and is_super(power_kind):
            bonus += owned_properties * 3
    elif commander.key == "andy" and active and is_super(power_kind):
        bonus += 10
    elif commander.key == "colin" and active and is_super(power_kind):
        bonus += owned_properties * 3
    if active:
        bonus += commander.power_attack_bonus
    return bonus


def defense_bonus(
    commander: CommanderDef,
    active: bool,
    unit: UnitState,
    distance: int = 1,
    power_kind: str = "power",
) -> int:
    bonus = commander.defense_bonus
    if commander.key == "eagle" and is_air(unit):
        bonus += 10
        if active:
            bonus += 10
    elif commander.key == "drake" and is_sea(unit):
        bonus += 10
    elif commander.key == "javier" and distance > 1:
        if active and is_super(power_kind):
            bonus += 80
        else:
            bonus += 40 if active else 20
    elif commander.key == "kanbei" and active and is_super(power_kind):
        bonus += 20
    if active:
        bonus += commander.power_defense_bonus
    return bonus


def capture_multiplier(
    commander: CommanderDef,
    _unit: UnitState,
    active: bool = False,
    power_kind: str = "power",
) -> int:
    if commander.key == "sami" and active and is_super(power_kind):
        return 10_000
    return commander.capture_multiplier


def move_bonus(
    commander: CommanderDef,
    active: bool,
    unit: UnitState,
    power_kind: str = "power",
) -> int:
    if commander.key == "max" and active and not is_troop(unit):
        return 1
    if commander.key == "sami":
        if is_transport(unit):
            return 1
        if active and is_troop(unit):
            return 2 if is_super(power_kind) else 1
    if commander.key == "drake" and is_sea(unit):
        return 1
    if commander.key == "jess" and active and is_land_vehicle(unit):
        return 2 if is_super(power_kind) else 1
    if commander.key == "sensei" and is_transport(unit):
        return 1
    if commander.key == "adder" and active:
        return 2 if is_super(power_kind) else 1
    if commander.key == "koal" and active:
        return 2 if is_super(power_kind) else 1
    if commander.key == "andy" and active and is_super(power_kind):
        return 1
    return 0


def weapon_range(
    commander: CommanderDef,
    active: bool,
    weapon: WeaponDef,
    power_kind: str = "power",
) -> tuple[int, int]:
    min_range = weapon.min_range
    max_range = weapon.max_range
    if commander.key == "max" and max_range > 1:
        max_range -= 1
    elif commander.key == "grit" and max_range > 1:
        max_range += 1
        if active:
            max_range += 2 if is_super(power_kind) else 1
    elif commander.key == "jake" and active and max_range > 1:
        max_range += 1
    return min_range, max(min_range, max_range)


def unit_cost(commander: CommanderDef, active: bool, unit_cost_value: int) -> int:
    if commander.key == "colin":
        return unit_cost_value * 80 // 100
    if commander.key == "kanbei":
        return unit_cost_value * 120 // 100
    if commander.key == "hachi":
        return unit_cost_value * (50 if active else 90) // 100
    return unit_cost_value


def idle_fuel_burn(commander: CommanderDef, unit: UnitState, burn: int) -> int:
    if commander.key == "eagle" and is_air(unit):
        return max(0, burn - 2)
    return burn


def repair_power(commander: CommanderDef) -> int:
    if commander.key == "rachel":
        return 30
    return 20


def income_per_property(commander: CommanderDef, base_income: int) -> int:
    if commander.key == "sasha" and base_income > 0:
        return base_income + 100
    return base_income


def weather_for_movement(commander: CommanderDef, weather: str) -> str:
    if commander.key == "drake" and weather == "rain":
        return "clear"
    if commander.key == "olaf":
        if weather == "snow":
            return "clear"
        if weather == "rain":
            return "snow"
    return weather


URBAN_TERRAINS = {
    "city",
    "hq",
    "factory",
    "airport",
    "temp_airport",
    "seaport",
    "temp_seaport",
    "lab",
    "ds_tower",
    "dor_tower",
}


def vision_bonus(
    commander: CommanderDef,
    active: bool,
    power_kind: str = "power",
) -> int:
    if commander.key == "sonja":
        return 2 if active else 1
    return 0


def luck_range(
    commander: CommanderDef,
    active: bool,
    power_kind: str = "power",
) -> tuple[int, int]:
    if commander.key == "nell":
        if active and is_super(power_kind):
            return (0, 99)
        return (0, 59 if active else 19)
    if commander.key == "rachel" and active:
        return (0, 39)
    if commander.key == "flak":
        if active and is_super(power_kind):
            return (-39, 89)
        return (-19, 49) if active else (-9, 24)
    if commander.key == "jugger":
        if active and is_super(power_kind):
            return (-44, 94)
        return (-24, 54) if active else (-14, 29)
    if commander.key == "sonja":
        return (-9, 9)
    return (0, 9)


def has_perfect_movement(
    commander: CommanderDef,
    active: bool,
    weather: str,
    power_kind: str = "power",
) -> bool:
    if weather == "snow":
        return False
    return commander.key == "sturm" or (commander.key == "lash" and active)


def counter_damage_multiplier(commander: CommanderDef) -> int:
    if commander.key == "sonja":
        return 150
    return 100


def preemptive_counter(
    commander: CommanderDef,
    active: bool,
    power_kind: str = "power",
) -> bool:
    return commander.key == "sonja" and active and is_super(power_kind)


def can_build_from_city(
    commander: CommanderDef,
    active: bool,
    unit_type: str,
    power_kind: str = "power",
) -> bool:
    ground_or_foot = unit_type in {
        "infantry",
        "mech",
        "apc",
        "artillery",
        "recon",
        "tank",
        "md_tank",
        "neotank",
        "megatank",
        "rockets",
        "anti_air",
        "mobilesam",
        "piperunner",
    }
    return commander.key == "hachi" and active and is_super(power_kind) and ground_or_foot
