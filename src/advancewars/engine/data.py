"""Built-in DefendPeace/AWBW rule data.

The tables here mirror the AWBW subset in DefendPeace closely enough for the
current Python engine. Complex behavior such as hiding, transforms, cargo, CO
powers, and fuel burn is represented as data but not fully simulated yet.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TerrainDef:
    key: str
    code: str
    defense: int
    capturable: bool = False
    profitable: bool = False
    heals_land: bool = False
    heals_air: bool = False
    heals_sea: bool = False
    vision_boost: int = 0
    provides_cover: bool = False


@dataclass(frozen=True)
class WeaponDef:
    name: str
    min_range: int
    max_range: int
    max_ammo: int | None
    damage: dict[str, int]
    can_fire_after_moving: bool = True


@dataclass(frozen=True)
class UnitDef:
    key: str
    name: str
    cost: int
    move: int
    move_type: str
    vision: int
    max_fuel: int
    weapons: tuple[WeaponDef, ...] = ()
    can_capture: bool = False
    build_from: tuple[str, ...] = ("factory",)
    unit_class: str = "land"
    cargo_capacity: int = 0
    carry_classes: tuple[str, ...] = ()
    transform_to: str | None = None
    max_hp: int = 100
    fuel_burn_idle: int = 0
    fuel_burn_per_tile: int = 1

    @property
    def max_ammo_by_weapon(self) -> dict[str, int]:
        return {w.name: w.max_ammo for w in self.weapons if w.max_ammo is not None}


TERRAIN_BY_CODE: dict[str, TerrainDef] = {
    "SE": TerrainDef("sea", "SE", 0),
    "SH": TerrainDef("shoal", "SH", 0),
    "GR": TerrainDef("grass", "GR", 1),
    "PL": TerrainDef("grass", "PL", 1),
    "FR": TerrainDef("forest", "FR", 2, provides_cover=True),
    "RF": TerrainDef("reef", "RF", 1, provides_cover=True),
    "RD": TerrainDef("road", "RD", 0),
    "BR": TerrainDef("bridge", "BR", 0),
    "RV": TerrainDef("river", "RV", 0),
    "MT": TerrainDef("mountain", "MT", 4, vision_boost=3),
    "DN": TerrainDef("dunes", "DN", 1),
    "CT": TerrainDef("city", "CT", 3, True, True, True),
    "HQ": TerrainDef("hq", "HQ", 4, True, True, True),
    "FC": TerrainDef("factory", "FC", 3, True, True, True),
    "AP": TerrainDef("airport", "AP", 3, True, True, False, True),
    "TA": TerrainDef("temp_airport", "TA", 1, True, False, False, True),
    "SP": TerrainDef("seaport", "SP", 3, True, True, False, False, True),
    "TS": TerrainDef("temp_seaport", "TS", 1, True, False, False, False, True),
    "LB": TerrainDef("lab", "LB", 3, True),
    "TT": TerrainDef("teletile", "TT", 0),
    "XX": TerrainDef("teletile", "TT", 0),
    "PI": TerrainDef("pillar", "PI", 0),
    "ME": TerrainDef("meteor", "ME", 0),
    "SR": TerrainDef("missile_silo", "SR", 3),
    "BK": TerrainDef("spent_missile_silo", "BK", 3),
    "TW": TerrainDef("ds_tower", "TW", 3, True),
    "T3": TerrainDef("ds_tower", "TW", 3, True),
    "T4": TerrainDef("dor_tower", "T4", 3, True, True),
}

TERRAIN_BY_KEY: dict[str, TerrainDef] = {}
for terrain in TERRAIN_BY_CODE.values():
    TERRAIN_BY_KEY.setdefault(terrain.key, terrain)


UNIT_ENUM = (
    "infantry",
    "mech",
    "recon",
    "tank",
    "md_tank",
    "neotank",
    "megatank",
    "apc",
    "artillery",
    "rockets",
    "piperunner",
    "anti_air",
    "mobilesam",
    "fighter",
    "bomber",
    "stealth",
    "stealth_hide",
    "b_copter",
    "t_copter",
    "bbomb",
    "carrier",
    "bboat",
    "battleship",
    "cruiser",
    "lander",
    "sub",
    "sub_sub",
)

AWBW_DAMAGE_ROWS: dict[str, tuple[int, ...]] = {
    "INFANTRYMGUN": (55, 45, 12, 5, 1, 1, 1, 14, 15, 25, 5, 5, 25, 0, 0, 0, 0, 7, 30, 0, 0, 0, 0, 0, 0, 0, 0),
    "MECHZOOKA": (0, 0, 85, 55, 15, 15, 5, 75, 70, 85, 55, 65, 85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    "MECHMGUN": (65, 55, 18, 6, 1, 1, 1, 20, 32, 35, 6, 6, 35, 0, 0, 0, 0, 9, 35, 0, 0, 0, 0, 0, 0, 0, 0),
    "RECONMGUN": (70, 65, 35, 6, 1, 1, 1, 45, 45, 55, 6, 4, 28, 0, 0, 0, 0, 10, 35, 0, 0, 0, 0, 0, 0, 0, 0),
    "TANKCANNON": (25, 25, 85, 55, 15, 15, 10, 75, 70, 85, 55, 65, 85, 0, 0, 0, 0, 0, 0, 0, 1, 10, 1, 5, 10, 1, 0),
    "TANKMGUN": (75, 70, 40, 6, 1, 1, 1, 45, 45, 55, 6, 5, 30, 0, 0, 0, 0, 10, 40, 0, 0, 0, 0, 0, 0, 0, 0),
    "MD_TANKCANNON": (30, 30, 105, 85, 55, 45, 25, 105, 105, 105, 85, 105, 105, 0, 0, 0, 0, 0, 0, 0, 10, 35, 10, 45, 35, 10, 0),
    "MD_TANKMGUN": (105, 95, 45, 8, 1, 1, 1, 45, 45, 55, 8, 7, 35, 0, 0, 0, 0, 12, 45, 0, 0, 0, 0, 0, 0, 0, 0),
    "NEOCANNON": (35, 35, 125, 105, 75, 55, 35, 125, 115, 125, 105, 115, 125, 0, 0, 0, 0, 0, 0, 0, 15, 40, 15, 50, 40, 15, 0),
    "NEOMGUN": (125, 115, 65, 10, 1, 1, 1, 65, 65, 75, 10, 17, 55, 0, 0, 0, 0, 22, 55, 0, 0, 0, 0, 0, 0, 0, 0),
    "MEGACANNON": (42, 42, 195, 180, 125, 115, 65, 195, 195, 195, 180, 195, 195, 0, 0, 0, 0, 0, 0, 0, 45, 105, 45, 65, 75, 45, 0),
    "MEGAMGUN": (135, 125, 65, 10, 1, 1, 1, 65, 65, 75, 10, 17, 55, 0, 0, 0, 0, 22, 55, 0, 0, 0, 0, 0, 0, 0, 0),
    "ARTILLERYCANNON": (90, 85, 80, 70, 45, 40, 15, 70, 75, 80, 70, 75, 80, 0, 0, 0, 0, 0, 0, 0, 45, 55, 40, 65, 55, 60, 0),
    "ROCKETS": (95, 90, 90, 80, 55, 50, 25, 80, 80, 85, 80, 85, 90, 0, 0, 0, 0, 0, 0, 0, 60, 60, 55, 85, 60, 85, 0),
    "PIPEGUN": (95, 90, 90, 80, 55, 50, 25, 80, 80, 85, 80, 85, 90, 65, 75, 75, 0, 105, 105, 120, 60, 60, 55, 60, 60, 85, 0),
    "ANTI_AIRMGUN": (105, 105, 60, 25, 10, 5, 1, 50, 50, 55, 25, 45, 55, 65, 75, 75, 0, 120, 120, 120, 0, 0, 0, 0, 0, 0, 0),
    "MOBILESAM": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 100, 0, 120, 120, 120, 0, 0, 0, 0, 0, 0, 0),
    "FIGHTERMISSILES": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 100, 85, 85, 100, 100, 120, 0, 0, 0, 0, 0, 0, 0),
    "BOMBERBOMBS": (110, 110, 105, 105, 95, 90, 35, 105, 105, 105, 105, 95, 105, 0, 0, 0, 0, 0, 0, 0, 105, 75, 75, 85, 95, 95, 0),
    "STEALTH_SHOTS": (90, 90, 85, 75, 70, 60, 15, 85, 75, 85, 80, 50, 85, 45, 70, 55, 55, 85, 95, 120, 45, 65, 45, 35, 65, 55, 0),
    "B_COPTERROCKETS": (0, 0, 55, 55, 25, 20, 10, 60, 65, 65, 55, 25, 65, 0, 0, 0, 0, 0, 0, 0, 25, 25, 25, 55, 25, 25, 0),
    "B_COPTERMGUN": (75, 75, 30, 6, 1, 1, 0, 20, 25, 35, 6, 6, 35, 0, 0, 0, 0, 65, 95, 0, 0, 0, 0, 0, 0, 0, 0),
    "CARRIERMISSILES": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 100, 0, 115, 115, 120, 0, 0, 0, 0, 0, 0, 0),
    "BATTLESHIPCANNON": (95, 90, 90, 80, 55, 50, 25, 80, 80, 85, 80, 85, 90, 0, 0, 0, 0, 0, 0, 0, 60, 95, 50, 95, 95, 95, 0),
    "CRUISERTORPEDOES": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 25, 0, 0, 0, 90, 90),
    "CRUISERMGUN": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 65, 100, 0, 115, 115, 120, 0, 0, 0, 0, 0, 0, 0),
    "SUBTORPEDOES": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 75, 95, 55, 25, 95, 55, 55),
}

WEAPON_SPECS = {
    "InfantryMGun": ("INFANTRYMGUN", 1, 1, False),
    "MechZooka": ("MECHZOOKA", 1, 1, True),
    "MechMGun": ("MECHMGUN", 1, 1, False),
    "ReconMGun": ("RECONMGUN", 1, 1, False),
    "TankCannon": ("TANKCANNON", 1, 1, True),
    "TankMGun": ("TANKMGUN", 1, 1, False),
    "MDTankCannon": ("MD_TANKCANNON", 1, 1, True),
    "MDTankMGun": ("MD_TANKMGUN", 1, 1, False),
    "NeoCannon": ("NEOCANNON", 1, 1, True),
    "NeoMGun": ("NEOMGUN", 1, 1, False),
    "MegaCannon": ("MEGACANNON", 1, 1, True),
    "MegaMGun": ("MEGAMGUN", 1, 1, False),
    "ArtilleryCannon": ("ARTILLERYCANNON", 2, 3, True),
    "RocketRockets": ("ROCKETS", 3, 5, True),
    "PipeGun": ("PIPEGUN", 2, 5, True),
    "AntiAirMGun": ("ANTI_AIRMGUN", 1, 1, True),
    "MobileSAMWeapon": ("MOBILESAM", 3, 5, True),
    "FighterMissiles": ("FIGHTERMISSILES", 1, 1, True),
    "BomberBombs": ("BOMBERBOMBS", 1, 1, True),
    "StealthShots": ("STEALTH_SHOTS", 1, 1, True),
    "CopterRockets": ("B_COPTERROCKETS", 1, 1, True),
    "CopterMGun": ("B_COPTERMGUN", 1, 1, False),
    "CarrierMissiles": ("CARRIERMISSILES", 3, 8, True),
    "BattleshipCannon": ("BATTLESHIPCANNON", 2, 6, True),
    "CruiserTorpedoes": ("CRUISERTORPEDOES", 1, 1, True),
    "CruiserMGun": ("CRUISERMGUN", 1, 1, False),
    "SubTorpedoes": ("SUBTORPEDOES", 1, 1, True),
}


def _damage(weapon_type: str) -> dict[str, int]:
    return dict(zip(UNIT_ENUM, AWBW_DAMAGE_ROWS[weapon_type]))


def _weapon(class_name: str, ammo: int) -> WeaponDef:
    weapon_type, min_range, max_range, uses_ammo = WEAPON_SPECS[class_name]
    return WeaponDef(
        name=class_name,
        min_range=min_range,
        max_range=max_range,
        max_ammo=ammo if uses_ammo else None,
        damage=_damage(weapon_type),
        can_fire_after_moving=min_range == 1,
    )


def _unit(
    key: str,
    name: str,
    cost: int,
    move: int,
    move_type: str,
    vision: int,
    max_fuel: int,
    max_ammo: int,
    weapons: tuple[str, ...] = (),
    can_capture: bool = False,
    build_from: tuple[str, ...] = ("factory",),
    unit_class: str = "land",
    cargo_capacity: int = 0,
    carry_classes: tuple[str, ...] = (),
    transform_to: str | None = None,
    fuel_burn_idle: int = 0,
    fuel_burn_per_tile: int = 1,
) -> UnitDef:
    return UnitDef(
        key=key,
        name=name,
        cost=cost,
        move=move,
        move_type=move_type,
        vision=vision,
        max_fuel=max_fuel,
        weapons=tuple(_weapon(w, max_ammo) for w in weapons),
        can_capture=can_capture,
        build_from=build_from,
        unit_class=unit_class,
        cargo_capacity=cargo_capacity,
        carry_classes=carry_classes,
        transform_to=transform_to,
        fuel_burn_idle=fuel_burn_idle,
        fuel_burn_per_tile=fuel_burn_per_tile,
    )


UNITS: dict[str, UnitDef] = {
    "infantry": _unit("infantry", "Infantry", 1000, 3, "foot_standard", 2, 99, -1, ("InfantryMGun",), True, unit_class="troop"),
    "mech": _unit("mech", "Mech", 3000, 2, "foot_mech", 2, 99, 3, ("MechZooka", "MechMGun"), True, unit_class="troop"),
    "apc": _unit("apc", "APC", 5000, 6, "tread", 1, 70, -1, cargo_capacity=1, carry_classes=("troop",)),
    "recon": _unit("recon", "Recon", 4000, 8, "tires", 5, 80, -1, ("ReconMGun",)),
    "tank": _unit("tank", "Tank", 7000, 6, "tread", 3, 70, 9, ("TankCannon", "TankMGun")),
    "md_tank": _unit("md_tank", "Md Tank", 16000, 5, "tread", 1, 50, 8, ("MDTankCannon", "MDTankMGun")),
    "neotank": _unit("neotank", "Neotank", 22000, 6, "tread", 1, 99, 9, ("NeoCannon", "NeoMGun")),
    "megatank": _unit("megatank", "Megatank", 28000, 4, "tread", 1, 50, 3, ("MegaCannon", "MegaMGun")),
    "artillery": _unit("artillery", "Artillery", 6000, 5, "tread", 1, 50, 9, ("ArtilleryCannon",)),
    "rockets": _unit("rockets", "Rockets", 15000, 5, "tires", 1, 50, 6, ("RocketRockets",)),
    "piperunner": _unit("piperunner", "Piperunner", 20000, 9, "pipe", 4, 99, 9, ("PipeGun",)),
    "anti_air": _unit("anti_air", "Anti-Air", 8000, 6, "tread", 2, 60, 9, ("AntiAirMGun",)),
    "mobilesam": _unit("mobilesam", "Missiles", 12000, 4, "tires", 5, 50, 9, ("MobileSAMWeapon",)),
    "t_copter": _unit("t_copter", "T-Copter", 5000, 6, "flight", 2, 99, -1, build_from=("airport",), unit_class="air", cargo_capacity=1, carry_classes=("troop",), fuel_burn_idle=2),
    "b_copter": _unit("b_copter", "B-Copter", 9000, 6, "flight", 3, 99, 6, ("CopterRockets", "CopterMGun"), build_from=("airport",), unit_class="air", fuel_burn_idle=2),
    "fighter": _unit("fighter", "Fighter", 20000, 9, "flight", 2, 99, 9, ("FighterMissiles",), build_from=("airport",), unit_class="air", fuel_burn_idle=5),
    "bomber": _unit("bomber", "Bomber", 22000, 7, "flight", 2, 99, 9, ("BomberBombs",), build_from=("airport",), unit_class="air", fuel_burn_idle=5),
    "stealth": _unit("stealth", "Stealth", 24000, 6, "flight", 4, 60, 6, ("StealthShots",), build_from=("airport",), unit_class="air", transform_to="stealth_hide", fuel_burn_idle=5),
    "stealth_hide": _unit("stealth_hide", "Stealth", 24000, 6, "flight", 4, 60, 6, ("StealthShots",), build_from=(), unit_class="air", transform_to="stealth", fuel_burn_idle=8),
    "bbomb": _unit("bbomb", "BBomb", 25000, 9, "flight", 1, 45, -1, build_from=("airport",), unit_class="air", fuel_burn_idle=5),
    "lander": _unit("lander", "Lander", 12000, 6, "float_light", 1, 99, -1, build_from=("seaport",), unit_class="sea", cargo_capacity=2, carry_classes=("troop", "land"), fuel_burn_idle=1),
    "cruiser": _unit("cruiser", "Cruiser", 18000, 6, "float_heavy", 3, 99, 9, ("CruiserTorpedoes", "CruiserMGun"), build_from=("seaport",), unit_class="sea", cargo_capacity=2, carry_classes=("air",), fuel_burn_idle=1),
    "sub": _unit("sub", "Sub", 20000, 5, "float_heavy", 5, 60, 6, ("SubTorpedoes",), build_from=("seaport",), unit_class="sea", transform_to="sub_sub", fuel_burn_idle=1),
    "sub_sub": _unit("sub_sub", "Sub", 20000, 5, "float_heavy", 5, 60, 6, ("SubTorpedoes",), build_from=(), unit_class="sea", transform_to="sub", fuel_burn_idle=5),
    "battleship": _unit("battleship", "Battleship", 28000, 5, "float_heavy", 2, 99, 9, ("BattleshipCannon",), build_from=("seaport",), unit_class="sea", fuel_burn_idle=1),
    "carrier": _unit("carrier", "Carrier", 30000, 5, "float_heavy", 4, 99, 9, ("CarrierMissiles",), build_from=("seaport",), unit_class="sea", cargo_capacity=2, carry_classes=("air",), fuel_burn_idle=1),
    "bboat": _unit("bboat", "BBoat", 7500, 7, "float_light", 1, 60, -1, build_from=("seaport",), unit_class="sea", cargo_capacity=2, carry_classes=("troop",), fuel_burn_idle=1),
}

UNIT_ALIASES = {
    key: key for key in UNITS
} | {
    definition.name.lower().replace("-", "_").replace(" ", "_"): key
    for key, definition in UNITS.items()
} | {
    "mdtank": "md_tank",
    "md_tank": "md_tank",
    "medium_tank": "md_tank",
    "anti_air": "anti_air",
    "antiair": "anti_air",
    "missiles": "mobilesam",
    "mobile_sam": "mobilesam",
    "tcopter": "t_copter",
    "t_copter": "t_copter",
    "bcopter": "b_copter",
    "b_copter": "b_copter",
    "black_bomb": "bbomb",
    "black_boat": "bboat",
}

BUILD_LISTS = {
    "factory": (
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
    ),
    "airport": ("t_copter", "b_copter", "fighter", "bomber", "stealth", "bbomb"),
    "seaport": ("lander", "cruiser", "sub", "battleship", "carrier", "bboat"),
}

WEATHERS = ("clear", "rain", "snow")

LAND_COMMON = {
    "grass": 1,
    "forest": 1,
    "road": 1,
    "bridge": 1,
    "city": 1,
    "hq": 1,
    "factory": 1,
    "airport": 1,
    "temp_airport": 1,
    "seaport": 1,
    "temp_seaport": 1,
    "lab": 1,
    "shoal": 1,
    "dunes": 1,
    "missile_silo": 1,
    "spent_missile_silo": 1,
    "ds_tower": 1,
    "dor_tower": 1,
}

CLEAR_MOVEMENT_COSTS: dict[str, dict[str, int | None]] = {
    "foot_standard": LAND_COMMON | {"mountain": 2, "river": 2},
    "foot_mech": LAND_COMMON,
    "foot": LAND_COMMON | {"mountain": 2, "river": 2},
    "tread": LAND_COMMON | {"forest": 2, "dunes": 2},
    "tires": LAND_COMMON | {"grass": 2, "forest": 3, "dunes": 3},
    "pipe": {"pillar": 1, "meteor": 1, "factory": 1},
    "flight": {
        terrain: 1 for terrain in TERRAIN_BY_KEY if terrain not in {"pillar", "meteor"}
    },
    "float_light": {"sea": 1, "reef": 2, "shoal": 1, "seaport": 1, "temp_seaport": 1},
    "float_heavy": {"sea": 1, "reef": 2, "seaport": 1, "temp_seaport": 1},
}

RAIN_MOVEMENT_COSTS: dict[str, dict[str, int | None]] = {
    **CLEAR_MOVEMENT_COSTS,
    "tread": CLEAR_MOVEMENT_COSTS["tread"] | {"grass": 2, "forest": 3},
    "tires": CLEAR_MOVEMENT_COSTS["tires"] | {"grass": 3, "forest": 4},
}

SNOW_MOVEMENT_COSTS: dict[str, dict[str, int | None]] = {
    **CLEAR_MOVEMENT_COSTS,
    "foot_standard": CLEAR_MOVEMENT_COSTS["foot_standard"]
    | {"grass": 2, "forest": 2, "mountain": 4},
    "foot_mech": CLEAR_MOVEMENT_COSTS["foot_mech"] | {"mountain": 2},
    "foot": CLEAR_MOVEMENT_COSTS["foot"] | {"grass": 2, "forest": 2, "mountain": 4},
    "tread": CLEAR_MOVEMENT_COSTS["tread"] | {"grass": 2},
    "tires": CLEAR_MOVEMENT_COSTS["tires"] | {"grass": 3},
    "flight": {
        terrain: 2 for terrain in TERRAIN_BY_KEY if terrain not in {"pillar", "meteor"}
    },
    "float_light": CLEAR_MOVEMENT_COSTS["float_light"] | {"sea": 2},
    "float_heavy": CLEAR_MOVEMENT_COSTS["float_heavy"] | {"sea": 2},
}

MOVEMENT_COSTS_BY_WEATHER: dict[str, dict[str, dict[str, int | None]]] = {
    "clear": CLEAR_MOVEMENT_COSTS,
    "rain": RAIN_MOVEMENT_COSTS,
    "snow": SNOW_MOVEMENT_COSTS,
}
MOVEMENT_COSTS = CLEAR_MOVEMENT_COSTS

RULESETS = {
    "defendpeace_awbw": {
        "income_per_city": 1000,
        "starting_funds": 0,
        "unit_cap": 50,
        "capture_threshold": 20,
        "unit_scheme": "AWBWUnits",
    }
}

BUILTIN_MAPS = {
    "duel": """
 0HQ 0FC  GR  CT  GR 1FC 1HQ
  GR  RD  RD  RD  RD  RD  GR
  GR  CT  GR  MT  GR  CT  GR
  GR  RD  RD  RD  RD  RD  GR
  GR  GR  GR  CT  GR  GR  GR

0, Infantry, 0, 1
1, Infantry, 6, 1
""".strip("\n"),
    "triangle": """
 0HQ  GR  CT  GR 1HQ
  GR  RD  MT  RD  GR
 2HQ  GR  CT  GR  FC

0, Infantry, 0, 1
1, Infantry, 4, 1
2, Infantry, 0, 2
""".strip("\n"),
}
