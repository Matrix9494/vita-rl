"""Battle report image rendering for saved rollout trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


TERRAIN_COLORS = {
    "grass": "#a8d46f",
    "road": "#c6b49a",
    "bridge": "#b7a789",
    "city": "#d4d0c8",
    "hq": "#e5d8ad",
    "factory": "#c7c7c7",
    "airport": "#cfd9e6",
    "seaport": "#b5d2df",
    "mountain": "#9a8a70",
    "forest": "#4f9b5d",
    "sea": "#5aa6d6",
    "shoal": "#d6c07d",
    "river": "#72b8db",
    "reef": "#4f83aa",
    "missile_silo": "#d9c37b",
    "spent_missile_silo": "#a89472",
}
DEFAULT_TERRAIN_COLOR = "#b8c49b"
PLAYER_COLORS = {0: "#e84a5f", 1: "#3178c6", 2: "#3aa86d", 3: "#c8842d"}
NEUTRAL_COLOR = "#7a7a7a"
TEXT_COLOR = "#222222"
MUTED_TEXT = "#666666"


def render_battle_report(
    trajectory_or_path: str | Path | dict[str, Any],
    output_path: str | Path,
    *,
    columns: int = 2,
    tile_size: int = 34,
    include_initial: bool = False,
    max_event_text: int = 72,
) -> Path:
    """Render a saved rollout JSON as one concatenated PNG battle report."""
    trajectory = _load_trajectory(trajectory_or_path)
    frames = _frames_from_trajectory(trajectory, include_initial=include_initial)
    if not frames:
        raise ValueError("Trajectory has no renderable frames.")

    fonts = _fonts()
    first_state = frames[0]["state"]
    board_width = int(first_state["map"]["width"]) * tile_size
    board_height = int(first_state["map"]["height"]) * tile_size
    panel_padding = 12
    header_height = 88
    footer_height = 46
    panel_width = board_width + panel_padding * 2
    panel_height = board_height + header_height + footer_height + panel_padding
    title_height = 78
    gap = 16
    columns = max(1, columns)
    rows = (len(frames) + columns - 1) // columns

    image_width = columns * panel_width + (columns + 1) * gap
    image_height = title_height + rows * panel_height + (rows + 1) * gap
    image = Image.new("RGB", (image_width, image_height), "#f4f0e8")
    draw = ImageDraw.Draw(image)

    _draw_title(draw, trajectory, len(frames), image_width, fonts)
    for index, frame in enumerate(frames):
        row = index // columns
        col = index % columns
        x = gap + col * (panel_width + gap)
        y = title_height + gap + row * (panel_height + gap)
        _draw_panel(
            draw,
            frame,
            x,
            y,
            panel_width,
            board_width,
            board_height,
            panel_padding,
            header_height,
            tile_size,
            max_event_text,
            fonts,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def _load_trajectory(trajectory_or_path: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(trajectory_or_path, dict):
        return trajectory_or_path
    return json.loads(Path(trajectory_or_path).read_text())


def _frames_from_trajectory(
    trajectory: dict[str, Any],
    *,
    include_initial: bool,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    if include_initial:
        initial = trajectory["initial"]["game"]
        frames.append(
            {
                "label": "Initial",
                "agent": "-",
                "action": None,
                "events": [],
                "reward": {},
                "state": initial["state"],
            }
        )
    for step in trajectory.get("steps", []):
        if "state_after" not in step:
            continue
        frames.append(
            {
                "label": f"Step {int(step['step']):02d}",
                "agent": step.get("agent", "-"),
                "action": step.get("selected_action"),
                "events": step.get("events", []),
                "reward": step.get("rewards_after", {}),
                "state": step["state_after"]["state"],
            }
        )
    return frames


def _fonts() -> dict[str, ImageFont.ImageFont]:
    def truetype(size: int, bold: bool = False) -> ImageFont.ImageFont:
        names = (
            "DejaVuSans-Bold.ttf",
            "DejaVuSans.ttf",
        ) if bold else (
            "DejaVuSans.ttf",
            "DejaVuSans-Bold.ttf",
        )
        for name in names:
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    return {
        "title": truetype(24, bold=True),
        "panel": truetype(16, bold=True),
        "body": truetype(13),
        "small": truetype(11),
        "unit": truetype(13, bold=True),
        "hp": truetype(9, bold=True),
    }


def _draw_title(
    draw: ImageDraw.ImageDraw,
    trajectory: dict[str, Any],
    frame_count: int,
    image_width: int,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    metadata = trajectory.get("metadata", {})
    final_state = trajectory.get("final", {}).get("game", {}).get("state", {})
    winner = final_state.get("winner")
    title = f"AdvanceWars Duel Battle Report - {frame_count} boards"
    subtitle = (
        f"map={metadata.get('map_name')} config={metadata.get('config')} "
        f"seed={metadata.get('seed')} winner={winner}"
    )
    draw.rectangle((0, 0, image_width, 78), fill="#2f3437")
    draw.text((18, 14), title, fill="#ffffff", font=fonts["title"])
    draw.text((20, 48), subtitle, fill="#d6d6d6", font=fonts["body"])


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    frame: dict[str, Any],
    x: int,
    y: int,
    panel_width: int,
    board_width: int,
    board_height: int,
    padding: int,
    header_height: int,
    tile_size: int,
    max_event_text: int,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    state = frame["state"]
    panel_height = board_height + header_height + 46 + padding
    draw.rounded_rectangle(
        (x, y, x + panel_width, y + panel_height),
        radius=8,
        fill="#fffaf1",
        outline="#d0c3ad",
        width=1,
    )
    _draw_panel_header(draw, frame, state, x + padding, y + 10, fonts)
    board_x = x + padding
    board_y = y + header_height
    _draw_board(draw, state, board_x, board_y, tile_size, fonts)
    _draw_panel_footer(
        draw,
        frame,
        board_x,
        board_y + board_height + 8,
        board_width,
        max_event_text,
        fonts,
    )


def _draw_panel_header(
    draw: ImageDraw.ImageDraw,
    frame: dict[str, Any],
    state: dict[str, Any],
    x: int,
    y: int,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    action = frame.get("action")
    action_text = _action_text(action)
    draw.text((x, y), frame["label"], fill=TEXT_COLOR, font=fonts["panel"])
    draw.text(
        (x, y + 22),
        f"{frame['agent']}  {action_text}",
        fill=TEXT_COLOR,
        font=fonts["body"],
    )
    funds = "  ".join(
        f"P{pid}: ${player['funds']} U{_unit_count(state, int(pid))}"
        for pid, player in sorted(state["players"].items())
    )
    draw.text(
        (x, y + 42),
        f"turn={state['turn']} current=P{state['current_player']}  {funds}",
        fill=MUTED_TEXT,
        font=fonts["small"],
    )


def _draw_board(
    draw: ImageDraw.ImageDraw,
    state: dict[str, Any],
    x: int,
    y: int,
    tile_size: int,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    tiles = state["map"]["tiles"]
    for tile_y, row in enumerate(tiles):
        for tile_x, tile in enumerate(row):
            left = x + tile_x * tile_size
            top = y + tile_y * tile_size
            right = left + tile_size
            bottom = top + tile_size
            terrain = tile["terrain"]
            owner = tile["owner"]
            color = TERRAIN_COLORS.get(terrain, DEFAULT_TERRAIN_COLOR)
            draw.rectangle((left, top, right, bottom), fill=color, outline="#f5f5f5")
            if owner is not None:
                owner_color = PLAYER_COLORS.get(int(owner), NEUTRAL_COLOR)
                draw.rectangle((left, top, left + 5, bottom), fill=owner_color)
            draw.text(
                (left + 7, top + 3),
                _terrain_label(terrain),
                fill="#333333",
                font=fonts["small"],
            )
    for unit in sorted(state["units"].values(), key=lambda item: item["id"]):
        _draw_unit(draw, unit, x, y, tile_size, fonts)


def _draw_unit(
    draw: ImageDraw.ImageDraw,
    unit: dict[str, Any],
    x: int,
    y: int,
    tile_size: int,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    if unit.get("carried_by") is not None or int(unit.get("hp", 0)) <= 0:
        return
    coord = unit["coord"]
    left = x + int(coord["x"]) * tile_size
    top = y + int(coord["y"]) * tile_size
    owner = int(unit["owner"])
    fill = PLAYER_COLORS.get(owner, NEUTRAL_COLOR)
    cx = left + tile_size // 2
    cy = top + tile_size // 2 + 2
    radius = max(9, tile_size // 3)
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=fill,
        outline="#222222",
        width=2,
    )
    label = _unit_label(unit["unit_type"])
    _center_text(draw, (cx, cy - 1), label, "#ffffff", fonts["unit"])
    hp_text = str(int(unit["hp"]) // 10)
    _center_text(
        draw,
        (cx + radius - 2, cy + radius - 2),
        hp_text,
        "#ffffff",
        fonts["hp"],
    )


def _draw_panel_footer(
    draw: ImageDraw.ImageDraw,
    frame: dict[str, Any],
    x: int,
    y: int,
    width: int,
    max_event_text: int,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    reward = frame.get("reward", {})
    reward_text = " ".join(
        f"{agent}:{value:g}" for agent, value in sorted(reward.items())
    )
    events = _event_text(frame.get("events", []), max_event_text)
    draw.text(
        (x, y),
        _clip(reward_text or "reward: -", width, draw, fonts["small"]),
        fill=MUTED_TEXT,
        font=fonts["small"],
    )
    draw.text(
        (x, y + 18),
        _clip(events or "events: -", width, draw, fonts["small"]),
        fill=MUTED_TEXT,
        font=fonts["small"],
    )


def _action_text(action: dict[str, Any] | None) -> str:
    if action is None:
        return "-"
    action_type = action.get("type")
    unit = action.get("unit_id")
    target = action.get("target")
    build = action.get("build_unit")
    if build:
        return f"{action_type} {build} at {_coord_text(target)}"
    if target:
        return f"{action_type} u{unit} -> {_coord_text(target)}"
    path = action.get("path") or []
    if path:
        return f"{action_type} u{unit} -> {_coord_text(path[-1])}"
    if unit is not None:
        return f"{action_type} u{unit}"
    return str(action_type)


def _event_text(events: list[dict[str, Any]], max_chars: int) -> str:
    parts = []
    for event in events:
        payload = event.get("payload", {})
        text = event.get("type", "")
        if "unit_id" in payload:
            text += f" u{payload['unit_id']}"
        if "unit_type" in payload:
            text += f" {payload['unit_type']}"
        if "damage" in payload:
            text += f" dmg={payload['damage']}"
        if "coord" in payload:
            text += f" at {payload['coord']}"
        if "to" in payload:
            text += f" to {payload['to']}"
        parts.append(text)
    return _shorten("; ".join(parts), max_chars)


def _terrain_label(terrain: str) -> str:
    labels = {
        "grass": "GR",
        "road": "RD",
        "city": "CT",
        "hq": "HQ",
        "factory": "FC",
        "mountain": "MT",
        "forest": "FR",
        "sea": "SE",
        "shoal": "SH",
    }
    return labels.get(terrain, terrain[:2].upper())


def _unit_label(unit_type: str) -> str:
    labels = {
        "infantry": "Inf",
        "mech": "Mech",
        "recon": "Rec",
        "tank": "Tank",
        "artillery": "Art",
        "apc": "APC",
    }
    return labels.get(unit_type, unit_type[:4].title())


def _coord_text(coord: dict[str, Any] | None) -> str:
    if coord is None:
        return "-"
    return f"({coord['x']},{coord['y']})"


def _unit_count(state: dict[str, Any], player_id: int) -> int:
    return sum(
        1
        for unit in state["units"].values()
        if int(unit["owner"]) == player_id
        and unit.get("carried_by") is None
        and int(unit.get("hp", 0)) > 0
    )


def _center_text(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    text: str,
    fill: str,
    font: ImageFont.ImageFont,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text(
        (center[0] - width / 2, center[1] - height / 2),
        text,
        fill=fill,
        font=font,
    )


def _clip(
    text: str,
    width: int,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
) -> str:
    if draw.textlength(text, font=font) <= width:
        return text
    clipped = text
    while clipped and draw.textlength(clipped + "...", font=font) > width:
        clipped = clipped[:-1]
    return clipped + "..."


def _shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
