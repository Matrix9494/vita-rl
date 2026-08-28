"""Small AWBW map catalog probe.

This module intentionally only reads the public category summary page. It does
not crawl individual map pages, which keeps benchmark setup polite and cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


AWBW_CATEGORIES_URL = "https://awbw.amarriner.com/categories.php"
AWBW_MAP_API_URL = "https://awbw.amarriner.com/api/map/map_info.php?maps_id={map_id}"
QUALITY_BUCKETS = frozenset({"S-Rank", "A-Rank", "B-Rank", "C-Rank", "New"})
DEFAULT_EXCLUDED_CATEGORIES = frozenset(
    {
        "FFA Multiplay",
        "Fog of War",
        "Gimmick",
        "HFOG",
        "High Funds",
        "Historical/Geographical",
        "Joke",
        "Sprite",
        "Team Play",
        "Teleport Tile",
        "Toy-Box",
    }
)


@dataclass(frozen=True)
class AWBWCategory:
    """A map category row from AWBW's public category page."""

    category_id: int
    name: str
    count: int
    group: str


@dataclass(frozen=True)
class AWBWMapListEntry:
    """A map row parsed from an AWBW category listing."""

    map_id: int
    name: str
    author: str | None
    categories: tuple[str, ...]
    category_ids: tuple[int, ...]
    player_count: int | None
    width: int | None
    height: int | None
    rating: float | None
    rating_votes: int | None
    comments: int | None
    favorites: int | None
    first_published: str | None
    last_published: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "map_id": self.map_id,
            "name": self.name,
            "author": self.author,
            "categories": list(self.categories),
            "category_ids": list(self.category_ids),
            "player_count": self.player_count,
            "width": self.width,
            "height": self.height,
            "rating": self.rating,
            "rating_votes": self.rating_votes,
            "comments": self.comments,
            "favorites": self.favorites,
            "first_published": self.first_published,
            "last_published": self.last_published,
        }


class _CategorySummaryParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._active_href: str | None = None
        self._items: list[tuple[str, str | None]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        self._active_href = href

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._active_href = None

    def handle_data(self, data: str) -> None:
        for raw_line in data.splitlines():
            text = " ".join(raw_line.split())
            if text:
                self._items.append((text, self._active_href))

    @property
    def items(self) -> list[tuple[str, str | None]]:
        return self._items


def _read_url(url: str, timeout: float, retries: int = 5) -> str:
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to read {url}")


def fetch_categories(
    url: str = AWBW_CATEGORIES_URL, timeout: float = 15
) -> list[AWBWCategory]:
    """Fetch and parse AWBW's public map category summary."""

    return parse_categories(_read_url(url, timeout))


def parse_categories(html: str) -> list[AWBWCategory]:
    """Parse AWBW category counts from ``categories.php`` HTML."""

    parser = _CategorySummaryParser()
    parser.feed(html)

    categories: list[AWBWCategory] = []
    current_group = ""
    pending: tuple[int, str, str] | None = None
    groups = {"Map Quality", "Map Function", "Map Features"}

    for text, href in parser.items:
        if pending and text.startswith("(") and "map" in text:
            count_text = text.strip("()").split()[0].replace(",", "")
            categories.append(
                AWBWCategory(
                    category_id=pending[0],
                    name=pending[1],
                    count=int(count_text),
                    group=pending[2],
                )
            )
            pending = None

        matched_group = next((group for group in groups if group in text), None)
        if matched_group is not None:
            current_group = matched_group
            pending = None
            continue

        if href and "categories.php?categories_id=" in href:
            category_id = int(href.rsplit("categories_id=", 1)[1].split("&", 1)[0])
            pending = (category_id, text, current_group)
            continue

    return categories


def quality_total(categories: list[AWBWCategory]) -> int:
    """Return the total over quality buckets, the closest public total count."""

    return sum(
        category.count for category in categories if category.name in QUALITY_BUCKETS
    )


def format_category_summary(categories: list[AWBWCategory]) -> str:
    """Render a compact text summary for logs or CLI output."""

    lines = [f"quality_total={quality_total(categories)}"]
    for group in ("Map Quality", "Map Function", "Map Features"):
        group_items = [category for category in categories if category.group == group]
        if not group_items:
            continue
        lines.append(group)
        for category in group_items:
            lines.append(
                f"  {category.category_id:>2} {category.name:<24} {category.count}"
            )
    return "\n".join(lines)


def fetch_category_maps(
    category_id: int,
    *,
    max_maps: int | None = None,
    timeout: float = 15,
    delay_seconds: float = 0.25,
) -> list[AWBWMapListEntry]:
    """Fetch map rows from an AWBW category listing."""

    maps: list[AWBWMapListEntry] = []
    seen: set[int] = set()
    start = 0

    while True:
        url = f"{AWBW_CATEGORIES_URL}?categories_id={category_id}"
        if start:
            url = f"{url}&start={start}"
        page_maps = parse_category_maps(_read_url(url, timeout))
        new_maps = [entry for entry in page_maps if entry.map_id not in seen]
        for entry in new_maps:
            seen.add(entry.map_id)
            maps.append(entry)
            if max_maps is not None and len(maps) >= max_maps:
                return maps
        if not page_maps or not new_maps:
            return maps
        start += 26
        if delay_seconds:
            time.sleep(delay_seconds)


def parse_category_maps(html: str) -> list[AWBWMapListEntry]:
    """Parse map rows from an AWBW category listing page."""

    parser = _CategorySummaryParser()
    parser.feed(html)

    entries: list[AWBWMapListEntry] = []
    current: dict[str, Any] | None = None
    expect_author = False

    for text, href in parser.items:
        map_id = _parse_id(href, "prevmaps.php?maps_id=")
        if map_id is not None:
            if current is not None:
                entries.append(_entry_from_partial(current))
            current = {
                "map_id": map_id,
                "name": text,
                "author": None,
                "categories": [],
                "category_ids": [],
            }
            expect_author = False
            continue

        if current is None:
            continue

        if text == "Creator:":
            expect_author = True
            continue
        if expect_author:
            current["author"] = text
            expect_author = False
            continue

        category_id = _parse_id(href, "categories.php?categories_id=")
        if category_id is not None:
            current["categories"].append(text)
            current["category_ids"].append(category_id)
            continue

        _update_entry_metadata(current, text)

    if current is not None:
        entries.append(_entry_from_partial(current))

    return entries


def select_default_maps(
    entries: list[AWBWMapListEntry],
    *,
    count: int,
    excluded_ids: set[int] | None = None,
) -> list[AWBWMapListEntry]:
    """Select deterministic 2p Standard clear maps for the first benchmark."""

    excluded_ids = excluded_ids or set()
    selected: list[AWBWMapListEntry] = []
    for entry in entries:
        if entry.map_id in excluded_ids:
            continue
        if not _is_default_benchmark_map(entry):
            continue
        selected.append(entry)
        if len(selected) >= count:
            return selected
    return selected


def build_default_dataset(
    output_dir: str | Path = "datasets/awbw_maps",
    *,
    train_count: int = 200,
    test_count: int = 20,
    timeout: float = 15,
    delay_seconds: float = 0.25,
) -> dict[str, Any]:
    """Build a small local AWBW map split and download raw map JSON files."""

    output = Path(output_dir)
    raw_dir = output / "raw"
    split_dir = output / "splits"
    raw_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    test_candidates = fetch_category_maps(
        25, timeout=timeout, delay_seconds=delay_seconds
    )
    test_maps = select_default_maps(test_candidates, count=test_count)
    if len(test_maps) < test_count:
        raise RuntimeError(f"Only selected {len(test_maps)} test maps.")

    train_pool: list[AWBWMapListEntry] = []
    for category_id in (42, 22, 38):
        train_pool.extend(
            fetch_category_maps(
                category_id,
                max_maps=max(train_count + test_count, 260),
                timeout=timeout,
                delay_seconds=delay_seconds,
            )
        )
        train_maps = select_default_maps(
            train_pool,
            count=train_count,
            excluded_ids={entry.map_id for entry in test_maps},
        )
        if len(train_maps) >= train_count:
            break

    if len(train_maps) < train_count:
        raise RuntimeError(f"Only selected {len(train_maps)} train maps.")

    manifest = {
        "metadata": {
            "source": "Advance Wars By Web public map pages",
            "source_url": AWBW_CATEGORIES_URL,
            "selection": (
                "2-player Standard clear maps; test from S-Rank, train from "
                "A/B/C-Rank; excludes fog, team/FFA, high-funds, gimmick, "
                "joke/sprite/toy-box, teleport, and historical categories."
            ),
            "train_count": train_count,
            "test_count": test_count,
        },
        "train": [entry.to_dict() for entry in train_maps],
        "test": [entry.to_dict() for entry in test_maps],
    }

    all_maps = train_maps + test_maps
    for index, entry in enumerate(all_maps, start=1):
        path = raw_dir / f"{entry.map_id}.json"
        if not path.exists():
            map_json = fetch_map_json(entry.map_id, timeout=timeout)
            path.write_text(json.dumps(map_json, indent=2, sort_keys=True))
            if delay_seconds:
                time.sleep(delay_seconds)
        print(f"[{index:03}/{len(all_maps)}] {entry.map_id} {entry.name}", flush=True)

    (split_dir / "train_200.json").write_text(
        json.dumps(manifest["train"], indent=2, sort_keys=True)
    )
    (split_dir / "test_20.json").write_text(
        json.dumps(manifest["test"], indent=2, sort_keys=True)
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def fetch_map_json(map_id: int, *, timeout: float = 15) -> dict[str, Any]:
    """Download one AWBW map JSON payload."""

    text = _read_url(AWBW_MAP_API_URL.format(map_id=map_id), timeout)
    return json.loads(text)


def _is_default_benchmark_map(entry: AWBWMapListEntry) -> bool:
    categories = set(entry.categories)
    if entry.player_count != 2:
        return False
    if entry.width is None or entry.height is None:
        return False
    if entry.width < 15 or entry.height < 15 or entry.width > 30 or entry.height > 30:
        return False
    if "Standard" not in categories:
        return False
    return not (categories & DEFAULT_EXCLUDED_CATEGORIES)


def _parse_id(href: str | None, prefix: str) -> int | None:
    if not href or prefix not in href:
        return None
    suffix = href.split(prefix, 1)[1]
    digits = []
    for char in suffix:
        if not char.isdigit():
            break
        digits.append(char)
    return int("".join(digits)) if digits else None


def _update_entry_metadata(entry: dict[str, Any], text: str) -> None:
    if match := re.match(r"Players: (\d+)", text):
        entry["player_count"] = int(match.group(1))
    elif match := re.match(r"Size: (\d+) x (\d+)", text):
        entry["width"] = int(match.group(1))
        entry["height"] = int(match.group(2))
    elif match := re.match(r"Rating: ([0-9.]+)/(\d+)", text):
        entry["rating"] = float(match.group(1))
        entry["rating_votes"] = int(match.group(2))
    elif match := re.match(r"Comments: (\d+)", text):
        entry["comments"] = int(match.group(1))
    elif match := re.match(r"Favorites: (\d+)", text):
        entry["favorites"] = int(match.group(1))
    elif text.startswith("First Published: "):
        entry["first_published"] = text.removeprefix("First Published: ")
    elif text.startswith("Last Published: "):
        entry["last_published"] = text.removeprefix("Last Published: ")


def _entry_from_partial(entry: dict[str, Any]) -> AWBWMapListEntry:
    return AWBWMapListEntry(
        map_id=entry["map_id"],
        name=entry["name"],
        author=entry.get("author"),
        categories=tuple(entry.get("categories", [])),
        category_ids=tuple(entry.get("category_ids", [])),
        player_count=entry.get("player_count"),
        width=entry.get("width"),
        height=entry.get("height"),
        rating=entry.get("rating"),
        rating_votes=entry.get("rating_votes"),
        comments=entry.get("comments"),
        favorites=entry.get("favorites"),
        first_published=entry.get("first_published"),
        last_published=entry.get("last_published"),
    )
