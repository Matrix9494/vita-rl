from advancewars.eval.awbw_maps import (
    parse_category_maps,
    format_category_summary,
    parse_categories,
    select_default_maps,
    quality_total,
)


SAMPLE_HTML = """
<html><body>
Design Map Categories
Map Quality
<a href="categories.php?categories_id=25">S-Rank</a> (69 maps)
<a href="categories.php?categories_id=1">A-Rank</a> (1,105 maps)
<a href="categories.php?categories_id=2">B-Rank</a> (7780 maps)
<a href="categories.php?categories_id=3">C-Rank</a> (3851 maps)
<a href="categories.php?categories_id=24">New</a> (9 maps)
Map Function
<a href="categories.php?categories_id=20">Global League</a> (42 maps)
Map Features
<a href="categories.php?categories_id=26">Standard</a> (9408 maps)
</body></html>
"""


def test_parse_awbw_category_summary():
    categories = parse_categories(SAMPLE_HTML)

    assert categories[0].name == "S-Rank"
    assert categories[0].category_id == 25
    assert categories[0].count == 69
    assert categories[0].group == "Map Quality"
    assert categories[-1].name == "Standard"
    assert categories[-1].group == "Map Features"
    assert quality_total(categories) == 12814


def test_format_awbw_category_summary():
    text = format_category_summary(parse_categories(SAMPLE_HTML))

    assert "quality_total=12814" in text
    assert "Map Quality" in text
    assert "Standard" in text


def test_parse_awbw_category_map_rows():
    html = """
    <a href=prevmaps.php?maps_id=123>Test Map</a>
    Creator: <a href="profile.php?username=Alice">Alice</a>
    Players: 2
    Size: 21 x 19
    First Published: 01/02/2020
    Last Published: 01/03/2021
    Rating: 6.25/12
    Comments: 7
    Favorites: 42
    <a href="categories.php?categories_id=42">A-Rank</a>
    <a href="categories.php?categories_id=41">Standard</a>
    <a href=prevmaps.php?maps_id=456>Fog Map</a>
    Creator: <a href="profile.php?username=Bob">Bob</a>
    Players: 2
    Size: 21 x 19
    <a href="categories.php?categories_id=25">S-Rank</a>
    <a href="categories.php?categories_id=34">Fog of War</a>
    <a href="categories.php?categories_id=41">Standard</a>
    """

    entries = parse_category_maps(html)

    assert len(entries) == 2
    assert entries[0].map_id == 123
    assert entries[0].author == "Alice"
    assert entries[0].player_count == 2
    assert entries[0].width == 21
    assert entries[0].height == 19
    assert entries[0].rating == 6.25
    assert entries[0].rating_votes == 12
    assert entries[0].favorites == 42
    assert entries[0].categories == ("A-Rank", "Standard")

    selected = select_default_maps(entries, count=10)
    assert [entry.map_id for entry in selected] == [123]
