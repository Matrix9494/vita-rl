from pathlib import Path

from advancewars.render import render_battle_report
from examples.heuristic_duel_rollout import run_rollout


def test_render_battle_report_writes_png(tmp_path: Path):
    trajectory_path = tmp_path / "rollout.json"
    report_path = tmp_path / "report.png"
    run_rollout(
        output_path=trajectory_path,
        config="no_production",
        max_steps=4,
        max_turns=4,
        seed=0,
    )

    output = render_battle_report(trajectory_path, report_path, columns=2)

    assert output == report_path
    assert report_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert report_path.stat().st_size > 0
