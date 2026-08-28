import json

from advancewars.training import run_iteration


def test_scripted_training_iteration_smoke(tmp_path):
    output = tmp_path / "selfplay"

    summary = run_iteration(
        iteration=0,
        output_root=output,
        game_count=1,
        max_steps=10,
        max_turns=3,
        seed=0,
    )

    iteration_dir = output / "iter_0000"
    saved = json.loads((iteration_dir / "summary.json").read_text())

    assert summary["metadata"]["learner"] == "A"
    assert saved["metadata"]["game_count"] == 1
    assert (iteration_dir / "policy_A.yaml").exists()
    assert (iteration_dir / "policy_B.yaml").exists()
    assert (iteration_dir / "next_policy_A.yaml").exists()
    assert (iteration_dir / "next_policy_B.yaml").exists()
    assert (iteration_dir / "codex_improvement_request.md").exists()
    assert (iteration_dir / "diagnostics.json").exists()
    assert len(list((iteration_dir / "games").glob("*.json"))) == 1
