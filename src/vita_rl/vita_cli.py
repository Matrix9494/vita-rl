"""VitaBench CLI entry point with root-owned harness registration."""

from vita_rl.harness import (
    register_standard_harness,
    register_state_delta_harness,
    register_stateful_harness,
    register_recent_turns_harness,
    register_summary_harness,
)
from vita_rl.deterministic_user import register_deterministic_task_user
from vita_rl.autonomous_user_runner import install_autonomous_deterministic_runner


def main() -> None:
    register_deterministic_task_user()
    install_autonomous_deterministic_runner()
    register_standard_harness()
    register_stateful_harness()
    register_summary_harness()
    register_recent_turns_harness()
    register_state_delta_harness()
    from vita.cli import main as vita_main

    vita_main()


if __name__ == "__main__":
    main()
