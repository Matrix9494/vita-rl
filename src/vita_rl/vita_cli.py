"""VitaBench CLI entry point with root-owned harness registration."""

from vita_rl.harness import register_standard_harness, register_stateful_harness


def main() -> None:
    register_standard_harness()
    register_stateful_harness()
    from vita.cli import main as vita_main

    vita_main()


if __name__ == "__main__":
    main()
