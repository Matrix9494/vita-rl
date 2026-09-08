"""Compatibility module; use :mod:`vita_rl.runtime_server` for new code."""

from vita_rl.runtime_server import (
    EpisodeRequest,
    EpisodeResponse,
    EpisodeValidationError,
    create_app,
    main,
    run_environment_episode,
)


# Existing launchers imported this function by its Vita-specific name.
run_vita_episode = run_environment_episode


if __name__ == "__main__":
    main()
