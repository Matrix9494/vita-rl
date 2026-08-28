"""Run a random legal-action rollout with the AEC-style API."""

from __future__ import annotations

import random

from advancewars import raw_env


def main() -> None:
    env = raw_env(render_mode="ansi", max_turns=10)
    env.reset(seed=0)

    for step, agent in enumerate(env.agent_iter(max_iter=100)):
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            break
        legal_indices = [
            index for index, value in enumerate(observation["action_mask"]) if value
        ]
        action = random.choice(legal_indices)
        env.step(action)
        if step % 10 == 0:
            print(f"step={step} agent={agent} reward={reward}")

    print(env.render())


if __name__ == "__main__":
    main()
