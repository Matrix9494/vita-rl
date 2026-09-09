"""Regression tests for the standalone delayed-recall RL scaffold."""

import unittest

from memory_1 import Memory1Config, Memory1Env


class Memory1EnvTest(unittest.TestCase):
    def test_reset_is_seeded_and_cue_is_then_hidden(self):
        first = Memory1Env()
        second = Memory1Env()
        observation_a, _ = first.reset(seed=9)
        observation_b, _ = second.reset(seed=9)
        self.assertEqual(observation_a, observation_b)
        self.assertLess(observation_a["cue"], first.config.num_items)

        observation, reward, terminated, truncated, _ = first.step(0)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(observation["cue"], first.config.num_items)

    def test_query_reward_uses_the_remembered_cue(self):
        env = Memory1Env(Memory1Config(num_items=4, delay_steps=2))
        observation, _ = env.reset(seed=4)
        target = observation["cue"]
        env.step(0)  # cue -> delay
        env.step(1)  # delay
        observation, reward, terminated, truncated, _ = env.step(2)  # delay -> query
        self.assertEqual(observation["phase"], 2)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

        _, reward, terminated, truncated, info = env.step(target)
        self.assertEqual(reward, 1.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["correct"])
        self.assertEqual(info["target_action"], target)

    def test_invalid_and_post_terminal_steps_raise(self):
        env = Memory1Env(Memory1Config(delay_steps=0))
        env.reset(seed=2)
        with self.assertRaises(ValueError):
            env.step(99)
        env.step(0)  # cue -> query
        env.step(0)  # terminal query
        with self.assertRaises(RuntimeError):
            env.step(0)


if __name__ == "__main__":
    unittest.main()
