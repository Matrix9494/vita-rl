import unittest

from vita_mini import DifficultyConfig, MiniEnvironment, TaskGenerator
from vita_mini.oracle import verify_oracle_solution


class MiniEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.env = MiniEnvironment()

    def test_delivery_tools_are_openai_function_schemas(self):
        self.env.reset("delivery_revision")
        names = {schema["function"]["name"] for schema in self.env.openai_tools()}
        self.assertEqual(names, {
            "search_stores", "search_products", "get_store", "get_product",
            "create_order", "modify_order", "cancel_order", "pay_order", "get_order",
        })

    def test_revision_is_hidden_then_revealed_and_supersedes_prior_preference(self):
        initial = self.env.reset("delivery_revision")
        self.assertIn("cold brews", initial["user_message"])
        self.assertTrue(self.env.call_tool("search_products", {"query": "coffee"}).ok)
        events = self.env.next_user_event(tool_name="search_products", tool_success=True)
        self.assertEqual(len(events), 1)
        self.assertIn("cannot have caffeine", events[0]["message"])
        self.assertEqual(self.env.snapshot()["user"]["current_preferences"]["product_id"], "product-herbal-tea")

    def test_composite_objective_requires_modify_then_pay(self):
        self.env.reset("delivery_revision")
        self.env.call_tool("search_products", {"query": "coffee"})
        self.env.next_user_event(tool_name="search_products", tool_success=True)
        created = self.env.call_tool("create_order", {
            "store_id": "store-target", "product_id": "product-herbal-tea", "quantity": 2,
            "address": "home", "delivery_time": "2026-01-01 12:00:00",
        })
        self.assertTrue(created.ok)
        self.env.next_user_event(tool_name="create_order", tool_success=True)
        self.assertFalse(self.env.evaluate().success)
        order_id = created.result["order_id"]
        self.assertTrue(self.env.call_tool("modify_order", {"order_id": order_id, "address": "office"}).ok)
        self.assertTrue(self.env.call_tool("pay_order", {"order_id": order_id}).ok)
        evaluation = self.env.evaluate()
        self.assertTrue(evaluation.success)
        self.assertEqual(evaluation.reward, 1.0)
        self.assertEqual(evaluation.constraint_score, 1.0)

    def test_seeded_generator_is_deterministic_and_oracle_feasible(self):
        config = DifficultyConfig(num_constraints=8, num_distractors=3, num_revisions=1)
        first = TaskGenerator(seed=123).generate("generated", config)
        second = TaskGenerator(seed=123).generate("generated", config)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertTrue(verify_oracle_solution(first))


if __name__ == "__main__":
    unittest.main()
