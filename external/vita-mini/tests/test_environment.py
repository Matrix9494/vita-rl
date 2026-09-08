import unittest

from vita_mini import MiniEnvironment


class MiniEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.env = MiniEnvironment()

    def test_agent_tools_are_openai_function_schemas(self):
        schemas = self.env.openai_tools()
        names = {schema["function"]["name"] for schema in schemas}
        self.assertEqual(names, set(self.env.tool_names()))
        add = next(schema for schema in schemas if schema["function"]["name"] == "add_to_cart")
        self.assertEqual(add["function"]["parameters"]["required"], ["product_id", "quantity"])

    def test_buy_coffee_is_deterministic_and_rule_evaluated(self):
        observation = self.env.reset("buy_coffee")
        self.assertIn("two cold brew", observation["user_message"].lower())
        self.assertTrue(self.env.call_tool("add_to_cart", {"product_id": "cold-brew", "quantity": 2}).ok)
        self.assertTrue(self.env.call_tool("set_delivery_address", {"address": "1 Mini Way"}).ok)
        order = self.env.call_tool("checkout", {})
        self.assertTrue(order.ok)
        self.assertEqual(order.result["status"], "submitted")
        evaluation = self.env.evaluate()
        self.assertTrue(evaluation.success)
        self.assertEqual(evaluation.reward, 1.0)

    def test_invalid_tool_use_is_a_structured_agent_visible_failure(self):
        self.env.reset("buy_coffee")
        result = self.env.call_tool("add_to_cart", {"product_id": "missing", "quantity": 1})
        self.assertFalse(result.ok)
        self.assertIn("Unknown product", result.error)

    def test_cancellation_task_restores_inventory(self):
        self.env.reset("cancel_order")
        inventory_before = self.env.call_tool("get_product", {"product_id": "bagel"}).result["inventory"]
        cancelled = self.env.call_tool("cancel_order", {"order_id": "mini-order-0001"})
        self.assertTrue(cancelled.ok)
        inventory_after = self.env.call_tool("get_product", {"product_id": "bagel"}).result["inventory"]
        self.assertEqual(inventory_after, inventory_before + 1)
        self.assertTrue(self.env.evaluate().success)


if __name__ == "__main__":
    unittest.main()
