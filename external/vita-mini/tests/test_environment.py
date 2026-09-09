import unittest

from vita_mini import DifficultyConfig, MiniEnvironment, TaskGenerator
from vita_mini.oracle import verify_oracle_solution


class MiniEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.env = MiniEnvironment()

    def test_delivery_tools_are_openai_function_schemas(self):
        self.env.reset("delivery_revision")
        schemas = self.env.openai_tools()
        names = {schema["function"]["name"] for schema in schemas}
        native = {
            "search_stores", "search_products", "get_store", "get_product",
            "create_order", "modify_order", "cancel_order", "pay_order", "get_order",
        }
        self.assertEqual(names, native)
        self.assertEqual(len(schemas), 9)
        for schema in schemas:
            self.assertEqual(schema["type"], "function")
            parameters = schema["function"]["parameters"]
            self.assertEqual(parameters["type"], "object")
            self.assertIn("additionalProperties", parameters)
            self.assertFalse(parameters["additionalProperties"])

        by_name = {schema["function"]["name"]: schema["function"] for schema in schemas}
        self.assertIn("not search product inventory", by_name["search_stores"]["description"])
        create_properties = by_name["create_order"]["parameters"]["properties"]
        self.assertEqual(create_properties["address"]["enum"], ["1 Home Street", "2 Office Plaza", "home", "office"])
        self.assertEqual(create_properties["delivery_time"]["pattern"], r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
        self.assertIn("2026-01-01 09:00:00", create_properties["delivery_time"]["description"])

    def test_revision_is_hidden_then_revealed_and_supersedes_prior_preference(self):
        initial = self.env.reset("delivery_revision")
        self.assertIn("cold brews", initial["user_message"])
        self.assertTrue(self.env.call_tool("search_products", {"query": "coffee"}).ok)
        events = self.env.next_user_event(tool_name="search_products", tool_success=True)
        self.assertEqual(len(events), 1)
        self.assertIn("cannot have caffeine", events[0]["message"])
        self.assertIn("under $10", events[0]["message"])
        self.assertEqual(self.env.snapshot()["user"]["current_preferences"]["product_id"], "product-herbal-tea")
        self.assertEqual(self.env.snapshot()["user"]["current_preferences"]["max_price_cents"], 1000)

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

    def test_procedural_task_identifier_constructs_a_replayable_environment(self):
        initial = self.env.reset("generated:123")
        self.assertEqual(initial["task_id"], "generated:123")
        first_snapshot = self.env.snapshot()
        final_product_id = first_snapshot["user"]["latent_constraints"]["correct_product"]
        first_price = first_snapshot["database"]["products"][final_product_id]["price_cents"]
        self.env.reset("generated:123")
        second_snapshot = self.env.snapshot()
        self.assertEqual(second_snapshot["user"]["latent_constraints"]["correct_product"], final_product_id)
        second_price = second_snapshot["database"]["products"][final_product_id]["price_cents"]
        self.assertEqual(first_price, second_price)

    def test_procedural_generation_varies_catalogues_and_action_graphs(self):
        tasks = [TaskGenerator(seed=seed).generate(f"generated:{seed}") for seed in range(40)]
        self.assertGreaterEqual(len({task.difficulty["scenario_index"] for task in tasks}), 4)
        self.assertGreaterEqual(len({task.initial_message for task in tasks}), 20)
        self.assertTrue(all(task.difficulty["num_revisions"] == 2 for task in tasks))
        self.assertTrue(all(task.difficulty["num_distractors"] == 4 for task in tasks))
        final_products = {
            next(constraint.expected for constraint in task.latent_constraints if constraint.constraint_id == "correct_product")
            for task in tasks
        }
        self.assertGreaterEqual(len(final_products), 5)
        for task in tasks:
            final_product_id = next(constraint.expected for constraint in task.latent_constraints if constraint.constraint_id == "correct_product")
            final_name = task.initial_state.products[final_product_id].name
            self.assertGreaterEqual(sum(product.name == final_name for product in task.initial_state.products.values()), 2)
        self.assertTrue(all(verify_oracle_solution(task) for task in tasks))

    def test_memory_families_defer_constraints_until_the_decisive_order(self):
        tasks = [TaskGenerator(seed=seed).generate(f"generated:{seed}") for seed in range(100)]
        memory_tasks = {
            task.metadata["task_family"]: task
            for task in tasks
            if task.metadata["task_family"].startswith("memory_")
        }
        self.assertEqual(set(memory_tasks), {"memory_delayed", "memory_accumulate", "memory_partial"})
        memory_products = {
            next(constraint.expected for constraint in task.latent_constraints if constraint.constraint_id == "correct_product")
            for task in tasks if task.metadata["task_family"].startswith("memory_")
        }
        self.assertEqual({product_id.split("-")[1] for product_id in memory_products}, {"tea", "fruit", "oat", "soup", "water"})

        for family, task in memory_tasks.items():
            event_index = {event.event_id: index for index, event in enumerate(task.user_script)}
            requirements = task.metadata["memory_requirements"]
            self.assertGreaterEqual(task.metadata["retention_distance"], 2)
            self.assertTrue(all(not requirement["db_recoverable_before_use"] for requirement in requirements))
            for requirement in requirements:
                self.assertEqual(
                    requirement["retention_distance"],
                    event_index[requirement["required_event"]] - event_index[requirement["last_disclosure_event"]],
                )
                self.assertEqual(requirement["disclosure_event_index"], event_index[requirement["disclosure_event"]])
                self.assertEqual(requirement["required_event_index"], event_index[requirement["required_event"]])
                if requirement["persistent"]:
                    self.assertEqual(requirement["required_action"], "create_order")

            env = MiniEnvironment(tasks={task.task_id: task})
            env.reset(task.task_id)
            first = env.call_tool("search_products", {"query": "delivery"})
            self.assertTrue(first.ok)
            first_events = env.next_user_event(tool_name="search_products", tool_success=True)
            self.assertEqual(len(first_events), 1)
            self.assertEqual(env.snapshot()["database"]["orders"], {})

            final_product_id = next(
                constraint.expected for constraint in task.latent_constraints if constraint.constraint_id == "correct_product"
            )
            second = env.call_tool("search_products", {"query": task.initial_state.products[final_product_id].name})
            self.assertTrue(second.ok)
            second_events = env.next_user_event(tool_name="search_products", tool_success=True)
            self.assertTrue(second_events)
            if family == "memory_delayed":
                self.assertEqual(second_events[0]["event_id"], "finalize")
                final_event = second_events[0]
            else:
                third = env.call_tool("get_product", {"product_id": final_product_id})
                self.assertTrue(third.ok)
                third_events = env.next_user_event(tool_name="get_product", tool_success=True)
                self.assertEqual(third_events[0]["event_id"], "finalize")
                final_event = third_events[0]
            self.assertNotIn("unit", final_event["message"].lower())
            self.assertNotIn("under $", final_event["message"].lower())
            snapshot = env.snapshot()
            self.assertEqual(snapshot["database"]["orders"], {})
            self.assertEqual(snapshot["task_metadata"], task.metadata)

        partial_requirements = memory_tasks["memory_partial"].metadata["memory_requirements"]
        self.assertTrue(any(not requirement["persistent"] and requirement["superseded_by"] == "revise-quantity" for requirement in partial_requirements))


if __name__ == "__main__":
    unittest.main()
