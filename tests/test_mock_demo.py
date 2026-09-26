"""
Unit Tests for Mock Demo Simulation Mode
Verifies deterministic execution of Scenarios 1 (Sarcasm), 2 (Praise), and 3 (Frustration).
"""

import unittest
from app.mock_demo import MockDemoRunner, DEMO_SCENARIOS


class TestMockDemo(unittest.TestCase):

    def setUp(self):
        self.telemetry_history = []
        self.cue_history = []

        def on_telemetry(p_v, p_a, p_s, jsd):
            self.telemetry_history.append((p_v, p_a, p_s, jsd))

        def on_cue(cue_data):
            self.cue_history.append(cue_data)

        self.runner = MockDemoRunner(
            telemetry_callback=on_telemetry,
            cue_callback=on_cue,
            threshold=0.40
        )

    def test_scenario_1_deadpan_sarcasm(self):
        """Scenario 1: High divergence -> Sarcasm Cue Triggered."""
        res = self.runner.trigger_scenario(1)

        self.assertTrue(res["is_trigger"])
        self.assertGreater(res["jsd_score"], 0.40)
        self.assertIsNotNone(res["cue_data"])
        self.assertEqual(res["cue_data"]["social_cue_type"], "Dry Sarcasm / Irony")
        self.assertGreater(len(self.cue_history), 0)

        print("\n[Scenario 1: Sarcasm] Result:")
        print(f"JSD: {res['jsd_score']:.3f} | Cue: {res['cue_data']['social_cue_type']}")
        print(f"Tip: {res['cue_data']['suggested_action']}")

    def test_scenario_2_sincere_praise(self):
        """Scenario 2: Low divergence -> Congruent In-Sync Cue Card."""
        res = self.runner.trigger_scenario(2)

        self.assertFalse(res["is_trigger"])
        self.assertLess(res["jsd_score"], 0.15)
        self.assertIsNotNone(res["cue_data"])
        self.assertEqual(res["cue_data"]["social_cue_type"], "In Sync / Authentic")
        self.assertGreater(len(self.cue_history), 0)

        print("\n[Scenario 2: Sincere Praise] Result:")
        print(f"JSD: {res['jsd_score']:.3f} | Trigger: {res['is_trigger']} | Cue: {res['cue_data']['social_cue_type']}")

    def test_scenario_3_concealed_frustration(self):
        """Scenario 3: Polite words with vocal/facial tension -> Frustration Cue."""
        res = self.runner.trigger_scenario(3)

        self.assertTrue(res["is_trigger"])
        self.assertGreater(res["jsd_score"], 0.40)
        self.assertIsNotNone(res["cue_data"])
        self.assertEqual(res["cue_data"]["social_cue_type"], "Concealed Frustration")

        print("\n[Scenario 3: Concealed Frustration] Result:")
        print(f"JSD: {res['jsd_score']:.3f} | Cue: {res['cue_data']['social_cue_type']}")
        print(f"Tip: {res['cue_data']['suggested_action']}")


if __name__ == "__main__":
    unittest.main()
