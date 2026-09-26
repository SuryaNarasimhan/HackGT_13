"""
Unit Tests for Jensen-Shannon Divergence Engine
Compatible with both Python's built-in unittest and pytest.
"""

import unittest
import numpy as np
from app.math_engine.jsd import (
    shannon_entropy,
    compute_pairwise_jsd,
    compute_tri_modal_jsd,
    analyze_cross_modal_conflict,
)
from app.pipelines.taxonomy import uniform_distribution, NUM_EMOTIONS


class TestJSD(unittest.TestCase):

    def test_entropy_properties(self):
        """Verify entropy is 0 for deterministic distribution and max for uniform."""
        deterministic = np.zeros(NUM_EMOTIONS)
        deterministic[0] = 1.0
        self.assertTrue(np.isclose(shannon_entropy(deterministic), 0.0, atol=1e-5))

        uniform = uniform_distribution()
        expected_max = np.log2(NUM_EMOTIONS)
        self.assertTrue(np.isclose(shannon_entropy(uniform), expected_max, atol=1e-5))

    def test_jsd_identical_distributions(self):
        """Identical distributions must yield JSD = 0.0."""
        p1 = np.array([0.7, 0.1, 0.1, 0.05, 0.05, 0.0, 0.0])
        p2 = np.array([0.7, 0.1, 0.1, 0.05, 0.05, 0.0, 0.0])
        p3 = np.array([0.7, 0.1, 0.1, 0.05, 0.05, 0.0, 0.0])

        self.assertTrue(np.isclose(compute_pairwise_jsd(p1, p2), 0.0, atol=1e-5))
        self.assertTrue(np.isclose(compute_tri_modal_jsd(p1, p2, p3, normalized=True), 0.0, atol=1e-5))
        self.assertTrue(np.isclose(compute_tri_modal_jsd(p1, p2, p3, normalized=False), 0.0, atol=1e-5))

    def test_jsd_orthogonal_distributions(self):
        """Completely disjoint/orthogonal distributions must reach theoretical maximum (1.0 normalized)."""
        p_v = np.zeros(NUM_EMOTIONS)
        p_v[0] = 1.0  # Joy

        p_a = np.zeros(NUM_EMOTIONS)
        p_a[1] = 1.0  # Surprise

        p_s = np.zeros(NUM_EMOTIONS)
        p_s[2] = 1.0  # Sadness

        self.assertTrue(np.isclose(compute_pairwise_jsd(p_v, p_a), 1.0, atol=1e-5))

        jsd_raw = compute_tri_modal_jsd(p_v, p_a, p_s, normalized=False)
        self.assertTrue(np.isclose(jsd_raw, np.log2(3.0), atol=1e-5))

        jsd_norm = compute_tri_modal_jsd(p_v, p_a, p_s, normalized=True)
        self.assertTrue(np.isclose(jsd_norm, 1.0, atol=1e-5))

    def test_jsd_symmetry(self):
        """Permuting the distributions must not change the JSD result."""
        p_v = np.array([0.8, 0.1, 0.05, 0.05, 0.0, 0.0, 0.0])
        p_a = np.array([0.1, 0.1, 0.6, 0.1, 0.05, 0.05, 0.0])
        p_s = np.array([0.05, 0.05, 0.1, 0.7, 0.05, 0.05, 0.0])

        jsd_1 = compute_tri_modal_jsd(p_v, p_a, p_s)
        jsd_2 = compute_tri_modal_jsd(p_s, p_v, p_a)
        jsd_3 = compute_tri_modal_jsd(p_a, p_s, p_v)

        self.assertTrue(np.isclose(jsd_1, jsd_2, atol=1e-6))
        self.assertTrue(np.isclose(jsd_2, jsd_3, atol=1e-6))

    def test_edge_cases_zeros_and_uniform(self):
        """Test robustness against zero probabilities, single elements, and uniform distributions."""
        zeros = np.zeros(NUM_EMOTIONS)
        uniform = uniform_distribution()
        peaked = np.array([1.0, 0, 0, 0, 0, 0, 0])

        jsd_val = compute_tri_modal_jsd(zeros, uniform, peaked)
        self.assertFalse(np.isnan(jsd_val))
        self.assertTrue(0.0 <= jsd_val <= 1.0)

    def test_sarcasm_scenario_trigger(self):
        """
        Test real-world sarcasm scenario:
        Words express joy ('Yeah, brilliant'), but tone and face are neutral/deadpan.
        Should trigger high divergence and flag words vs tone/face.
        """
        p_words = np.array([0.90, 0.02, 0.02, 0.02, 0.01, 0.01, 0.02])
        p_tone = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.88])
        p_face = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.88])

        analysis = analyze_cross_modal_conflict(p_face, p_tone, p_words)

        self.assertTrue(analysis["is_trigger"])
        self.assertGreater(analysis["tri_modal_jsd"], 0.40)
        self.assertIn("words", analysis["max_conflict_pair"])

    def test_sincere_scenario_no_trigger(self):
        """
        Test real-world sincere scenario:
        All 3 modalities agree on joy.
        Should result in very low divergence (well below threshold).
        """
        p_words = np.array([0.88, 0.04, 0.02, 0.02, 0.01, 0.01, 0.02])
        p_tone = np.array([0.82, 0.05, 0.03, 0.03, 0.02, 0.02, 0.03])
        p_face = np.array([0.91, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01])

        analysis = analyze_cross_modal_conflict(p_face, p_tone, p_words, threshold=0.45)

        self.assertFalse(analysis["is_trigger"])
        self.assertLess(analysis["tri_modal_jsd"], 0.15)


if __name__ == "__main__":
    unittest.main()
