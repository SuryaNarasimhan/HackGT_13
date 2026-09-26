"""
Unit Tests for Semantic Perception Pipeline
Verifies text emotion classification and projection to 7-emotion simplex.
"""

import unittest
import numpy as np

from app.pipelines.semantic_pipeline import SemanticPipeline
from app.pipelines.taxonomy import (
    validate_distribution,
    get_top_emotion,
    vector_to_dict,
    NUM_EMOTIONS
)


class TestSemanticPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = SemanticPipeline(lazy_load=True)

    def test_positive_praise_emotion(self):
        """Phrase with praise keywords must project to high joy/surprise."""
        text = "Oh wow, that's just fantastic news."
        vec = self.pipeline.classify_text_emotion(text)

        # 1. Must be shape (7,)
        self.assertEqual(vec.shape, (NUM_EMOTIONS,))
        # 2. Must be a valid probability distribution
        self.assertTrue(validate_distribution(vec))

        top_emotion, score = get_top_emotion(vec)
        print(f"\nText: '{text}'")
        print(f"Top Emotion: {top_emotion} ({score:.2%})")
        print(f"Full Distribution: {vector_to_dict(vec)}")

        self.assertIn(top_emotion, ["joy", "surprise"])
        self.assertGreater(score, 0.30)

    def test_frustration_emotion(self):
        """Phrase with frustration keywords must project to high anger/sadness."""
        text = "This code is completely broken and crashed again."
        vec = self.pipeline.classify_text_emotion(text)

        self.assertTrue(validate_distribution(vec))
        top_emotion, score = get_top_emotion(vec)
        self.assertIn(top_emotion, ["anger", "sadness"])

    def test_empty_text_returns_neutral(self):
        """Empty or whitespace-only text must return 100% neutral."""
        vec = self.pipeline.classify_text_emotion("   ")
        self.assertTrue(validate_distribution(vec))
        top_emotion, score = get_top_emotion(vec)
        self.assertEqual(top_emotion, "neutral")
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
