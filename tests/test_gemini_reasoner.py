"""
Test script for Gemini Cognitive Reasoner
Simulates deadpan sarcasm and verifies structured JSON response parsing.
"""

import json
import unittest
import numpy as np

from app.agent.gemini_reasoner import GeminiReasoner
from app.pipelines.taxonomy import CANONICAL_EMOTIONS, normalize_distribution


class TestGeminiReasoner(unittest.TestCase):

    def setUp(self):
        self.reasoner = GeminiReasoner()

    def test_sarcasm_scenario_synthesis(self):
        """Simulates deadpan sarcasm: positive words + monotone voice + deadpan face."""
        transcript = "Oh fantastic, my entire backend server just crashed right before the demo."
        
        # Words: High Joy / Positive
        p_semantic = np.array([0.85, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02])
        # Voice Tone: Monotone / Neutral
        p_audio = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.88])
        # Face: Neutral / Deadpan
        p_video = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.88])
        
        jsd_score = 0.446
        conflict_pair = ("words", "tone")
        max_conflict_val = 0.774

        result = self.reasoner.synthesize_cue(
            transcript=transcript,
            p_video=p_video,
            p_audio=p_audio,
            p_semantic=p_semantic,
            jsd_score=jsd_score,
            conflict_pair=conflict_pair,
            max_conflict_value=max_conflict_val
        )

        print("\n" + "=" * 60)
        print("GEMINI REASONER STRUCTURED OUTPUT:")
        print(json.dumps(result, indent=2))
        print("=" * 60 + "\n")

        # Validate schema compliance
        self.assertIn("social_cue_type", result)
        self.assertIn("confidence", result)
        self.assertIn("explanation", result)
        self.assertIn("suggested_action", result)

        self.assertIn(result["confidence"], ["High", "Medium", "Low"])
        self.assertTrue(len(result["explanation"]) > 10)
        self.assertTrue(len(result["suggested_action"]) > 10)
        # Should identify sarcasm / irony
        self.assertEqual(result["social_cue_type"], "Dry Sarcasm / Irony")

    def test_concealed_frustration_scenario(self):
        """Simulates words saying 'I'm fine' with tense vocal tone and furrowed brow."""
        transcript = "No, it's completely fine, really."

        # Words: Neutral
        p_semantic = np.array([0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.65])
        # Tone: Anger/Tense
        p_audio = np.array([0.02, 0.02, 0.02, 0.82, 0.05, 0.05, 0.02])
        # Face: Disgust / Frustration
        p_video = np.array([0.02, 0.02, 0.02, 0.02, 0.85, 0.05, 0.02])

        result = self.reasoner.synthesize_cue(
            transcript=transcript,
            p_video=p_video,
            p_audio=p_audio,
            p_semantic=p_semantic,
            jsd_score=0.62,
            conflict_pair=("words", "tone"),
            max_conflict_value=0.81
        )

        self.assertEqual(result["social_cue_type"], "Concealed Frustration")
        self.assertTrue(any(w in result["explanation"].lower() for w in ["frustrat", "upset", "tension", "stress"]))

    def test_in_sync_authentic_scenario(self):
        """Simulates congruent communication where words, tone, and face align in harmony."""
        transcript = "Thank you so much for helping me out today!"

        p_semantic = np.array([0.88, 0.04, 0.02, 0.02, 0.01, 0.01, 0.02])  # Joy
        p_audio = np.array([0.82, 0.05, 0.03, 0.03, 0.02, 0.02, 0.03])     # Joy
        p_video = np.array([0.91, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01])     # Joy / Smile

        result = self.reasoner.synthesize_cue(
            transcript=transcript,
            p_video=p_video,
            p_audio=p_audio,
            p_semantic=p_semantic,
            jsd_score=0.08,
            conflict_pair=("words", "tone"),
            max_conflict_value=0.06,
            is_trigger=False
        )

        self.assertEqual(result["social_cue_type"], "In Sync / Authentic")
        self.assertEqual(result["confidence"], "High")
        self.assertTrue(any(w in result["explanation"].lower() for w in ["harmony", "align", "genuine", "authentic", "sync"]))
        self.assertEqual(result["transcript"], transcript)


if __name__ == "__main__":
    unittest.main()
