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

    # Canonical Emotion Order: ['joy', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'neutral']
    JOY = np.array([0.88, 0.04, 0.02, 0.02, 0.01, 0.01, 0.02])
    FLAT = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.88])

    def test_flatter_voice_alone_is_not_sarcasm(self):
        """Sincere thanks with a flatter-than-usual voice and nothing ironic in the words stays ambiguous."""
        result = self.reasoner.synthesize_cue(
            transcript="Thanks, I really appreciate your help.",
            p_video=self.FLAT,
            p_audio=self.FLAT,
            p_semantic=self.JOY,
            jsd_score=0.77,
            conflict_pair=("words", "tone"),
            max_conflict_value=0.77,
            is_trigger=True,
            dialogue_history='- You (User): "I sent over the notes."',
            channel_status={"words": "used", "tone": "used", "face": "no_face"}
        )

        self.assertEqual(result["social_cue_type"], "Ambiguous")
        self.assertEqual(result["confidence"], "Low")
        self.assertNotIn("None", result["explanation"])

    def test_conversation_setback_supports_sarcasm(self):
        """The same delivery reads as sarcasm when the conversation mentions a setback."""
        result = self.reasoner.synthesize_cue(
            transcript="Great, love that.",
            p_video=self.FLAT,
            p_audio=self.FLAT,
            p_semantic=self.JOY,
            jsd_score=0.77,
            conflict_pair=("words", "tone"),
            max_conflict_value=0.77,
            is_trigger=True,
            dialogue_history='- Other Person: "The build failed again right before the deadline."',
            channel_status={"words": "used", "tone": "used", "face": "no_face"}
        )

        self.assertEqual(result["social_cue_type"], "Dry Sarcasm / Irony")
        self.assertIn("setback", result["explanation"])

    def test_face_value_explains_usual_delivery(self):
        """When voice and face match this speaker's usual, the card says so instead of claiming harmony."""
        result = self.reasoner.synthesize_cue(
            transcript="Thanks. This is really great work.",
            p_video=self.FLAT,
            p_audio=self.FLAT,
            p_semantic=self.JOY,
            jsd_score=0.0,
            is_trigger=False,
            channel_status={"words": "used", "tone": "usual", "face": "usual"}
        )

        self.assertEqual(result["social_cue_type"], "In Sync / Authentic")
        self.assertIn("usually come across", result["explanation"])
        self.assertNotIn("harmony", result["explanation"])

    def test_prompt_marks_unused_channels_and_does_not_presume_subtext(self):
        """Channels that didn't count are named, and a mismatch alone is not treated as proof of subtext."""
        prompt = self.reasoner._build_prompt(
            transcript="Great, love that.",
            p_video=self.FLAT,
            p_audio=self.FLAT,
            p_semantic=self.JOY,
            jsd_score=0.77,
            conflict_pair=("words", "tone"),
            max_conflict_value=0.77,
            is_trigger=True,
            channel_status={"words": "used", "tone": "used", "face": "usual"}
        )

        self.assertIn("Facial Expression: not used (usual for them)", prompt)
        self.assertIn("Vocal Tone / Prosody (what is stronger than their usual)", prompt)
        self.assertIn("never enough on their own", prompt)
        self.assertNotIn("indicates sarcasm", prompt)

    def test_no_words_is_not_called_authentic(self):
        """With nothing transcribed, the card says so instead of vouching for the speaker."""
        result = self.reasoner.synthesize_cue(
            transcript="[Speech detected without clear transcript]",
            p_video=self.FLAT,
            p_audio=self.FLAT,
            p_semantic=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            jsd_score=0.0,
            is_trigger=False,
            channel_status={"words": "no_words", "tone": "used", "face": "unavailable"}
        )

        self.assertEqual(result["social_cue_type"], "Ambiguous")
        self.assertEqual(result["confidence"], "Low")
        self.assertIn("Couldn't make out the words", result["explanation"])

    def test_unknown_category_becomes_ambiguous(self):
        """Categories outside the schema are not shown to the user."""
        result = self.reasoner._validate_response(
            {"social_cue_type": "Lying", "confidence": "High", "explanation": "x", "suggested_action": "y"},
            "Hi"
        )
        self.assertEqual(result["social_cue_type"], "Ambiguous")


if __name__ == "__main__":
    unittest.main()
