"""
Unit Tests for Per-Speaker Baselines
Verifies warm-up, recognizing a speaker's usual delivery, flagging changes from it, and reset.
"""

import unittest
import numpy as np

from app.pipelines.speaker_baseline import (
    STATUS_USED,
    STATUS_USUAL,
    STATUS_WARMING_UP,
    SpeakerBaseline,
)
from app.pipelines.taxonomy import get_top_emotion, validate_distribution

# Canonical Emotion Order: ['joy', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'neutral']
FLAT = np.array([0.02, 0.02, 0.05, 0.02, 0.02, 0.02, 0.85])
FLAT_WOBBLE = np.array([0.04, 0.02, 0.07, 0.02, 0.02, 0.03, 0.80])
TENSE = np.array([0.03, 0.02, 0.05, 0.50, 0.03, 0.02, 0.35])


class TestSpeakerBaseline(unittest.TestCase):

    def setUp(self):
        self.baseline = SpeakerBaseline(warmup=5, window=30, shift_threshold=0.15)

    def _learn_flat_speaker(self):
        for _ in range(5):
            self.baseline.compare_and_update(FLAT)

    def test_warm_up_never_counts(self):
        """The first readings only teach the baseline; they never count as a change."""
        for _ in range(5):
            status, p_relative = self.baseline.compare_and_update(TENSE)
            self.assertEqual(status, STATUS_WARMING_UP)
            self.assertTrue(np.allclose(p_relative, TENSE))

    def test_consistently_flat_speaker_is_usual(self):
        """A voice that is always flat is this speaker's normal, not a signal."""
        self._learn_flat_speaker()
        status, _ = self.baseline.compare_and_update(FLAT_WOBBLE)
        self.assertEqual(status, STATUS_USUAL)

    def test_change_from_usual_counts_and_shows_what_rose(self):
        """A clear change from usual counts, and the relative reading holds the emotion that rose."""
        self._learn_flat_speaker()
        status, p_relative = self.baseline.compare_and_update(TENSE)

        self.assertEqual(status, STATUS_USED)
        self.assertTrue(validate_distribution(p_relative))
        self.assertEqual(get_top_emotion(p_relative)[0], "anger")

    def test_reset_forgets_speaker(self):
        """Stopping a session forgets the learned style."""
        self._learn_flat_speaker()
        self.baseline.reset()

        self.assertEqual(len(self.baseline), 0)
        status, _ = self.baseline.compare_and_update(TENSE)
        self.assertEqual(status, STATUS_WARMING_UP)


if __name__ == "__main__":
    unittest.main()
