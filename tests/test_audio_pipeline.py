"""
Unit Tests for Audio Prosody Pipeline
Tests flat monotone (deadpan) vs. expressive pitch waveforms and canonical simplex mapping.
"""

import unittest
import numpy as np

from app.pipelines.audio_pipeline import AudioPipeline
from app.pipelines.taxonomy import (
    validate_distribution,
    get_top_emotion,
    vector_to_dict,
    NUM_EMOTIONS
)


class TestAudioPipeline(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 16000
        self.pipeline = AudioPipeline(sample_rate=self.sample_rate, lazy_load=True)

    def _generate_monotone_audio(self, duration: float = 2.0, freq: float = 140.0) -> np.ndarray:
        """Generates a steady, flat monotone harmonic tone (deadpan voice simulation)."""
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        # Fundamental + slight harmonics with static envelope
        waveform = 0.6 * np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * 2 * freq * t)
        return waveform.astype(np.float32)

    def _generate_expressive_audio(self, duration: float = 2.0) -> np.ndarray:
        """Generates a pitch-varying, dynamic waveform (excited/expressive speech simulation)."""
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        # Dynamic frequency modulation sweeping from 160 Hz to 340 Hz
        inst_freq = 240.0 + 80.0 * np.sin(2 * np.pi * 3.0 * t)
        phase = 2 * np.pi * np.cumsum(inst_freq) / self.sample_rate
        # Amplitude modulation (vocal energy inflection)
        envelope = 0.5 + 0.4 * np.sin(2 * np.pi * 2.5 * t)
        waveform = envelope * np.sin(phase)
        return waveform.astype(np.float32)

    def test_monotone_deadpan_detection(self):
        """Monotone audio with flat pitch must produce high neutral probability."""
        audio = self._generate_monotone_audio(duration=2.0, freq=135.0)
        p_audio, features = self.pipeline.process(audio)

        print("\n--- MONOTONE AUDIO TEST ---")
        print(f"Features: pitch_std={features['pitch_std']:.2f}, is_monotone={features['is_monotone']}")
        print(f"Distribution: {vector_to_dict(p_audio)}")

        self.assertEqual(p_audio.shape, (NUM_EMOTIONS,))
        self.assertTrue(validate_distribution(p_audio))
        self.assertTrue(features["is_monotone"])

        top_emotion, score = get_top_emotion(p_audio)
        self.assertEqual(top_emotion, "neutral")
        self.assertGreater(score, 0.60)

    def test_expressive_audio_detection(self):
        """Expressive audio with dynamic pitch inflection must boost joy/surprise."""
        audio = self._generate_expressive_audio(duration=2.0)
        p_audio, features = self.pipeline.process(audio)

        print("\n--- EXPRESSIVE AUDIO TEST ---")
        print(f"Features: pitch_std={features['pitch_std']:.2f}, is_monotone={features['is_monotone']}")
        print(f"Distribution: {vector_to_dict(p_audio)}")

        self.assertEqual(p_audio.shape, (NUM_EMOTIONS,))
        self.assertTrue(validate_distribution(p_audio))
        self.assertFalse(features["is_monotone"])
        self.assertGreater(features["pitch_std"], 25.0)

        top_emotion, score = get_top_emotion(p_audio)
        self.assertIn(top_emotion, ["sadness", "joy", "surprise"])

    def test_empty_audio_returns_neutral(self):
        """Empty audio buffer returns canonical neutral distribution."""
        empty_audio = np.array([], dtype=np.float32)
        p_audio, _ = self.pipeline.process(empty_audio)
        self.assertTrue(validate_distribution(p_audio))
        top_emotion, score = get_top_emotion(p_audio)
        self.assertEqual(top_emotion, "neutral")


if __name__ == "__main__":
    unittest.main()
