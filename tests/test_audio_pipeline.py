"""
Unit Tests for Audio Prosody Pipeline
Tests flat monotone (deadpan) vs. expressive pitch waveforms and canonical simplex mapping.
"""

import unittest
import numpy as np

from app.pipelines.audio_pipeline import AudioPipeline
from app.pipelines.speaker_baseline import (
    STATUS_TOO_QUIET,
    STATUS_USUAL,
    STATUS_WARMING_UP,
    SpeakerBaseline,
)
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

    def _generate_melody(self, base_freq: float, depth_semitones: float, duration: float = 2.0) -> np.ndarray:
        """Generates the same intonation contour (in semitones) around any base pitch."""
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        inst_freq = base_freq * 2 ** (depth_semitones * np.sin(2 * np.pi * 1.5 * t) / 12.0)
        phase = 2 * np.pi * np.cumsum(inst_freq) / self.sample_rate
        waveform = 0.6 * np.sin(phase) + 0.2 * np.sin(2 * phase)
        return waveform.astype(np.float32)

    def test_pitch_spread_same_for_low_and_high_voices(self):
        """
        The same melody gets the same pitch spread and monotone verdict in a low and a high voice.
        The old 18 Hz cutoff called the 100 Hz voice monotone and the 200 Hz voice expressive.
        """
        low = self.pipeline.extract_prosody_features(self._generate_melody(100.0, 3.0))
        high = self.pipeline.extract_prosody_features(self._generate_melody(200.0, 3.0))

        self.assertAlmostEqual(low["pitch_spread_st"], high["pitch_spread_st"], delta=0.5)
        self.assertFalse(low["is_monotone"])
        self.assertFalse(high["is_monotone"])
        self.assertLess(low["pitch_std"], 18.0)  # Would have been "monotone" under the Hz rule

    def test_relative_too_little_voice(self):
        """Silence is too little voice to judge, and teaches the baseline nothing."""
        baseline = SpeakerBaseline(warmup=3)
        p_display, p_compare, features = self.pipeline.process_relative(
            np.zeros(self.sample_rate, dtype=np.float32), baseline
        )

        self.assertEqual(features["status"], STATUS_TOO_QUIET)
        self.assertTrue(np.allclose(p_display, p_compare))
        self.assertEqual(len(baseline), 0)

    def test_relative_consistently_flat_voice_becomes_usual(self):
        """A voice that is always monotone is learned as this speaker's usual, not flagged."""
        baseline = SpeakerBaseline(warmup=3)
        audio = self._generate_monotone_audio(duration=2.0, freq=135.0)

        statuses = [self.pipeline.process_relative(audio, baseline)[2]["status"] for _ in range(5)]

        self.assertEqual(statuses, [STATUS_WARMING_UP] * 3 + [STATUS_USUAL] * 2)

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
