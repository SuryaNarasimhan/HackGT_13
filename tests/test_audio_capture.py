"""
Unit Tests for Audio Loopback Capture & VAD Speech Segmentation
Verifies circular ring buffer retrieval, speech pause detection, and utterance emission.
"""

import unittest
import numpy as np

from app.capture.audio_loopback import AudioLoopbackCapture
from app.capture.vad_detector import VADDetector


class TestAudioCaptureAndVAD(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 16000
        self.loopback = AudioLoopbackCapture(
            target_sample_rate=self.sample_rate,
            buffer_duration=8.0,
            chunk_size=512
        )
        self.vad = VADDetector(
            sample_rate=self.sample_rate,
            silence_threshold_ms=500,
            min_speech_duration_ms=400,
            force_energy_vad=True
        )

    def test_circular_buffer_retrieval(self):
        """Verifies ring buffer write, wrap-around, and retrieving recent seconds."""
        # 1. Feed 2 seconds of a sine wave (frequency 100)
        t2 = np.linspace(0, 2.0, self.sample_rate * 2, endpoint=False)
        audio2 = (0.5 * np.sin(2 * np.pi * 100 * t2)).astype(np.float32)
        self.loopback.feed_simulated_audio(audio2)

        # Retrieve last 1.0 second
        rec1 = self.loopback.get_last_n_seconds(1.0)
        self.assertEqual(len(rec1), self.sample_rate * 1)
        self.assertAlmostEqual(float(np.std(rec1)), float(np.std(audio2)), delta=0.05)

        # 2. Feed 7 more seconds to force wrap-around past the 8.0s capacity
        t7 = np.linspace(0, 7.0, self.sample_rate * 7, endpoint=False)
        audio7 = (0.3 * np.sin(2 * np.pi * 200 * t7)).astype(np.float32)
        self.loopback.feed_simulated_audio(audio7)

        # Retrieve last 3 seconds
        rec3 = self.loopback.get_last_n_seconds(3.0)
        self.assertEqual(len(rec3), self.sample_rate * 3)

    def test_vad_utterance_segmentation_on_silence(self):
        """
        Simulates an utterance:
        - 0.5s initial silence
        - 1.5s speech (tone at 180Hz, RMS > 0.05)
        - 0.6s trailing silence (>= 500ms)
        Must trigger utterance callback with the speech segment.
        """
        emitted_utterances = []

        def on_utterance(segment: np.ndarray):
            emitted_utterances.append(segment)

        self.vad.register_utterance_callback(on_utterance)

        # Generate audio components
        chunk_size = 512
        silence_chunk = np.zeros(chunk_size, dtype=np.float32)

        # 1. Feed 0.5s silence (~15 chunks)
        for _ in range(15):
            self.vad.process_chunk(silence_chunk)
        self.assertEqual(len(emitted_utterances), 0)

        # 2. Feed 1.5s speech (~47 chunks of 180Hz sine wave)
        t_chunk = np.linspace(0, chunk_size / self.sample_rate, chunk_size, endpoint=False)
        speech_chunk = (0.4 * np.sin(2 * np.pi * 180.0 * t_chunk)).astype(np.float32)
        for _ in range(47):
            self.vad.process_chunk(speech_chunk)
        # Should not have emitted yet because speech was active without pause
        self.assertEqual(len(emitted_utterances), 0)
        self.assertTrue(self.vad.is_speech_active)

        # 3. Feed 0.6s trailing silence (~19 chunks) to trigger the >= 500ms pause
        for _ in range(19):
            self.vad.process_chunk(silence_chunk)

        # Must have completed and emitted exactly 1 utterance!
        self.assertEqual(len(emitted_utterances), 1)
        emitted_seg = emitted_utterances[0]
        duration_sec = len(emitted_seg) / self.sample_rate

        print(f"\nVAD Utterance Segmentation Test:")
        print(f"Emitted Utterance Count: {len(emitted_utterances)}")
        print(f"Utterance Audio Duration: {duration_sec:.2f}s (Expected ~2.0s - 2.1s)")

        self.assertGreaterEqual(duration_sec, 1.4)
        self.assertLessEqual(duration_sec, 2.5)


if __name__ == "__main__":
    unittest.main()
