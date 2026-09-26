"""
Unit Tests for SocialLens Pipeline Coordinator
Verifies end-to-end utterance processing across Video, Audio, Semantics, JSD, and Reasoner.
"""

import unittest
import numpy as np

from app.capture.audio_loopback import AudioLoopbackCapture
from app.capture.screen_capture import ScreenCaptureManager
from app.capture.vad_detector import VADDetector
from app.coordinator import PipelineCoordinator


class TestCoordinator(unittest.TestCase):

    def setUp(self):
        self.audio = AudioLoopbackCapture()
        self.screen = ScreenCaptureManager()
        self.vad = VADDetector()

        self.telemetry_history = []
        self.cue_history = []
        self.transcript_history = []

        def on_telemetry(p_v, p_a, p_s, jsd):
            self.telemetry_history.append((p_v, p_a, p_s, jsd))

        def on_cue(cue_data):
            self.cue_history.append(cue_data)

        def on_transcript(text):
            self.transcript_history.append(text)

        self.coordinator = PipelineCoordinator(
            audio_capture=self.audio,
            screen_capture=self.screen,
            vad_detector=self.vad,
            telemetry_callback=on_telemetry,
            cue_callback=on_cue,
            transcript_callback=on_transcript,
            jsd_threshold=0.40
        )

    def tearDown(self):
        self.coordinator.stop()

    def test_coordinator_process_utterance(self):
        """Processes a simulated utterance and checks all channel outputs and JSD scores."""
        # 2 seconds of 16kHz speech audio
        t = np.linspace(0, 2.0, 32000, endpoint=False)
        audio = (0.4 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)

        # Seed screen buffer with a frame
        frame = np.full((240, 320, 3), 150, dtype=np.uint8)
        self.screen.push_frame(frame)

        result = self.coordinator.process_utterance(audio)

        self.assertIn("transcript", result)
        self.assertIn("p_video", result)
        self.assertIn("p_audio", result)
        self.assertIn("p_semantic", result)
        self.assertIn("jsd_score", result)
        self.assertIn("is_trigger", result)

        self.assertGreaterEqual(result["jsd_score"], 0.0)
        self.assertLessEqual(result["jsd_score"], 1.0)
        self.assertGreater(len(self.telemetry_history), 0)
        self.assertGreater(len(self.transcript_history), 0)
        self.assertEqual(self.transcript_history[-1], result["transcript"])

        print("\n--- COORDINATOR UTTERANCE PROCESSED ---")
        print(f"Transcript: {result['transcript']}")
        print(f"JSD Incongruence Score: {result['jsd_score']:.3f}")
        print(f"Is Trigger: {result['is_trigger']}")
        if result["cue_data"]:
            print(f"Cue Type: {result['cue_data'].get('social_cue_type')}")
            print(f"Tip: {result['cue_data'].get('suggested_action')}")

    def test_lifecycle_start_stop(self):
        """Verifies start and stop lifecycle execution without thread hangs."""
        self.coordinator.start()
        self.assertTrue(self.coordinator._running)
        self.coordinator.stop()
        self.assertFalse(self.coordinator._running)


if __name__ == "__main__":
    unittest.main()
