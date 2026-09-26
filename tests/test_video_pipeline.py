"""
Unit Tests for Video Perception Pipeline
Tests face detection, facial emotion scoring, keyframe aggregation, and fallback modes.
"""

import unittest
import numpy as np

from app.pipelines.video_pipeline import VideoPipeline
from app.pipelines.taxonomy import (
    validate_distribution,
    get_top_emotion,
    vector_to_dict,
    NUM_EMOTIONS
)


class TestVideoPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = VideoPipeline()

    def _create_synthetic_face_frame(self, height: int = 240, width: int = 320) -> np.ndarray:
        """Creates a synthetic RGB frame simulating a face bounding region."""
        frame = np.full((height, width, 3), 180, dtype=np.uint8)
        # Add oval skin-tone gradient
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        mask = ((x - center_x) ** 2) / (60**2) + ((y - center_y) ** 2) / (80**2) <= 1
        frame[mask] = [210, 175, 140]  # Skin tone
        # Add two dark spots for eyes
        frame[center_y - 20 : center_y - 10, center_x - 30 : center_x - 15] = [30, 30, 30]
        frame[center_y - 20 : center_y - 10, center_x + 15 : center_x + 30] = [30, 30, 30]
        return frame

    def test_single_frame_processing(self):
        """Processes a frame with a face and verifies valid 7-emotion simplex output."""
        frame = self._create_synthetic_face_frame()
        p_video, detected, bbox = self.pipeline.process_frame(frame)

        self.assertEqual(p_video.shape, (NUM_EMOTIONS,))
        self.assertTrue(validate_distribution(p_video))
        self.assertTrue(detected)
        self.assertIsNotNone(bbox)

        top_emotion, score = get_top_emotion(p_video)
        print(f"\nSynthetic Face Frame:")
        print(f"Detected: {detected}, BBox: {bbox}")
        print(f"Top Emotion: {top_emotion} ({score:.2%})")
        print(f"Distribution: {vector_to_dict(p_video)}")

    def test_keyframe_aggregation(self):
        """Processes a sequence of 4 keyframes and verifies averaged distribution."""
        frame1 = self._create_synthetic_face_frame()
        frame2 = self._create_synthetic_face_frame()
        frame3 = self._create_synthetic_face_frame()
        frame4 = self._create_synthetic_face_frame()

        keyframes = [frame1, frame2, frame3, frame4]
        agg_vec, detected = self.pipeline.process_keyframes(keyframes)

        self.assertEqual(agg_vec.shape, (NUM_EMOTIONS,))
        self.assertTrue(validate_distribution(agg_vec))
        self.assertTrue(detected)

    def test_empty_or_black_frame_fallback(self):
        """Empty or blank frames must trigger graceful neutral fallback with detected=False."""
        blank_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        p_video, detected, bbox = self.pipeline.process_frame(blank_frame)

        self.assertFalse(detected)
        self.assertIsNone(bbox)
        self.assertTrue(validate_distribution(p_video))

        top_emotion, score = get_top_emotion(p_video)
        self.assertEqual(top_emotion, "neutral")
        self.assertGreater(score, 0.70)


    def test_non_face_desktop_rejection(self):
        """High-variance desktop textures/text without faces must return detected=False."""
        # Simulated IDE code/text (alternating dark background and bright characters, high variance)
        desktop_frame = np.full((300, 400, 3), 30, dtype=np.uint8)
        # Add high-contrast white/green code syntax lines
        desktop_frame[20:25, 30:200] = [200, 200, 200]
        desktop_frame[40:45, 50:180] = [80, 220, 100]
        desktop_frame[60:65, 30:350] = [100, 150, 255]

        detected, bbox = self.pipeline.detect_face(desktop_frame)
        self.assertFalse(detected)
        self.assertIsNone(bbox)

        p_video, proc_detected, _ = self.pipeline.process_frame(desktop_frame)
        self.assertFalse(proc_detected)
        top_emotion, _ = get_top_emotion(p_video)
        self.assertEqual(top_emotion, "neutral")

    def test_rgb_bgr_color_handling(self):
        """Verifies classify_face_crop handles RGB and BGR without crashing or channel confusion."""
        face_crop = np.full((100, 100, 3), [210, 175, 140], dtype=np.uint8)
        # RGB crop
        dist_rgb = self.pipeline.classify_face_crop(face_crop, color_order="RGB")
        self.assertEqual(dist_rgb.shape, (NUM_EMOTIONS,))
        self.assertTrue(validate_distribution(dist_rgb))

        # BGR crop
        dist_bgr = self.pipeline.classify_face_crop(face_crop, color_order="BGR")
        self.assertEqual(dist_bgr.shape, (NUM_EMOTIONS,))
        self.assertTrue(validate_distribution(dist_bgr))


if __name__ == "__main__":
    unittest.main()
