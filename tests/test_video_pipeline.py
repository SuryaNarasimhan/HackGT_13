"""
Unit Tests for Video Perception Pipeline
Tests face detection, facial emotion scoring, keyframe aggregation, and fallback modes.
"""

import unittest
import numpy as np

from app.pipelines.video_pipeline import VideoPipeline
from app.pipelines.speaker_baseline import (
    STATUS_NO_FACE,
    STATUS_UNAVAILABLE,
    STATUS_USED,
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

    def test_skin_colored_non_faces_rejected_when_detector_available(self):
        """When OpenCV's detector finds no face, random noise or a beige wall is not a face."""
        if self.pipeline._cv2 is None or self.pipeline._face_cascade is None:
            self.skipTest("OpenCV face detector unavailable")

        noise = (np.random.default_rng(0).random((720, 1280, 3)) * 255).astype(np.uint8)
        empty_room = np.full((720, 1280, 3), 32, dtype=np.uint8)
        empty_room[100:500, 300:900] = [205, 170, 130]   # Beige wall in the video tile
        empty_room[560:620, 200:1000] = [150, 100, 60]   # Wooden desk

        self.assertEqual(self.pipeline.detect_face(noise), (False, None))
        self.assertEqual(self.pipeline.detect_face(empty_room), (False, None))

    def test_relative_no_face_is_missing(self):
        """No face means the face channel is missing, not a deadpan face, and nothing is learned."""
        baseline = SpeakerBaseline(warmup=2)
        blank_frame = np.zeros((10, 10, 3), dtype=np.uint8)

        _, _, status = self.pipeline.process_keyframes_relative([blank_frame], baseline)

        self.assertEqual(status, STATUS_NO_FACE)
        self.assertEqual(len(baseline), 0)

    def test_relative_without_model_is_unavailable(self):
        """Without the FER model every face reads as the same placeholder, so the face can't count."""
        self.pipeline._load_emotion_classifier = lambda: None
        self.pipeline._emotion_classifier = None
        baseline = SpeakerBaseline(warmup=2)

        _, _, status = self.pipeline.process_keyframes_relative([self._create_synthetic_face_frame()], baseline)

        self.assertEqual(status, STATUS_UNAVAILABLE)
        self.assertEqual(len(baseline), 0)

    def test_relative_learns_usual_expression(self):
        """A face that never smiles becomes 'usual'; a sudden smile counts as different from usual."""
        try:
            import PIL  # noqa: F401  (classify_face_crop hands the crop to the model as a PIL image)
        except ImportError:
            self.skipTest("Pillow unavailable")

        reading = {"label": "neutral"}

        def fake_fer_model(image):
            other = "happy" if reading["label"] == "neutral" else "neutral"
            return [{"label": reading["label"], "score": 0.9}, {"label": other, "score": 0.1}]

        self.pipeline._emotion_classifier = fake_fer_model
        baseline = SpeakerBaseline(warmup=3)
        frame = self._create_synthetic_face_frame()

        statuses = [self.pipeline.process_keyframes_relative([frame], baseline)[2] for _ in range(4)]
        self.assertEqual(statuses, [STATUS_WARMING_UP] * 3 + [STATUS_USUAL])

        reading["label"] = "happy"
        _, p_compare, status = self.pipeline.process_keyframes_relative([frame], baseline)
        self.assertEqual(status, STATUS_USED)
        self.assertEqual(get_top_emotion(p_compare)[0], "joy")


if __name__ == "__main__":
    unittest.main()
