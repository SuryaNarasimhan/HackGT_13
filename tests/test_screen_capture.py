"""
Unit Tests for Screen Capture Manager
Tests ROI setting, timestamped keyframe extraction, and screenshot saving.
"""

import os
import time
import unittest
import numpy as np

from app.capture.screen_capture import ScreenCaptureManager


class TestScreenCapture(unittest.TestCase):

    def setUp(self):
        self.output_dir = "tests/output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.manager = ScreenCaptureManager(
            fps=20,
            buffer_seconds=4.0,
            roi={"top": 50, "left": 50, "width": 320, "height": 240}
        )

    def tearDown(self):
        self.manager.stop()

    def test_roi_configuration(self):
        """Verifies setting and getting ROI bounding box."""
        self.manager.set_roi(top=120, left=240, width=400, height=300)
        roi = self.manager.get_roi()
        self.assertEqual(roi["top"], 120)
        self.assertEqual(roi["left"], 240)
        self.assertEqual(roi["width"], 400)
        self.assertEqual(roi["height"], 300)

    def test_timestamped_keyframe_extraction(self):
        """Pushes timestamped frames and extracts keyframes in a specific time window."""
        t_base = time.time()
        for i in range(10):
            frame = np.full((240, 320, 3), i * 20, dtype=np.uint8)
            # Push with artificial timestamps spaced 0.1s apart
            self.manager.push_frame(frame, timestamp=t_base + i * 0.1)

        # Request keyframes between t_base + 0.2 and t_base + 0.8
        keyframes = self.manager.get_keyframes_in_window(
            start_time=t_base + 0.2,
            end_time=t_base + 0.8,
            sample_count=4
        )

        self.assertEqual(len(keyframes), 4)
        for kf in keyframes:
            self.assertEqual(kf.shape, (240, 320, 3))

    def test_save_screenshot(self):
        """Captures or generates a frame and saves it to disk."""
        test_frame = np.full((120, 160, 3), 120, dtype=np.uint8)
        # Add visual feature
        test_frame[40:80, 50:110] = [255, 100, 50]
        
        screenshot_path = os.path.join(self.output_dir, "test_screenshot.ppm")
        saved = self.manager.save_screenshot(screenshot_path, frame=test_frame)

        self.assertTrue(saved)
        # Check either .ppm or specified path
        actual_path = screenshot_path if os.path.exists(screenshot_path) else screenshot_path + ".ppm"
        self.assertTrue(os.path.exists(actual_path))
        self.assertGreater(os.path.getsize(actual_path), 100)
        print(f"\nSaved test screenshot to: {actual_path} ({os.path.getsize(actual_path)} bytes)")


if __name__ == "__main__":
    unittest.main()
