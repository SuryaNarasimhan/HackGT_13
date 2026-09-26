"""
Unit Tests for Telemetry Widget & Incongruence Gauge
Verifies dynamic color mappings, JSD status transitions, and simulated telemetry updates.
"""

import unittest
import numpy as np

from app.ui.components.telemetry_widget import (
    TelemetryWidget,
    get_gauge_status,
    COLOR_GAUGE_SYNC,
    COLOR_GAUGE_NUANCE,
    COLOR_GAUGE_CUE,
)


from app.ui.qt_compat import QT_AVAILABLE, QtWidgets


class TestTelemetryWidget(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if QT_AVAILABLE and QtWidgets is not None:
            cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["--platform", "offscreen"])
        else:
            cls.app = None


    def test_gauge_status_boundaries(self):
        """Verifies gauge text and color transitions across boundary levels."""
        # 1. In Sync (below 0.35)
        status, color = get_gauge_status(0.12)
        self.assertEqual(status, "In Sync")
        self.assertEqual(color, COLOR_GAUGE_SYNC)

        # 2. Subtle Nuance (0.35 to 0.48)
        status, color = get_gauge_status(0.42)
        self.assertEqual(status, "Nuance")
        self.assertEqual(color, COLOR_GAUGE_NUANCE)

        # 3. Cue Detected (above 0.48)
        status, color = get_gauge_status(0.68)
        self.assertEqual(status, "Cue Detected")
        self.assertEqual(color, COLOR_GAUGE_CUE)

    def test_simulated_oscillating_telemetry(self):
        """Simulates 5 oscillating update frames and checks state consistency."""
        widget = TelemetryWidget()

        for step in range(5):
            angle = step * 0.5
            p_video = np.array([abs(np.sin(angle)), 0.05, 0.05, 0.05, 0.05, 0.05, abs(np.cos(angle))])
            p_audio = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.80])
            p_semantic = np.array([0.80, 0.05, 0.05, 0.02, 0.02, 0.02, 0.04])
            jsd_score = 0.20 + 0.35 * abs(np.sin(angle))

            res = widget.update_telemetry(p_video, p_audio, p_semantic, jsd_score)
            self.assertEqual(widget.latest_jsd, jsd_score)
            self.assertIsNotNone(res)

        print("\nTelemetry oscillating simulation test passed successfully!")


if __name__ == "__main__":
    unittest.main()
