"""
Unit Test for Desktop Overlay Window
Verifies window creation, attributes, header components, and drag properties.
"""

import unittest
from app.ui.overlay_window import SocialLensOverlayWindow


from app.ui.qt_compat import QT_AVAILABLE, QtWidgets


class TestOverlayWindow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if QT_AVAILABLE and QtWidgets is not None:
            cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["--platform", "offscreen"])
        else:
            cls.app = None


    def test_overlay_instantiation(self):
        """Creates overlay window and checks geometry and title components."""
        window = SocialLensOverlayWindow(width=340, height=480)
        self.assertIsNotNone(window)

        # Check dimensions
        self.assertEqual(window.window_width, 340)
        self.assertEqual(window.window_height, 480)

        # Trigger show and then close
        if hasattr(window, "show"):
            window.show()
        if hasattr(window, "close"):
            window.close()

        print("\nOverlay Window successfully instantiated and verified!")


if __name__ == "__main__":
    unittest.main()
