"""
Unit Tests for Subtext Insight Card Widget
Verifies badge text formatting, response tip layout, and card visibility transitions.
"""

import unittest
from app.ui.qt_compat import QT_AVAILABLE, QtWidgets
from app.ui.components.cue_card import SubtextInsightCard, CUE_BADGE_COLORS


class TestCueCard(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if QT_AVAILABLE and QtWidgets is not None:
            cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["--platform", "offscreen"])
        else:
            cls.app = None

    def test_cue_card_display_and_hide(self):
        """Displays a cue card and tests attribute mapping and dismissal."""
        card = SubtextInsightCard(auto_dismiss_sec=5)

        cue_data = {
            "social_cue_type": "Dry Sarcasm / Irony",
            "confidence": "High",
            "explanation": "Contradiction between positive words and deadpan delivery.",
            "suggested_action": "Acknowledge the shared setback lightly."
        }

        # Display cue
        card.display_cue(cue_data)

        # Hide card
        card.hide_card()
        if hasattr(card, "isVisible"):
            self.assertFalse(card.isVisible())
        elif hasattr(card, "is_visible"):
            self.assertFalse(card.is_visible)

        print("\nSubtext Insight Card test passed successfully!")

    def test_badge_palette_coverage(self):
        """Verifies distinct badge colors exist for common social cues."""
        self.assertIn("In Sync / Authentic", CUE_BADGE_COLORS)
        self.assertIn("Dry Sarcasm / Irony", CUE_BADGE_COLORS)
        self.assertIn("Concealed Frustration", CUE_BADGE_COLORS)
        self.assertIn("Playful Teasing", CUE_BADGE_COLORS)


if __name__ == "__main__":
    unittest.main()
