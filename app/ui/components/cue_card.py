"""
Subtext Insight Card Widget for SocialLens HUD
Displays animated social cue interpretations, explanations, and assistive response tips.
Includes auto-dismiss timer (10s), dismiss button, and manual preview trigger.
"""

import logging
from typing import Dict, Optional
from app.config import UI_CARD_AUTO_DISMISS_SECONDS
from app.ui.qt_compat import QT_AVAILABLE, QtCore, QtWidgets, QtGui
from app.ui.styles import (
    COLOR_BG_CARD,
    COLOR_BORDER_GLOW,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    COLOR_TEXT_MUTED,
)

logger = logging.getLogger(__name__)

CUE_BADGE_COLORS: Dict[str, str] = {
    "In Sync / Authentic": "#10B981",      # Emerald Green
    "Sincere / In Sync": "#10B981",        # Emerald Green
    "Dry Sarcasm / Irony": "#C084FC",      # Purple
    "Playful Teasing": "#38BDF8",          # Cyan
    "Concealed Frustration": "#F87171",    # Soft Red
    "Polite Agreement": "#34D399",         # Emerald
    "Understated Humor": "#FBBF24",        # Amber
    "Defensive Hesitation": "#FB923C",     # Orange
    "Ambiguous": "#94A3B8",                # Cool Gray
}


if QT_AVAILABLE:

    class SubtextInsightCard(QtWidgets.QWidget):
        """
        Animated card displaying Gemini's social cue interpretation.
        Fades in upon receiving a cue, auto-dismisses after 10s.
        """
        # Signal for thread-safe cue emission
        cue_received = QtCore.pyqtSignal(dict)

        def __init__(self, parent=None, auto_dismiss_sec: int = UI_CARD_AUTO_DISMISS_SECONDS):
            super().__init__(parent)
            self.auto_dismiss_sec = auto_dismiss_sec
            self.setObjectName("CueCardContainer")
            self._init_ui()

            # Connect signal
            self.cue_received.connect(self._on_cue_received)

            # Auto-dismiss timer
            self.dismiss_timer = QtCore.QTimer(self)
            self.dismiss_timer.setSingleShot(True)
            self.dismiss_timer.timeout.connect(self.hide_card)

            # Initially hidden until a cue triggers
            self.hide()

        def _init_ui(self):
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(12, 10, 12, 10)
            layout.setSpacing(6)

            self.setStyleSheet(
                f"""
                QWidget#CueCardContainer {{
                    background-color: {COLOR_BG_CARD};
                    border: 1px solid {COLOR_BORDER_GLOW};
                    border-radius: 12px;
                }}
                """
            )

            # 1. Top Row: Badge + Confidence + Dismiss
            top_row = QtWidgets.QHBoxLayout()
            top_row.setContentsMargins(0, 0, 0, 0)
            top_row.setSpacing(6)

            self.badge_label = QtWidgets.QLabel("DRY SARCASM / IRONY", self)
            self.badge_label.setStyleSheet(
                """
                color: #C084FC;
                font-family: 'Segoe UI', Inter, sans-serif;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 0.5px;
                background-color: rgba(192, 132, 252, 0.15);
                border: 1px solid rgba(192, 132, 252, 0.35);
                border-radius: 8px;
                padding: 2px 8px;
                """
            )

            self.confidence_label = QtWidgets.QLabel("High Confidence", self)
            self.confidence_label.setStyleSheet(
                f"color: {COLOR_TEXT_MUTED}; font-size: 10px; font-weight: 600;"
            )

            self.dismiss_btn = QtWidgets.QPushButton("✕", self)
            self.dismiss_btn.setFixedSize(18, 18)
            self.dismiss_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: transparent;
                    color: #94A3B8;
                    border: none;
                    font-size: 10px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    color: #FFFFFF;
                }
                """
            )
            self.dismiss_btn.clicked.connect(self.hide_card)

            top_row.addWidget(self.badge_label)
            top_row.addWidget(self.confidence_label)
            top_row.addStretch()
            top_row.addWidget(self.dismiss_btn)

            # 2. Spoken Transcript Quote
            self.transcript_label = QtWidgets.QLabel("", self)
            self.transcript_label.setWordWrap(True)
            self.transcript_label.setStyleSheet(
                "color: #94A3B8; font-style: italic; font-size: 10.5px; font-family: 'Segoe UI', Inter; margin-bottom: 2px;"
            )

            # 3. Explanation Text
            self.explanation_label = QtWidgets.QLabel(
                "Words express enthusiasm, but deadpan expression and flat monotone tone suggest sarcastic humor.",
                self
            )
            self.explanation_label.setWordWrap(True)
            self.explanation_label.setStyleSheet(
                f"""
                color: {COLOR_TEXT_PRIMARY};
                font-family: 'Segoe UI', Inter, sans-serif;
                font-size: 11px;
                line-height: 1.4;
                """
            )

            # 4. Suggested Action Pill Box
            self.action_box = QtWidgets.QFrame(self)
            self.action_box.setStyleSheet(
                """
                QFrame {
                    background-color: rgba(56, 189, 248, 0.08);
                    border-left: 3px solid #38BDF8;
                    border-radius: 4px;
                    padding: 6px;
                }
                """
            )
            action_layout = QtWidgets.QVBoxLayout(self.action_box)
            action_layout.setContentsMargins(6, 4, 6, 4)

            self.action_label = QtWidgets.QLabel(
                "💡 Response Tip: Acknowledge the shared setback lightly rather than taking the praise literally.",
                self.action_box
            )
            self.action_label.setWordWrap(True)
            self.action_label.setStyleSheet(
                "color: #BAE6FD; font-size: 10px; font-weight: 500;"
            )
            action_layout.addWidget(self.action_label)

            layout.addLayout(top_row)
            layout.addWidget(self.transcript_label)
            layout.addWidget(self.explanation_label)
            layout.addWidget(self.action_box)

        def display_cue(self, cue_data: Dict[str, str]):
            """Thread-safe public trigger."""
            self.cue_received.emit(cue_data)

        @QtCore.pyqtSlot(dict)
        def _on_cue_received(self, cue_data: Dict[str, str]):
            """UI Thread Slot: Updates texts, dynamic themes, and starts auto-dismiss timer."""
            cue_type = cue_data.get("social_cue_type", "In Sync / Authentic")
            confidence = cue_data.get("confidence", "Medium")
            explanation = cue_data.get("explanation", "")
            action = cue_data.get("suggested_action", "")
            transcript = str(cue_data.get("transcript", "")).strip()

            color = CUE_BADGE_COLORS.get(cue_type, "#10B981")
            is_sync = ("in sync" in cue_type.lower() or "authentic" in cue_type.lower())

            # Update container border glow
            self.setStyleSheet(
                f"""
                QWidget#CueCardContainer {{
                    background-color: {COLOR_BG_CARD};
                    border: 1px solid {color}66;
                    border-radius: 12px;
                }}
                """
            )

            # Update badge
            self.badge_label.setText(cue_type.upper())
            self.badge_label.setStyleSheet(
                f"""
                color: {color};
                font-family: 'Segoe UI', Inter, sans-serif;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 0.5px;
                background-color: {color}22;
                border: 1px solid {color}55;
                border-radius: 8px;
                padding: 2px 8px;
                """
            )

            self.confidence_label.setText(f"{confidence} Confidence")

            # Update transcript quote
            if transcript and not transcript.startswith("[Speech detected"):
                self.transcript_label.setText(f'"{transcript}"')
                self.transcript_label.show()
            else:
                self.transcript_label.hide()

            self.explanation_label.setText(explanation)

            # Action Box
            tip_prefix = "💡 Communication Note: " if is_sync else "💡 Response Tip: "
            self.action_box.setStyleSheet(
                f"""
                QFrame {{
                    background-color: {color}15;
                    border-left: 3px solid {color};
                    border-radius: 4px;
                    padding: 6px;
                }}
                """
            )
            self.action_label.setText(f"{tip_prefix}{action}")
            self.action_label.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: 500;")

            # Show card and start auto-dismiss timer
            self.show()
            self.dismiss_timer.start(int(self.auto_dismiss_sec * 1000))

        def hide_card(self):
            """Hides the cue card."""
            self.dismiss_timer.stop()
            self.hide()

else:
    # -----------------------------------------------------------------------
    # Zero-Dependency Fallback Cue Card (Tkinter / Headless)
    # -----------------------------------------------------------------------
    class SubtextInsightCard:
        """Lightweight fallback cue card tracking active cue state."""

        def __init__(self, parent=None, auto_dismiss_sec: int = UI_CARD_AUTO_DISMISS_SECONDS):
            self.auto_dismiss_sec = auto_dismiss_sec
            self.is_visible = False
            self.current_cue = None

        def display_cue(self, cue_data: Dict[str, str]):
            self.is_visible = True
            self.current_cue = cue_data
            return self.current_cue

        def hide_card(self):
            self.is_visible = False
            self.current_cue = None
