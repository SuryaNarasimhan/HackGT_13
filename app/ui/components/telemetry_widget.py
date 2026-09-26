"""
Tri-Channel Telemetry Widget & JSD Incongruence Gauge for SocialLens HUD
Displays real-time confidence bars for Face, Tone, and Words,
plus an Incongruence Gauge tracking cross-modal friction.
"""

import logging
from typing import Dict, Optional, Tuple, Union
import numpy as np

from app.pipelines.speaker_baseline import STATUS_LABELS, STATUS_USED
from app.pipelines.taxonomy import get_top_emotion
from app.ui.qt_compat import QT_AVAILABLE, QtCore, QtWidgets, QtGui
from app.ui.styles import (
    COLOR_BG_CARD,
    COLOR_BORDER,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    COLOR_TEXT_MUTED,
    COLOR_JOY,
    COLOR_SURPRISE,
    COLOR_SADNESS,
    COLOR_ANGER,
    COLOR_DISGUST,
    COLOR_FEAR,
    COLOR_NEUTRAL,
    COLOR_GAUGE_SYNC,
    COLOR_GAUGE_NUANCE,
    COLOR_GAUGE_CUE,
)

logger = logging.getLogger(__name__)

EMOTION_COLORS: Dict[str, str] = {
    "joy": COLOR_JOY,
    "surprise": COLOR_SURPRISE,
    "sadness": COLOR_SADNESS,
    "anger": COLOR_ANGER,
    "disgust": COLOR_DISGUST,
    "fear": COLOR_FEAR,
    "neutral": COLOR_NEUTRAL,
}


def get_gauge_status(jsd: float) -> Tuple[str, str]:
    """Returns (status_label, hex_color) corresponding to JSD score."""
    if jsd < 0.35:
        return "In Sync", COLOR_GAUGE_SYNC
    elif jsd <= 0.48:
        return "Nuance", COLOR_GAUGE_NUANCE
    else:
        return "Cue Detected", COLOR_GAUGE_CUE


if QT_AVAILABLE:

    class ChannelBar(QtWidgets.QWidget):
        """Single channel telemetry row: Name | Top Emotion Pill | Progress Bar."""

        def __init__(self, channel_name: str, icon_str: str, parent=None):
            super().__init__(parent)
            self.channel_name = channel_name
            self.icon_str = icon_str
            self._init_ui()

        def _init_ui(self):
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 4, 0, 4)
            layout.setSpacing(4)

            # Top label row: Channel (Left) ... Emotion + % (Right)
            top_row = QtWidgets.QHBoxLayout()
            top_row.setContentsMargins(0, 0, 0, 0)

            self.name_label = QtWidgets.QLabel(f"{self.icon_str} {self.channel_name}", self)
            self.name_label.setStyleSheet(
                f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; font-weight: 600; font-family: 'Segoe UI', Inter;"
            )

            self.emotion_label = QtWidgets.QLabel("Neutral (0%)", self)
            self.emotion_label.setStyleSheet(
                f"color: {COLOR_TEXT_PRIMARY}; font-size: 11px; font-weight: 700; font-family: 'Segoe UI', Inter;"
            )

            # Why the channel didn't count this time (e.g. "usual for them"); hidden when it counted
            self.note_label = QtWidgets.QLabel("", self)
            self.note_label.setStyleSheet(
                f"color: {COLOR_TEXT_MUTED}; font-size: 10px; font-style: italic; font-family: 'Segoe UI', Inter;"
            )
            self.note_label.hide()

            top_row.addWidget(self.name_label)
            top_row.addWidget(self.note_label)
            top_row.addStretch()
            top_row.addWidget(self.emotion_label)

            # Progress Bar
            self.progress = QtWidgets.QProgressBar(self)
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress.setTextVisible(False)
            self.progress.setFixedHeight(6)
            self.set_color(COLOR_NEUTRAL)

            layout.addLayout(top_row)
            layout.addWidget(self.progress)

        def set_color(self, hex_color: str):
            """Applies dynamic color to the bar chunk."""
            self.progress.setStyleSheet(
                f"""
                QProgressBar {{
                    background-color: rgba(255, 255, 255, 0.08);
                    border-radius: 3px;
                }}
                QProgressBar::chunk {{
                    background-color: {hex_color};
                    border-radius: 3px;
                }}
                """
            )

        def update_distribution(self, p_vec: np.ndarray):
            """Updates top emotion label, value, and bar accent color."""
            top_emotion, score = get_top_emotion(p_vec)
            pct = int(score * 100)
            self.emotion_label.setText(f"{top_emotion.capitalize()} ({pct}%)")
            self.progress.setValue(pct)
            color = EMOTION_COLORS.get(top_emotion, COLOR_NEUTRAL)
            self.set_color(color)

        def set_note(self, note: Optional[str]):
            """Shows why the channel didn't count (e.g. 'usual for them'); None clears it."""
            if note:
                self.note_label.setText(f"· {note}")
                self.note_label.show()
            else:
                self.note_label.clear()
                self.note_label.hide()


    class IncongruenceGauge(QtWidgets.QWidget):
        """Gauge displaying cross-modal JSD score and dynamic status indicator."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self._init_ui()

        def _init_ui(self):
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 6, 0, 4)
            layout.setSpacing(4)

            # Header row: 'Cross-Modal Divergence' ... 'In Sync (0.05)'
            header_row = QtWidgets.QHBoxLayout()
            header_row.setContentsMargins(0, 0, 0, 0)

            title = QtWidgets.QLabel("Divergence (JSD)", self)
            title.setStyleSheet(
                f"color: {COLOR_TEXT_MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;"
            )

            self.status_badge = QtWidgets.QLabel("In Sync (0.00)", self)
            self.status_badge.setStyleSheet(
                f"color: {COLOR_GAUGE_SYNC}; font-size: 11px; font-weight: 700;"
            )

            header_row.addWidget(title)
            header_row.addStretch()
            header_row.addWidget(self.status_badge)

            # Gauge Bar
            self.gauge_bar = QtWidgets.QProgressBar(self)
            self.gauge_bar.setRange(0, 100)
            self.gauge_bar.setValue(0)
            self.gauge_bar.setTextVisible(False)
            self.gauge_bar.setFixedHeight(8)
            self.set_gauge_color(COLOR_GAUGE_SYNC)

            layout.addLayout(header_row)
            layout.addWidget(self.gauge_bar)

        def set_gauge_color(self, hex_color: str):
            self.gauge_bar.setStyleSheet(
                f"""
                QProgressBar {{
                    background-color: rgba(255, 255, 255, 0.08);
                    border-radius: 4px;
                }}
                QProgressBar::chunk {{
                    background-color: {hex_color};
                    border-radius: 4px;
                }}
                """
            )

        def update_jsd(self, jsd_score: float):
            """Updates gauge progress, score badge, and transition color."""
            clamped = float(np.clip(jsd_score, 0.0, 1.0))
            pct = int(clamped * 100)
            status_text, color = get_gauge_status(clamped)

            self.status_badge.setText(f"{status_text} ({clamped:.2f})")
            self.status_badge.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: 700;")
            self.gauge_bar.setValue(pct)
            self.set_gauge_color(color)


    class TelemetryWidget(QtWidgets.QWidget):
        """
        Complete Telemetry card combining Face, Tone, Words bars and the Incongruence Gauge.
        Receives updates via thread-safe Qt Signals.
        """
        # Qt Signals for safe cross-thread emission
        telemetry_updated = QtCore.pyqtSignal(object, object, object, float)
        channel_status_updated = QtCore.pyqtSignal(dict)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("ContentCard")
            self.latest_video = None
            self.latest_audio = None
            self.latest_semantic = None
            self.latest_jsd = 0.0
            self.latest_channel_status = {}
            self._init_ui()
            # Connect internal signals to UI slots
            self.telemetry_updated.connect(self._on_telemetry_updated)
            self.channel_status_updated.connect(self._on_channel_status_updated)

        def _init_ui(self):
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(12, 10, 12, 10)
            layout.setSpacing(6)

            self.face_bar = ChannelBar("Face", "👤", self)
            self.tone_bar = ChannelBar("Tone", "🎙️", self)
            self.words_bar = ChannelBar("Words", "💬", self)
            self.gauge = IncongruenceGauge(self)

            layout.addWidget(self.face_bar)
            layout.addWidget(self.tone_bar)
            layout.addWidget(self.words_bar)
            layout.addWidget(self.gauge)

        def update_telemetry(
            self,
            p_video: np.ndarray,
            p_audio: np.ndarray,
            p_semantic: np.ndarray,
            jsd_score: float
        ):
            """Thread-safe public entry point."""
            self.latest_video = p_video
            self.latest_audio = p_audio
            self.latest_semantic = p_semantic
            self.latest_jsd = jsd_score
            self.telemetry_updated.emit(p_video, p_audio, p_semantic, jsd_score)
            status, color = get_gauge_status(jsd_score)
            return {
                "face": get_top_emotion(p_video),
                "tone": get_top_emotion(p_audio),
                "words": get_top_emotion(p_semantic),
                "jsd": jsd_score,
                "status": status,
                "color": color
            }



        @QtCore.pyqtSlot(object, object, object, float)
        def _on_telemetry_updated(
            self,
            p_video: np.ndarray,
            p_audio: np.ndarray,
            p_semantic: np.ndarray,
            jsd_score: float
        ):
            """UI Thread Slot."""
            self.face_bar.update_distribution(p_video)
            self.tone_bar.update_distribution(p_audio)
            self.words_bar.update_distribution(p_semantic)
            self.gauge.update_jsd(jsd_score)

        def update_channel_status(self, channel_status: Dict[str, str]):
            """Thread-safe public entry point: marks channels that didn't count, and why."""
            self.latest_channel_status = dict(channel_status)
            self.channel_status_updated.emit(dict(channel_status))

        @QtCore.pyqtSlot(dict)
        def _on_channel_status_updated(self, channel_status: Dict[str, str]):
            """UI Thread Slot."""
            for name, bar in (("face", self.face_bar), ("tone", self.tone_bar), ("words", self.words_bar)):
                status = channel_status.get(name)
                bar.set_note(None if status in (None, STATUS_USED) else STATUS_LABELS.get(status, status))

else:
    # -----------------------------------------------------------------------
    # Zero-Dependency Fallback Telemetry Widget (Tkinter / Headless)
    # -----------------------------------------------------------------------
    class TelemetryWidget:
        """Lightweight fallback widget maintaining telemetry state."""

        def __init__(self, parent=None):
            self.latest_video = None
            self.latest_audio = None
            self.latest_semantic = None
            self.latest_jsd = 0.0
            self.latest_channel_status = {}

        def update_channel_status(self, channel_status: Dict[str, str]):
            self.latest_channel_status = dict(channel_status)

        def update_telemetry(
            self,
            p_video: np.ndarray,
            p_audio: np.ndarray,
            p_semantic: np.ndarray,
            jsd_score: float
        ):
            self.latest_video = p_video
            self.latest_audio = p_audio
            self.latest_semantic = p_semantic
            self.latest_jsd = jsd_score
            status, color = get_gauge_status(jsd_score)
            return {
                "face": get_top_emotion(p_video),
                "tone": get_top_emotion(p_audio),
                "words": get_top_emotion(p_semantic),
                "jsd": jsd_score,
                "status": status,
                "color": color
            }
