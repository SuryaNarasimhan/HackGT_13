"""
Desktop Overlay Window for SocialLens
Frameless, transparent, draggable, and pinned Always-on-Top over video calls.
Implements PyQt6 glassmorphic UI with a Tkinter fallback for zero-dependency execution.
"""

import logging
import sys
from typing import Optional

from app.config import UI_WINDOW_WIDTH, UI_WINDOW_HEIGHT
from app.ui.qt_compat import QT_AVAILABLE, QtCore, QtWidgets, QtGui
from app.ui.styles import OVERLAY_STYLESHEET

logger = logging.getLogger(__name__)


if QT_AVAILABLE:

    class SocialLensOverlayWindow(QtWidgets.QMainWindow):
        """
        PyQt6 Frameless, Translucent, Always-On-Top Desktop HUD Window.
        """
        window_closed = QtCore.pyqtSignal()

        def __init__(self, width: int = UI_WINDOW_WIDTH, height: int = UI_WINDOW_HEIGHT):
            super().__init__()
            self.window_width = width
            self.window_height = height
            self._drag_pos: Optional[QtCore.QPoint] = None
            self._is_collapsed = False
            self.demo_runner = None

            self._init_window_flags()
            self._init_ui()

        def _init_window_flags(self):
            """Sets frameless, translucent, and always-on-top window hints."""
            self.setWindowFlags(
                QtCore.Qt.WindowType.FramelessWindowHint
                | QtCore.Qt.WindowType.WindowStaysOnTopHint
                | QtCore.Qt.WindowType.SubWindow
            )
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.resize(self.window_width, self.window_height)
            # Default position: Top-right corner of screen
            self.move(100, 100)

        def _init_ui(self):
            """Initializes layout and widgets."""
            self.setStyleSheet(OVERLAY_STYLESHEET)

            # Central glass container
            self.central_container = QtWidgets.QWidget(self)
            self.central_container.setObjectName("CentralContainer")
            self.setCentralWidget(self.central_container)

            self.main_layout = QtWidgets.QVBoxLayout(self.central_container)
            self.main_layout.setContentsMargins(12, 12, 12, 12)
            self.main_layout.setSpacing(10)

            # 1. Header Bar
            self.header_bar = QtWidgets.QWidget(self.central_container)
            self.header_bar.setObjectName("HeaderBar")
            header_layout = QtWidgets.QHBoxLayout(self.header_bar)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(8)

            # Title & Logo
            self.title_label = QtWidgets.QLabel("SocialLens", self.header_bar)
            self.title_label.setObjectName("TitleLabel")

            # Status pill indicator ("● Listening")
            self.status_pill = QtWidgets.QLabel("● Listening", self.header_bar)
            self.status_pill.setObjectName("StatusPill")

            # Window controls
            self.minimize_btn = QtWidgets.QPushButton("–", self.header_bar)
            self.minimize_btn.setObjectName("MinimizeBtn")
            self.minimize_btn.setToolTip("Toggle Compact View")
            self.minimize_btn.clicked.connect(self.toggle_collapse)

            self.close_btn = QtWidgets.QPushButton("✕", self.header_bar)
            self.close_btn.setObjectName("CloseBtn")
            self.close_btn.setToolTip("Close Overlay")
            self.close_btn.clicked.connect(self.close)

            header_layout.addWidget(self.title_label)
            header_layout.addWidget(self.status_pill)
            header_layout.addStretch()
            header_layout.addWidget(self.minimize_btn)
            header_layout.addWidget(self.close_btn)

            self.main_layout.addWidget(self.header_bar)

            # 2. Content Area
            from app.ui.components.telemetry_widget import TelemetryWidget
            from app.ui.components.cue_card import SubtextInsightCard

            self.telemetry = TelemetryWidget(self.central_container)
            self.main_layout.addWidget(self.telemetry)

            self.cue_card = SubtextInsightCard(self.central_container)
            self.main_layout.addWidget(self.cue_card)

            # Quick action buttons row: Auto-Lock Video Tile & Preview Cue
            action_btn_box = QtWidgets.QWidget(self.central_container)
            action_layout = QtWidgets.QHBoxLayout(action_btn_box)
            action_layout.setContentsMargins(0, 0, 0, 0)
            action_layout.setSpacing(6)

            self.lock_video_btn = QtWidgets.QPushButton("🎯 Auto-Lock Video", action_btn_box)
            self.lock_video_btn.setToolTip("Auto-detect video call window or speaker face")
            self.lock_video_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: rgba(56, 189, 248, 0.12);
                    color: #38BDF8;
                    border: 1px solid rgba(56, 189, 248, 0.3);
                    border-radius: 8px;
                    font-size: 10px;
                    padding: 5px 8px;
                }
                QPushButton:hover {
                    background-color: rgba(56, 189, 248, 0.25);
                    color: #FFFFFF;
                }
                """
            )
            self.lock_video_btn.clicked.connect(self._trigger_auto_lock_video)

            self.test_preview_btn = QtWidgets.QPushButton("⚡ Preview Cue", action_btn_box)
            self.test_preview_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: rgba(255, 255, 255, 0.05);
                    color: #94A3B8;
                    border: 1px dashed rgba(255, 255, 255, 0.15);
                    border-radius: 8px;
                    font-size: 10px;
                    padding: 5px 8px;
                }
                QPushButton:hover {
                    background-color: rgba(168, 85, 247, 0.2);
                    color: #FFFFFF;
                }
                """
            )
            self.test_preview_btn.clicked.connect(self._trigger_test_cue)

            action_layout.addWidget(self.lock_video_btn)
            action_layout.addWidget(self.test_preview_btn)
            self.main_layout.addWidget(action_btn_box)

        def _trigger_auto_lock_video(self):
            """Triggers dynamic auto-face discovery or window snapping."""
            screen_capture = getattr(self, "screen_capture", None)
            if screen_capture is not None:
                # Try snapping to active video meeting window first, then full-screen face scan
                snapped = screen_capture.snap_to_window()
                if not snapped:
                    snapped = screen_capture.find_and_lock_face()

                if snapped:
                    roi = screen_capture.get_roi()
                    self.set_status(f"Locked ({roi['width']}x{roi['height']})")
                else:
                    self.set_status("Scanning screen...")
            else:
                self.set_status("Locked (Auto)")

        def _trigger_test_cue(self):
            """Manual trigger for demo and testing preview."""
            sample_cue = {
                "social_cue_type": "Dry Sarcasm / Irony",
                "confidence": "High",
                "explanation": "Spoken words express enthusiasm ('Great job'), but vocal tone was monotone and facial expression was deadpan.",
                "suggested_action": "Acknowledge the shared irony lightly rather than taking the literal praise at face value."
            }
            self.cue_card.display_cue(sample_cue)

        def enable_demo_mode(self, demo_runner):
            """Enables demo mode HUD controls and hotkeys."""
            self.demo_runner = demo_runner
            self.status_pill.setText("● Demo Mode (Press 1, 2, 3)")
            self.status_pill.setStyleSheet(
                "color: #FBBF24; background-color: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 10px; padding: 2px 8px;"
            )

            # Demo action buttons row
            demo_box = QtWidgets.QWidget(self.central_container)
            demo_layout = QtWidgets.QHBoxLayout(demo_box)
            demo_layout.setContentsMargins(0, 4, 0, 0)
            demo_layout.setSpacing(4)

            btn_sarcasm = QtWidgets.QPushButton("[1] Sarcasm", demo_box)
            btn_sarcasm.setStyleSheet("font-size: 9px; padding: 3px 6px; background-color: rgba(192, 132, 252, 0.2); color: #C084FC; border-radius: 6px;")
            btn_sarcasm.clicked.connect(lambda: self.trigger_demo_scenario(1))

            btn_praise = QtWidgets.QPushButton("[2] Praise", demo_box)
            btn_praise.setStyleSheet("font-size: 9px; padding: 3px 6px; background-color: rgba(16, 185, 129, 0.2); color: #34D399; border-radius: 6px;")
            btn_praise.clicked.connect(lambda: self.trigger_demo_scenario(2))

            btn_stress = QtWidgets.QPushButton("[3] Frustration", demo_box)
            btn_stress.setStyleSheet("font-size: 9px; padding: 3px 6px; background-color: rgba(248, 113, 113, 0.2); color: #F87171; border-radius: 6px;")
            btn_stress.clicked.connect(lambda: self.trigger_demo_scenario(3))

            demo_layout.addWidget(btn_sarcasm)
            demo_layout.addWidget(btn_praise)
            demo_layout.addWidget(btn_stress)

            self.main_layout.addWidget(demo_box)

        def trigger_demo_scenario(self, scenario_id: int):
            """Executes a demo scenario asynchronously without freezing the Qt main thread."""
            if getattr(self, "_demo_in_progress", False):
                logger.info(f"Scenario analysis currently running. Ignoring click for scenario {scenario_id}.")
                return
            self._demo_in_progress = True

            scenario_names = {1: "Sarcasm", 2: "Praise", 3: "Frustration"}
            name = scenario_names.get(scenario_id, str(scenario_id))
            self.status_pill.setText(f"● Analyzing [{name}]...")
            self.status_pill.setStyleSheet(
                "color: #38BDF8; background-color: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 10px; padding: 2px 8px;"
            )

            import threading
            def _worker():
                try:
                    self.demo_runner.trigger_scenario(scenario_id)
                except Exception as e:
                    logger.error(f"Error executing demo scenario {scenario_id}: {e}", exc_info=True)
                finally:
                    self._demo_in_progress = False
                    QtCore.QTimer.singleShot(0, self._restore_demo_pill)

            threading.Thread(target=_worker, daemon=True).start()

        def _restore_demo_pill(self):
            """Restores demo status pill label."""
            if getattr(self, "demo_runner", None) is not None:
                self.status_pill.setText("● Demo Mode (Press 1, 2, 3)")
                self.status_pill.setStyleSheet(
                    "color: #FBBF24; background-color: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 10px; padding: 2px 8px;"
                )

        def keyPressEvent(self, event):
            """Hotkey triggers (keys 1, 2, 3) for live judging presentations."""
            if getattr(self, "demo_runner", None) is not None:
                key = event.key()
                if key == QtCore.Qt.Key.Key_1:
                    self.trigger_demo_scenario(1)
                    event.accept()
                    return
                elif key == QtCore.Qt.Key.Key_2:
                    self.trigger_demo_scenario(2)
                    event.accept()
                    return
                elif key == QtCore.Qt.Key.Key_3:
                    self.trigger_demo_scenario(3)
                    event.accept()
                    return
            super().keyPressEvent(event)

        def set_status(self, status: str):
            """Updates HUD status indicator pill."""
            if getattr(self, "demo_runner", None) is not None:
                return  # Preserve demo instructions
            if status == "Speaking":
                self.status_pill.setText("● Speaking...")
                self.status_pill.setStyleSheet(
                    "color: #38BDF8; background-color: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 10px; padding: 2px 8px;"
                )
            elif status == "Analyzing":
                self.status_pill.setText("● Analyzing...")
                self.status_pill.setStyleSheet(
                    "color: #FBBF24; background-color: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 10px; padding: 2px 8px;"
                )
            else:
                self.status_pill.setText("● Listening")
                self.status_pill.setStyleSheet(
                    "color: #34D399; background-color: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 2px 8px;"
                )

        def set_speech_active(self, is_speaking: bool):
            """Slot for speech state toggles."""
            self.set_status("Speaking" if is_speaking else "Listening")

        def toggle_collapse(self):
            """Toggles between full HUD card and minimized pill bar."""
            self._is_collapsed = not self._is_collapsed
            if self._is_collapsed:
                self.resize(self.window_width, 50)
                self.minimize_btn.setText("+")
            else:
                self.resize(self.window_width, self.window_height)
                self.minimize_btn.setText("–")


        def mousePressEvent(self, event):
            """Captures initial mouse click position for smooth dragging."""
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                pos = event.globalPosition().toPoint()
                self._drag_pos = pos - self.frameGeometry().topLeft()
                event.accept()

        def mouseMoveEvent(self, event):
            """Moves the frameless overlay with the mouse cursor."""
            if event.buttons() == QtCore.Qt.MouseButton.LeftButton and self._drag_pos is not None:
                self.move(event.globalPosition().toPoint() - self._drag_pos)
                event.accept()

        def mouseReleaseEvent(self, event):
            """Resets dragging anchor."""
            self._drag_pos = None

        def closeEvent(self, event):
            """Emits window_closed and accepts event to cleanly terminate app."""
            try:
                self.window_closed.emit()
            except Exception:
                pass
            super().closeEvent(event)


else:
    # -----------------------------------------------------------------------
    # Tkinter Fallback: Zero-Dependency Translucent Desktop HUD Window
    # -----------------------------------------------------------------------
    import tkinter as tk

    class SocialLensOverlayWindow:
        """
        Lightweight Tkinter fallback overlay that provides frameless, translucent,
        always-on-top dragging without requiring PyQt6 pre-installed.
        """

        def __init__(self, width: int = UI_WINDOW_WIDTH, height: int = UI_WINDOW_HEIGHT):
            self.window_width = width
            self.window_height = height
            self._drag_x = 0
            self._drag_y = 0
            self._is_collapsed = False
            self.demo_runner = None

            self.root = tk.Tk()
            self._init_window()

        def _init_window(self):
            self.root.title("SocialLens HUD")
            self.root.geometry(f"{self.window_width}x{self.window_height}+100+100")
            
            # Frameless & Always On Top
            self.root.overrideredirect(True)
            self.root.attributes("-topmost", True)
            self.root.attributes("-alpha", 0.92)  # Semi-transparent glass

            # Styling
            self.root.configure(bg="#12161F")

            # Main Container Frame
            self.container = tk.Frame(self.root, bg="#12161F", highlightbackground="#334155", highlightthickness=1)
            self.container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

            # Header Bar
            self.header = tk.Frame(self.container, bg="#12161F", height=32)
            self.header.pack(fill=tk.X, padx=8, pady=6)

            title = tk.Label(
                self.header,
                text="SocialLens",
                fg="#FFFFFF",
                bg="#12161F",
                font=("Segoe UI", 10, "bold")
            )
            title.pack(side=tk.LEFT)

            self.status_pill = tk.Label(
                self.header,
                text="● Listening",
                fg="#34D399",
                bg="#1E293B",
                font=("Segoe UI", 8, "bold"),
                padx=6,
                pady=1
            )
            self.status_pill.pack(side=tk.LEFT, padx=8)

            close_btn = tk.Button(
                self.header,
                text="✕",
                fg="#94A3B8",
                bg="#1E293B",
                bd=0,
                padx=5,
                pady=0,
                command=self.root.destroy
            )
            close_btn.pack(side=tk.RIGHT)

            # Drag Bindings
            self.header.bind("<Button-1>", self._start_drag)
            self.header.bind("<B1-Motion>", self._do_drag)
            title.bind("<Button-1>", self._start_drag)
            title.bind("<B1-Motion>", self._do_drag)
            self.container.bind("<Button-1>", self._start_drag)
            self.container.bind("<B1-Motion>", self._do_drag)

        def _start_drag(self, event):
            self._drag_x = event.x
            self._drag_y = event.y

        def _do_drag(self, event):
            x = self.root.winfo_x() + (event.x - self._drag_x)
            y = self.root.winfo_y() + (event.y - self._drag_y)
            self.root.geometry(f"+{x}+{y}")

        def set_status(self, status: str):
            if getattr(self, "demo_runner", None) is not None:
                return
            if hasattr(self, "status_pill"):
                self.status_pill.config(text=f"● {status}")

        def set_speech_active(self, is_speaking: bool):
            self.set_status("Speaking" if is_speaking else "Listening")

        def show(self):
            self.root.update()

        def close(self):
            self.root.destroy()
