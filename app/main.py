"""
SocialLens Application Entry Point
HackGT 2026 | Meta Track: Bringing People Closer Together with AI

Launches the SocialLens Desktop HUD overlay and coordinates real-time
multimodal perception streams across video calls.
"""

import argparse
import logging
import signal
import sys
import time
import warnings

# Suppress harmless WASAPI loopback silence/buffer discontinuity warnings
warnings.filterwarnings("ignore", message=".*data discontinuity in recording.*")
warnings.filterwarnings("ignore", message=".*discontinuity.*")

from app.capture.audio_loopback import AudioLoopbackCapture
from app.capture.screen_capture import ScreenCaptureManager
from app.capture.vad_detector import VADDetector
from app.config import JSD_THRESHOLD
from app.coordinator import PipelineCoordinator, QtCoordinatorBridge
from app.ui.overlay_window import SocialLensOverlayWindow
from app.ui.qt_compat import QT_AVAILABLE, QtWidgets, QtCore

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SocialLens")


def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="SocialLens: Real-Time Multimodal Social Cue HUD for Video Calls"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in interactive hackathon demo simulation mode"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=JSD_THRESHOLD,
        help=f"JSD incongruence gating threshold (default: {JSD_THRESHOLD})"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    logger.info("Initializing SocialLens...")

    # 1. Initialize GUI / Qt Application FIRST (prevents Windows COM/OLE conflict)
    app = None
    if QT_AVAILABLE and QtWidgets is not None:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # 2. Initialize Overlay Window
    window = SocialLensOverlayWindow()

    # 3. Initialize Capture Managers (after OLE has been initialized)
    audio_capture = AudioLoopbackCapture()
    screen_capture = ScreenCaptureManager()
    vad_detector = VADDetector()
    window.screen_capture = screen_capture


    # 3. Setup Coordinator Bridge
    if QT_AVAILABLE:
        bridge = QtCoordinatorBridge()
        # Connect bridge signals to UI components
        if hasattr(window, "telemetry"):
            bridge.telemetry_signal.connect(window.telemetry.update_telemetry)
        if hasattr(window, "cue_card"):
            bridge.cue_signal.connect(window.cue_card.display_cue)
        if hasattr(window, "set_status"):
            bridge.status_signal.connect(window.set_status)
        if hasattr(window, "set_speech_active"):
            bridge.speech_state_signal.connect(window.set_speech_active)
        if hasattr(window, "set_transcript"):
            bridge.transcript_signal.connect(window.set_transcript)

        coordinator = PipelineCoordinator(
            audio_capture=audio_capture,
            screen_capture=screen_capture,
            vad_detector=vad_detector,
            telemetry_callback=bridge.emit_telemetry,
            cue_callback=bridge.emit_cue,
            speech_state_callback=bridge.emit_speech_state,
            status_callback=bridge.emit_status,
            transcript_callback=bridge.emit_transcript,
            jsd_threshold=args.threshold
        )
    else:
        # Fallback callback binding
        def on_telemetry(p_v, p_a, p_s, jsd):
            if hasattr(window, "telemetry"):
                window.telemetry.update_telemetry(p_v, p_a, p_s, jsd)

        def on_cue(cue_data):
            if hasattr(window, "cue_card"):
                window.cue_card.display_cue(cue_data)

        def on_speech_state(is_speaking):
            if hasattr(window, "set_speech_active"):
                window.set_speech_active(is_speaking)

        def on_status(status):
            if hasattr(window, "set_status"):
                window.set_status(status)

        def on_transcript(text):
            if hasattr(window, "set_transcript"):
                window.set_transcript(text)

        coordinator = PipelineCoordinator(
            audio_capture=audio_capture,
            screen_capture=screen_capture,
            vad_detector=vad_detector,
            telemetry_callback=on_telemetry,
            cue_callback=on_cue,
            speech_state_callback=on_speech_state,
            status_callback=on_status,
            transcript_callback=on_transcript,
            jsd_threshold=args.threshold
        )

    # 4. Setup Demo Mode if --demo is passed
    if args.demo:
        from app.mock_demo import MockDemoRunner
        if QT_AVAILABLE:
            demo_runner = MockDemoRunner(
                telemetry_callback=bridge.emit_telemetry,
                cue_callback=bridge.emit_cue,
                reasoner=coordinator.reasoner,
                threshold=args.threshold,
                transcript_callback=bridge.emit_transcript
            )
        else:
            demo_runner = MockDemoRunner(
                telemetry_callback=on_telemetry,
                cue_callback=on_cue,
                reasoner=coordinator.reasoner,
                threshold=args.threshold,
                transcript_callback=on_transcript
            )
        if hasattr(window, "enable_demo_mode"):
            window.enable_demo_mode(demo_runner)
        logger.info("Demo Mode ACTIVE. Hotkeys: [1] Sarcasm, [2] Sincere Praise, [3] Frustration")

    # 5. Clean Shutdown Handler
    _shutting_down = False

    def shutdown_app(*_):
        nonlocal _shutting_down
        if _shutting_down:
            return
        _shutting_down = True
        logger.info("Shutting down SocialLens...")

        # Hide window immediately so user gets instant UI feedback
        if hasattr(window, "hide"):
            try:
                window.hide()
            except Exception:
                pass

        # Disconnect window close signal to avoid recursive re-entrancy
        if hasattr(window, "window_closed"):
            try:
                window.window_closed.disconnect()
            except Exception:
                pass

        try:
            coordinator.stop()
        except Exception:
            pass

        import os
        os._exit(0)

    # Native Windows Console Ctrl Handler for instantaneous, clean Ctrl+C termination
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            def _win_ctrl_handler(dwCtrlType):
                shutdown_app()
                return True

            _HandlerRoutine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
            _ctrl_handler = _HandlerRoutine(_win_ctrl_handler)
            ctypes.windll.kernel32.SetConsoleCtrlHandler(_ctrl_handler, True)
        except Exception as e:
            logger.debug(f"Console ctrl handler setup note: {e}")

    # Hook window close signal
    if hasattr(window, "window_closed"):
        window.window_closed.connect(shutdown_app)

    signal.signal(signal.SIGINT, shutdown_app)
    signal.signal(signal.SIGTERM, shutdown_app)

    # Allow Python to catch Ctrl+C on Windows during Qt event loop
    interrupt_timer = None
    if QT_AVAILABLE and QtCore is not None:
        interrupt_timer = QtCore.QTimer()
        interrupt_timer.timeout.connect(lambda: None)
        interrupt_timer.start(100)

    # 6. Start Perception Coordinator (live mode only; demo mode uses deterministic triggers)
    if not args.demo:
        coordinator.start()
        logger.info("SocialLens HUD is active. Pinned Always-On-Top.")
    else:
        logger.info("SocialLens HUD active in Demo Mode (Background capture inactive).")

    window.show()

    # 7. Event Loop Execution
    if QT_AVAILABLE and app is not None:
        app.exec()
        shutdown_app()
    else:
        try:
            if hasattr(window, "root"):
                window.root.mainloop()
        except KeyboardInterrupt:
            shutdown_app()



if __name__ == "__main__":
    main()
