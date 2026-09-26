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


    # 3. Setup Coordinator Bridge
    if QT_AVAILABLE:
        bridge = QtCoordinatorBridge()
        # Connect bridge signals to UI components
        if hasattr(window, "telemetry"):
            bridge.telemetry_signal.connect(window.telemetry.update_telemetry)
        if hasattr(window, "cue_card"):
            bridge.cue_signal.connect(window.cue_card.display_cue)

        coordinator = PipelineCoordinator(
            audio_capture=audio_capture,
            screen_capture=screen_capture,
            vad_detector=vad_detector,
            telemetry_callback=bridge.emit_telemetry,
            cue_callback=bridge.emit_cue,
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

        coordinator = PipelineCoordinator(
            audio_capture=audio_capture,
            screen_capture=screen_capture,
            vad_detector=vad_detector,
            telemetry_callback=on_telemetry,
            cue_callback=on_cue,
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
                threshold=args.threshold
            )
        else:
            demo_runner = MockDemoRunner(
                telemetry_callback=on_telemetry,
                cue_callback=on_cue,
                reasoner=coordinator.reasoner,
                threshold=args.threshold
            )
        if hasattr(window, "enable_demo_mode"):
            window.enable_demo_mode(demo_runner)
        logger.info("Demo Mode ACTIVE. Hotkeys: [1] Sarcasm, [2] Sincere Praise, [3] Frustration")

    # 5. Clean Shutdown Handler
    def shutdown_app(*_):
        logger.info("Shutting down SocialLens gracefully...")
        try:
            coordinator.stop()
        except Exception:
            pass
        if hasattr(window, "close"):
            try:
                window.close()
            except Exception:
                pass
        if app is not None:
            try:
                app.quit()
            except Exception:
                pass
        import os
        os._exit(0)

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
        interrupt_timer.start(200)

    # 6. Start Perception Coordinator
    coordinator.start()
    window.show()

    logger.info("SocialLens HUD is active. Pinned Always-On-Top.")

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
