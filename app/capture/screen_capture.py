"""
Screen Capture & Video ROI Cropping for SocialLens
Captures screen / video call tile at low CPU overhead, maintains a timestamped
ring buffer of frames, and allows dynamic region-of-interest (ROI) calibration.
"""

from collections import deque
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.config import SCREEN_FPS, FRAME_SAMPLE_COUNT

logger = logging.getLogger(__name__)


class ScreenCaptureManager:
    """
    Manages continuous screen grabbing using `mss` with low CPU usage.
    Stores timestamped RGB frames in a circular buffer for audio-aligned temporal analysis.
    """

    def __init__(
        self,
        fps: int = SCREEN_FPS,
        buffer_seconds: float = 8.0,
        roi: Optional[Dict[str, int]] = None
    ):
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.max_frames = int(self.fps * self.buffer_seconds)

        # Region of interest {"top": int, "left": int, "width": int, "height": int}
        # If None, captures primary monitor default region
        self.roi: Dict[str, int] = roi or {"top": 100, "left": 100, "width": 640, "height": 480}

        # Timestamped frame ring buffer: [(timestamp, np.ndarray)]
        self._frame_buffer: deque = deque(maxlen=self.max_frames)
        self._lock = threading.Lock()

        # Threading state
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def set_roi(self, top: int, left: int, width: int, height: int):
        """Updates the capture bounding box (speaker's video tile)."""
        with self._lock:
            self.roi = {
                "top": max(0, int(top)),
                "left": max(0, int(left)),
                "width": max(64, int(width)),
                "height": max(64, int(height))
            }
        logger.info(f"Updated screen capture ROI: {self.roi}")

    def get_roi(self) -> Dict[str, int]:
        """Returns the current ROI dictionary."""
        with self._lock:
            return dict(self.roi)

    def start(self):
        """Starts background screen capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Screen capture thread started.")

    def stop(self):
        """Stops background screen capture thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("Screen capture thread stopped.")

    def _capture_loop(self):
        """Internal grab loop using mss with fallback simulation."""
        mss_instance = None
        try:
            import mss
            mss_instance = mss.mss()
            logger.info("mss screen capture initialized successfully.")
        except Exception as e:
            logger.info(f"mss unavailable ({e}). Using simulated frame stream.")
            mss_instance = None

        frame_interval = 1.0 / self.fps

        while self._running:
            start_t = time.time()

            if mss_instance is not None:
                try:
                    with self._lock:
                        curr_roi = dict(self.roi)
                    # Grab screen region (BGRX format)
                    sct_img = mss_instance.grab(curr_roi)
                    # Convert to RGB numpy array
                    frame_rgb = np.array(sct_img, dtype=np.uint8)[:, :, :3][:, :, ::-1]
                    self.push_frame(frame_rgb, start_t)
                except Exception as e:
                    logger.warning(f"Error during mss grab: {e}")
                    self._push_fallback_frame(start_t)
            else:
                self._push_fallback_frame(start_t)

            elapsed = time.time() - start_t
            sleep_time = max(0.001, frame_interval - elapsed)
            time.sleep(sleep_time)

        if mss_instance is not None:
            try:
                mss_instance.close()
            except Exception:
                pass

    def _push_fallback_frame(self, timestamp: float):
        """Generates a synthetic frame when mss is not installed or offline."""
        with self._lock:
            w, h = self.roi["width"], self.roi["height"]
        frame = np.full((h, w, 3), 160, dtype=np.uint8)
        # Gentle dynamic pattern
        pulse = int(20 * np.sin(timestamp * 2.0))
        frame[:, :, 1] = np.clip(160 + pulse, 0, 255)
        self.push_frame(frame, timestamp)

    def push_frame(self, frame: np.ndarray, timestamp: Optional[float] = None):
        """Appends a frame to the timestamped ring buffer."""
        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            self._frame_buffer.append((ts, frame))

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Returns the most recent frame, or None if buffer is empty."""
        with self._lock:
            if not self._frame_buffer:
                return None
            return self._frame_buffer[-1][1].copy()

    def get_keyframes_in_window(
        self,
        start_time: float,
        end_time: float,
        sample_count: int = FRAME_SAMPLE_COUNT
    ) -> List[np.ndarray]:
        """
        Retrieves `sample_count` evenly spaced keyframes captured between
        `start_time` and `end_time` to align with a speech utterance.
        """
        with self._lock:
            candidates = [f for ts, f in self._frame_buffer if start_time <= ts <= end_time]
            if not candidates and self._frame_buffer:
                # If window had no exact match, use the most recent frames
                candidates = [f for _, f in list(self._frame_buffer)[-sample_count:]]

        if not candidates:
            return []

        if len(candidates) <= sample_count:
            return [f.copy() for f in candidates]

        # Uniformly subsample `sample_count` frames
        indices = np.linspace(0, len(candidates) - 1, sample_count, dtype=int)
        return [candidates[i].copy() for i in indices]

    def save_screenshot(self, filepath: str, frame: Optional[np.ndarray] = None) -> bool:
        """
        Saves a frame to disk. Supports OpenCV, PIL, or pure PPM binary format fallback.
        """
        target_frame = frame if frame is not None else self.get_latest_frame()
        if target_frame is None:
            logger.error("No frame available to save.")
            return False

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # 1. Try OpenCV
        try:
            import cv2
            # Convert RGB to BGR for OpenCV
            bgr = target_frame[:, :, ::-1] if len(target_frame.shape) == 3 else target_frame
            cv2.imwrite(filepath, bgr)
            return True
        except ImportError:
            pass

        # 2. Try PIL
        try:
            from PIL import Image
            img = Image.fromarray(target_frame)
            img.save(filepath)
            return True
        except ImportError:
            pass

        # 3. Fallback: Save as binary PPM (P6) format (standard, opens everywhere)
        try:
            ppm_path = filepath if filepath.endswith(".ppm") else filepath + ".ppm"
            h, w = target_frame.shape[:2]
            header = f"P6\n{w} {h}\n255\n".encode("ascii")
            with open(ppm_path, "wb") as f:
                f.write(header)
                f.write(target_frame.astype(np.uint8).tobytes())
            logger.info(f"Saved fallback screenshot to {ppm_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")
            return False
