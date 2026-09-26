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
    Supports dynamic auto-face discovery across the screen and window snapping (Zoom, Meet, Teams).
    """

    def __init__(
        self,
        fps: int = SCREEN_FPS,
        buffer_seconds: float = 8.0,
        roi: Optional[Dict[str, int]] = None,
        auto_track: bool = True
    ):
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.max_frames = int(self.fps * self.buffer_seconds)

        # Region of interest {"top": int, "left": int, "width": int, "height": int}
        self._manual_roi = (roi is not None)
        self.auto_track = auto_track and not self._manual_roi
        self.roi: Dict[str, int] = roi or {"top": 100, "left": 100, "width": 640, "height": 480}

        # Timestamped frame ring buffer: [(timestamp, np.ndarray)]
        self._frame_buffer: deque = deque(maxlen=self.max_frames)
        self._lock = threading.Lock()

        # Threading state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_auto_scan_time = 0.0
        self._is_scanning = False

    def set_roi(self, top: int, left: int, width: int, height: int, is_manual: bool = True):
        """Updates the capture bounding box (speaker's video tile)."""
        with self._lock:
            self.roi = {
                "top": max(0, int(top)),
                "left": max(0, int(left)),
                "width": max(64, int(width)),
                "height": max(64, int(height))
            }
            if is_manual:
                self._manual_roi = True
        logger.info(f"Updated screen capture ROI: {self.roi} (manual={self._manual_roi})")

    def get_roi(self) -> Dict[str, int]:
        """Returns the current ROI dictionary."""
        with self._lock:
            return dict(self.roi)

    def enable_auto_track(self, enable: bool = True):
        """Enables or disables automatic full-screen face tracking."""
        with self._lock:
            self.auto_track = enable
            if enable:
                self._manual_roi = False
        logger.info(f"Auto-track mode set to: {enable}")

    def find_and_lock_face(self, mss_instance=None) -> bool:
        """
        Scans primary monitor to automatically locate the speaker's video tile and center the ROI.
        Downscales monitor frame to ~640px wide for rapid detection (<15ms).
        Returns True if a face was detected and ROI was locked.
        """
        close_mss = False
        try:
            if mss_instance is None:
                import mss
                mss_instance = mss.mss()
                close_mss = True

            # Use primary monitor (monitor 1 if multi-monitor, or monitor 0)
            monitor = mss_instance.monitors[1] if len(mss_instance.monitors) > 1 else mss_instance.monitors[0]
            sct_img = mss_instance.grab(monitor)
            frame_rgb = np.array(sct_img, dtype=np.uint8)[:, :, :3][:, :, ::-1]

            h, w = frame_rgb.shape[:2]
            target_w = 640
            scale = target_w / float(w)
            target_h = int(h * scale)

            # Fast downsampling
            step_x = max(1, w // target_w)
            step_y = max(1, h // target_h)
            small_frame = frame_rgb[::step_y, ::step_x]

            # Detect face on downscaled frame
            face_found, bbox = False, None
            try:
                import cv2
                cascades_path = cv2.data.haarcascades
                cascade = cv2.CascadeClassifier(cascades_path + "haarcascade_frontalface_default.xml")
                gray = cv2.cvtColor(small_frame, cv2.COLOR_RGB2GRAY)
                gray_eq = cv2.equalizeHist(gray)
                faces = cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))
                if len(faces) > 0:
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    face_found, bbox = True, tuple(int(v) for v in largest_face)
            except Exception:
                pass

            if not face_found or bbox is None:
                # Fallback to skin clustering on downscaled frame
                r = small_frame[:, :, 0].astype(np.int32)
                g = small_frame[:, :, 1].astype(np.int32)
                b = small_frame[:, :, 2].astype(np.int32)
                skin = (r > 75) & (g > 40) & (b > 20) & (r > g) & (r > b) & (np.abs(r - g) > 12)
                ys, xs = np.where(skin)
                if len(ys) > 100:
                    y_min, y_max = int(np.min(ys)), int(np.max(ys))
                    x_min, x_max = int(np.min(xs)), int(np.max(xs))
                    bw, bh = x_max - x_min, y_max - y_min
                    if bw > 30 and bh > 30 and 0.5 <= bh / bw <= 2.2:
                        face_found, bbox = True, (x_min, y_min, bw, bh)

            if face_found and bbox is not None:
                bx, by, bw, bh = bbox
                # Map back to monitor coordinate space
                actual_scale_x = w / float(small_frame.shape[1])
                actual_scale_y = h / float(small_frame.shape[0])

                screen_face_x = monitor["left"] + int(bx * actual_scale_x)
                screen_face_y = monitor["top"] + int(by * actual_scale_y)
                screen_face_w = int(bw * actual_scale_x)
                screen_face_h = int(bh * actual_scale_y)

                # Add comfortable video-call tile padding around the face (head & shoulders)
                pad_w = int(screen_face_w * 0.8)
                pad_h = int(screen_face_h * 1.0)
                new_left = max(monitor["left"], screen_face_x - pad_w // 2)
                new_top = max(monitor["top"], screen_face_y - int(pad_h * 0.3))
                new_width = min(monitor["width"] - (new_left - monitor["left"]), max(480, screen_face_w + pad_w))
                new_height = min(monitor["height"] - (new_top - monitor["top"]), max(360, screen_face_h + pad_h))

                self.set_roi(new_top, new_left, new_width, new_height, is_manual=False)
                logger.info(f"Auto-locked ROI to detected face at: {self.roi}")
                return True

        except Exception as e:
            logger.debug(f"Auto-face discovery check ended: {e}")
        finally:
            if close_mss and mss_instance is not None:
                try:
                    mss_instance.close()
                except Exception:
                    pass

        return False

    def snap_to_window(self, title_query: Optional[str] = None) -> bool:
        """
        Locates an active video call window (e.g. Zoom, Meet, Teams, Discord)
        and snaps capture ROI to the window bounds.
        """
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32

            targets = [title_query.lower()] if title_query else [
                "zoom", "meet", "teams", "discord", "skype", "webex", "google meet", "slack", "chrome", "firefox", "edge"
            ]

            found_rect = None
            found_title = None

            def enum_proc(hwnd, lParam):
                nonlocal found_rect, found_title
                if found_rect is not None:
                    return False
                if user32.IsWindowVisible(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buff, length + 1)
                        title_lower = buff.value.lower()
                        for target in targets:
                            if target in title_lower:
                                rect = wintypes.RECT()
                                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                                w = rect.right - rect.left
                                h = rect.bottom - rect.top
                                if w >= 320 and h >= 240:
                                    found_rect = (rect.top, rect.left, w, h)
                                    found_title = buff.value
                                    return False
                return True

            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
            user32.EnumWindows(WNDENUMPROC(enum_proc), 0)

            if found_rect is not None:
                top, left, w, h = found_rect
                self.set_roi(top, left, w, h, is_manual=True)
                logger.info(f"Snapped capture ROI to window '{found_title}': {self.roi}")
                return True

        except Exception as e:
            logger.warning(f"Could not snap to window: {e}")

        return False

    def start(self):
        """Starts background screen capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Screen capture thread started.")

    def stop(self):
        """Stops background screen capture thread without blocking."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.1)
            self._thread = None
        logger.info("Screen capture thread stopped.")

    def _async_auto_scan(self):
        """Executes full-monitor face localization in detached thread without stalling 30 FPS grab loop."""
        try:
            self.find_and_lock_face()
        except Exception as e:
            logger.debug(f"Async auto-face tracking check error: {e}")
        finally:
            self._is_scanning = False

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
        last_grab_error_time = 0.0

        while self._running:
            start_t = time.time()

            if mss_instance is not None:
                # Trigger asynchronous background auto-face discovery without blocking 30 FPS grab loop
                if self.auto_track and not self._manual_roi and (start_t - self._last_auto_scan_time > 6.0):
                    self._last_auto_scan_time = start_t
                    if not self._is_scanning:
                        self._is_scanning = True
                        threading.Thread(target=self._async_auto_scan, daemon=True).start()

                try:
                    with self._lock:
                        curr_roi = dict(self.roi)
                    # Grab screen region (BGRX format)
                    sct_img = mss_instance.grab(curr_roi)
                    # Convert to RGB numpy array
                    frame_rgb = np.array(sct_img, dtype=np.uint8)[:, :, :3][:, :, ::-1]
                    self.push_frame(frame_rgb, start_t)
                except Exception as e:
                    now = time.time()
                    if now - last_grab_error_time >= 5.0:
                        logger.warning(f"Error during mss grab: {e}")
                        last_grab_error_time = now
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
