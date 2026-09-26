"""
User Microphone Audio Capture for SocialLens
Captures local user microphone speech via soundcard and downsamples to 16kHz mono.
Provides user conversational audio chunks for two-way dialogue context tracking.
"""

import logging
import threading
import time
from typing import Callable, List, Optional
import warnings
import numpy as np
from scipy import signal

# Suppress harmless WASAPI buffer discontinuity warnings
warnings.filterwarnings("ignore", message=".*data discontinuity in recording.*")
try:
    import soundcard as sc
    if hasattr(sc, "SoundcardRuntimeWarning"):
        warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)
except (ImportError, AttributeError):
    pass

from app.config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_CHUNK_SIZE
)

logger = logging.getLogger(__name__)


class UserMicrophoneCapture:
    """
    Captures the local user's microphone audio (16kHz mono).
    Used to track the user's conversational turns and questions for context grounding.
    """

    def __init__(
        self,
        target_sample_rate: int = AUDIO_SAMPLE_RATE,
        chunk_size: int = AUDIO_CHUNK_SIZE
    ):
        self.target_rate = target_sample_rate
        self.chunk_size = chunk_size

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._chunk_callbacks: List[Callable[[np.ndarray], None]] = []
        self._lock = threading.Lock()

    def register_chunk_callback(self, cb: Callable[[np.ndarray], None]):
        """Registers a listener for live 16kHz mono microphone chunks."""
        self._chunk_callbacks.append(cb)

    def start(self):
        """Starts background microphone recording thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="UserMicThread")
        self._thread.start()
        logger.info("User microphone capture thread started.")

    def stop(self):
        """Gracefully halts microphone capture."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None
        logger.info("User microphone capture stopped.")

    def _capture_loop(self):
        """Continuous microphone recording with downsampling and simulation fallback."""
        mic = None
        soundcard_loaded = False

        try:
            import soundcard as sc
            mic = sc.default_microphone()
            soundcard_loaded = True
            logger.info(f"Connected to local microphone: {mic.name}")
        except Exception as e:
            logger.info(f"Local microphone unavailable ({e}). Running in mic simulation mode.")
            soundcard_loaded = False

        if soundcard_loaded and mic is not None:
            try:
                native_rate = 48000
                with mic.recorder(samplerate=native_rate, channels=1) as recorder:
                    chunk_native = int(native_rate * (self.chunk_size / self.target_rate))
                    while self._running:
                        data = recorder.record(numframes=chunk_native)
                        if len(data) > 0:
                            # Downsample to 16kHz mono
                            resampled = signal.resample_poly(data[:, 0], self.target_rate, native_rate)
                            chunk_16k = resampled.astype(np.float32)
                            self._dispatch_chunk(chunk_16k)
            except Exception as e:
                logger.warning(f"Microphone recording stream error: {e}. Reverting to idle mode.")

        # Idle fallback loop if microphone hardware is unavailable
        while self._running:
            time.sleep(self.chunk_size / self.target_rate)
            # Ambient silence
            silent_chunk = np.zeros(self.chunk_size, dtype=np.float32)
            self._dispatch_chunk(silent_chunk)

    def _dispatch_chunk(self, chunk: np.ndarray):
        """Delivers audio chunk to registered callbacks."""
        for cb in self._chunk_callbacks:
            try:
                cb(chunk)
            except Exception as e:
                logger.error(f"Error in mic chunk callback: {e}")
