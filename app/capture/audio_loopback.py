"""
Windows WASAPI Speaker Loopback Audio Capture for SocialLens
Continuously captures desktop/Zoom audio output into a thread-safe circular buffer (16kHz mono).
"""

import logging
import threading
import time
from typing import Callable, List, Optional
import warnings
import numpy as np
from scipy import signal

# Suppress harmless WASAPI loopback silence/buffer discontinuity warnings
warnings.filterwarnings("ignore", message=".*data discontinuity in recording.*")
try:
    import soundcard as sc
    if hasattr(sc, "SoundcardRuntimeWarning"):
        warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)
except (ImportError, AttributeError):
    pass


from app.config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_BUFFER_SECONDS,
    AUDIO_CHUNK_SIZE
)

logger = logging.getLogger(__name__)


class AudioLoopbackCapture:
    """
    Captures system audio via Windows WASAPI loopback into a thread-safe circular buffer.
    Automatically downsamples to 16kHz mono for speech models.
    """

    def __init__(
        self,
        target_sample_rate: int = AUDIO_SAMPLE_RATE,
        buffer_duration: float = AUDIO_BUFFER_SECONDS,
        chunk_size: int = AUDIO_CHUNK_SIZE
    ):
        self.target_rate = target_sample_rate
        self.buffer_duration = buffer_duration
        self.buffer_capacity = int(self.target_rate * self.buffer_duration)
        self.chunk_size = chunk_size

        # Circular buffer
        self._buffer = np.zeros(self.buffer_capacity, dtype=np.float32)
        self._write_pos = 0
        self._total_samples_written = 0
        self._lock = threading.Lock()

        # Threading state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._chunk_callbacks: List[Callable[[np.ndarray], None]] = []

    def register_chunk_callback(self, cb: Callable[[np.ndarray], None]):
        """Registers a callback receiving every new 16kHz mono audio chunk."""
        self._chunk_callbacks.append(cb)

    def start(self):
        """Starts the background audio capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Audio loopback capture thread started.")

    def stop(self):
        """Stops the audio capture thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None
        logger.info("Audio loopback capture stopped.")

    def _capture_loop(self):
        """Internal capture loop using soundcard WASAPI loopback, with fallback simulation."""
        soundcard_loaded = False
        sc_mic = None

        try:
            import soundcard as sc
            # Get default speaker and its loopback microphone
            speaker = sc.default_speaker()
            sc_mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            soundcard_loaded = True
            logger.info(f"Connected to Windows WASAPI loopback: {speaker.name}")
        except Exception as e:
            logger.info(f"Soundcard WASAPI unavailable ({e}). Running in loopback simulation mode.")
            soundcard_loaded = False

        if soundcard_loaded and sc_mic is not None:
            try:
                native_rate = 48000
                with sc_mic.recorder(samplerate=native_rate, channels=1) as recorder:
                    chunk_native = int(native_rate * (self.chunk_size / self.target_rate))
                    while self._running:
                        data = recorder.record(numframes=chunk_native)
                        # Downsample to 16kHz mono
                        resampled = signal.resample_poly(data[:, 0], self.target_rate, native_rate)
                        self._append_chunk(resampled.astype(np.float32))
            except Exception as e:
                logger.warning(f"Native recorder error: {e}. Switching to simulation fallback.")

        # Fallback simulation loop (keeps thread alive, feeds silence/gentle ambient noise)
        while self._running:
            time.sleep(self.chunk_size / self.target_rate)
            # Simulated ambient low-level silence
            ambient = (np.random.randn(self.chunk_size) * 0.001).astype(np.float32)
            self._append_chunk(ambient)

    def _append_chunk(self, chunk: np.ndarray):
        """Thread-safe write into the circular buffer."""
        chunk_len = len(chunk)
        if chunk_len == 0:
            return

        with self._lock:
            if self._write_pos + chunk_len <= self.buffer_capacity:
                self._buffer[self._write_pos : self._write_pos + chunk_len] = chunk
                self._write_pos = (self._write_pos + chunk_len) % self.buffer_capacity
            else:
                first_part = self.buffer_capacity - self._write_pos
                self._buffer[self._write_pos :] = chunk[:first_part]
                second_part = chunk_len - first_part
                self._buffer[:second_part] = chunk[first_part:]
                self._write_pos = second_part

            self._total_samples_written += chunk_len

        # Notify chunk listeners (e.g., VAD)
        for cb in self._chunk_callbacks:
            try:
                cb(chunk)
            except Exception as e:
                logger.error(f"Error in audio chunk callback: {e}")

    def feed_simulated_audio(self, audio: np.ndarray):
        """Directly feeds an audio segment into the buffer and callbacks (for tests & demo)."""
        self._append_chunk(audio.astype(np.float32))

    def get_last_n_seconds(self, seconds: float) -> np.ndarray:
        """Returns the most recent N seconds of recorded audio as a contiguous 1D array."""
        samples_needed = int(seconds * self.target_rate)
        samples_needed = min(samples_needed, self.buffer_capacity)

        with self._lock:
            if self._total_samples_written < self.buffer_capacity:
                # Buffer has not wrapped yet
                available = min(self._write_pos, samples_needed)
                start_idx = max(0, self._write_pos - available)
                return self._buffer[start_idx : self._write_pos].copy()
            else:
                # Buffer has wrapped
                out = np.empty(samples_needed, dtype=np.float32)
                if self._write_pos >= samples_needed:
                    out[:] = self._buffer[self._write_pos - samples_needed : self._write_pos]
                else:
                    part2_len = self._write_pos
                    part1_len = samples_needed - part2_len
                    out[:part1_len] = self._buffer[self.buffer_capacity - part1_len :]
                    out[part1_len:] = self._buffer[:part2_len]
                return out

    def get_audio_slice(self, sample_count: int) -> np.ndarray:
        """Retrieves exactly `sample_count` recent samples."""
        duration = sample_count / self.target_rate
        return self.get_last_n_seconds(duration)
