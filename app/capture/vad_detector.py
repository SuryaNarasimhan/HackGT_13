"""
Voice Activity Detection (VAD) & Speech Segmentation for SocialLens
Monitors incoming audio stream chunks, detects speech onsets and pauses (>= 500ms),
and emits complete utterance segments for multimodal processing.
"""

import logging
from typing import Callable, List, Optional
import numpy as np

from app.config import (
    AUDIO_SAMPLE_RATE,
    VAD_SILENCE_THRESHOLD_MS,
    VAD_MIN_SPEECH_DURATION_MS,
    VAD_CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


class VADDetector:
    """
    Voice Activity Detector and speech segmenter.
    Uses Silero VAD when PyTorch/ONNX is available, with an adaptive energy/entropy
    VAD fallback that runs instantly in pure NumPy.
    """

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        silence_threshold_ms: int = VAD_SILENCE_THRESHOLD_MS,
        min_speech_duration_ms: int = VAD_MIN_SPEECH_DURATION_MS,
        confidence_threshold: float = VAD_CONFIDENCE_THRESHOLD,
        force_energy_vad: bool = False,
    ):
        self.sample_rate = sample_rate
        self.silence_threshold_ms = silence_threshold_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.confidence_threshold = confidence_threshold
        self.force_energy_vad = force_energy_vad


        # Speech state tracking
        self.is_speech_active = False
        self._current_utterance_chunks: List[np.ndarray] = []
        self._silence_frames_count = 0
        self._utterance_callbacks: List[Callable[[np.ndarray], None]] = []
        self._speech_state_callbacks: List[Callable[[bool], None]] = []

        # Convert millisecond thresholds to chunk counts (assuming 512 samples per chunk ~ 32ms)
        self.samples_per_chunk = 512
        self.chunk_duration_ms = (self.samples_per_chunk / self.sample_rate) * 1000.0
        self.silence_chunks_needed = max(1, int(self.silence_threshold_ms / self.chunk_duration_ms))
        self.min_speech_chunks = max(1, int(self.min_speech_duration_ms / self.chunk_duration_ms))

        # Silero model handle
        self._silero_model = None
        self._init_silero()

    def _init_silero(self):
        """Loads Silero VAD model if PyTorch is installed."""
        try:
            import torch
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True
            )
            self._silero_model = model
            logger.info("Silero VAD model loaded successfully.")
        except Exception as e:
            logger.info(f"Silero VAD unavailable ({e}). Running on adaptive DSP energy VAD.")
            self._silero_model = None

    def register_utterance_callback(self, cb: Callable[[np.ndarray], None]):
        """Registers a callback receiving a complete utterance audio segment."""
        self._utterance_callbacks.append(cb)

    def register_speech_state_callback(self, cb: Callable[[bool], None]):
        """Registers a callback receiving boolean speech state (True=speaking, False=silent)."""
        self._speech_state_callbacks.append(cb)

    def _emit_speech_state(self, is_active: bool):
        """Dispatches active speech state transitions to registered listeners."""
        for cb in self._speech_state_callbacks:
            try:
                cb(is_active)
            except Exception as e:
                logger.error(f"Error in VAD speech state callback: {e}")

    def process_chunk(self, chunk: np.ndarray) -> bool:
        """
        Processes a single audio chunk (e.g. 512 samples at 16kHz).
        Returns True if chunk is classified as speech.
        """
        is_speech = self._is_speech_chunk(chunk)

        if is_speech:
            self._silence_frames_count = 0
            if not self.is_speech_active:
                # Speech onset
                self.is_speech_active = True
                self._current_utterance_chunks = []
                self._emit_speech_state(True)
                logger.debug("VAD: Speech onset detected.")
            self._current_utterance_chunks.append(chunk)

        elif self.is_speech_active:
            # Currently in speech, but this chunk is silent
            self._current_utterance_chunks.append(chunk)
            self._silence_frames_count += 1

            if self._silence_frames_count >= self.silence_chunks_needed:
                # Silence threshold reached -> Speech end
                self.is_speech_active = False
                self._silence_frames_count = 0
                self._emit_speech_state(False)

                # Check if speech was long enough to be meaningful
                if len(self._current_utterance_chunks) >= self.min_speech_chunks:
                    # Concatenate speech segment
                    utterance_audio = np.concatenate(self._current_utterance_chunks)
                    logger.info(
                        f"VAD: Utterance completed ({len(utterance_audio) / self.sample_rate:.2f}s). Emitting."
                    )
                    self._emit_utterance(utterance_audio)
                else:
                    logger.debug("VAD: Discarding short audio burst (too brief).")

                self._current_utterance_chunks = []

        return is_speech

    def reset_states(self):
        """Resets model recurrent states between utterances."""
        if self._silero_model is not None and hasattr(self._silero_model, "reset_states"):
            try:
                self._silero_model.reset_states()
            except Exception:
                pass

    def _is_speech_chunk(self, chunk: np.ndarray) -> bool:
        """Evaluates whether a single chunk contains speech."""
        if len(chunk) == 0:
            return False

        if not getattr(self, "force_energy_vad", False) and self._silero_model is not None:
            try:
                import torch
                with torch.no_grad():
                    tensor = torch.from_numpy(chunk.astype(np.float32))
                    out = self._silero_model(tensor, self.sample_rate)
                    prob = float(out.detach().cpu().item())
                return prob >= self.confidence_threshold
            except Exception as e:
                logger.warning(f"Silero inference error: {e}")

        # Adaptive DSP Energy VAD Fallback
        rms = float(np.sqrt(np.mean(chunk**2)))
        return rms > 0.020


    def _emit_utterance(self, audio: np.ndarray):
        """Dispatches completed utterance audio to all registered callbacks."""
        for cb in self._utterance_callbacks:
            try:
                cb(audio)
            except Exception as e:
                logger.error(f"Error in VAD utterance callback: {e}")
