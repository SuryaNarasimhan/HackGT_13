"""NDJSON bridge between the Electron capture renderer and the Python AI pipeline.

Input and output are session-only messages on stdin/stdout. Logs stay on stderr so
they can never corrupt the message stream.
"""

from __future__ import annotations

import base64
from collections import deque
import json
import logging
import sys
import threading
import time
from typing import Callable, Optional

import numpy as np

from app.capture.vad_detector import VADDetector
from app.coordinator import PipelineCoordinator
from app.pipelines.taxonomy import CANONICAL_EMOTIONS

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("MSAS.ElectronBridge")
MAX_AUDIO_BYTES = 64 * 1024
MAX_FRAME_BYTES = 700 * 1024


class InputAudioCapture:
    """Capture-compatible adapter fed by Electron instead of a physical device."""

    def __init__(self):
        self.callbacks: list[Callable[[np.ndarray], None]] = []
        self.running = False

    def register_chunk_callback(self, callback):
        self.callbacks.append(callback)

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def push(self, samples: np.ndarray):
        if not self.running:
            return
        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
        for start in range(0, len(chunk), 512):
            part = chunk[start:start + 512]
            if len(part) < 512:
                part = np.pad(part, (0, 512 - len(part)))
            for callback in self.callbacks:
                callback(part)


class InputScreenCapture:
    """Timestamped frame buffer compatible with PipelineCoordinator."""

    def __init__(self, max_frames: int = 30):
        self.frames = deque(maxlen=max_frames)
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False
        with self.lock:
            self.frames.clear()

    def push_frame(self, frame: np.ndarray, timestamp: Optional[float] = None):
        if self.running and frame is not None and frame.size:
            with self.lock:
                self.frames.append((timestamp or time.time(), frame))

    def get_keyframes_in_window(self, start_time, end_time, sample_count=4):
        with self.lock:
            candidates = [frame for ts, frame in self.frames if start_time <= ts <= end_time]
            if not candidates:
                candidates = [frame for _, frame in list(self.frames)[-sample_count:]]
        if not candidates:
            return []
        indices = np.linspace(0, len(candidates) - 1, min(sample_count, len(candidates)), dtype=int)
        return [candidates[index].copy() for index in indices]


class NullCapture(InputAudioCapture):
    """Prevents PipelineCoordinator from opening the laptop microphone."""


class LocalEnergyVAD(VADDetector):
    """Energy-only VAD that never initializes or downloads a Silero model."""

    def _init_silero(self):
        self._silero_model = None


def probability_map(vector) -> dict[str, float]:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    return {name: round(float(values[index]), 4) for index, name in enumerate(CANONICAL_EMOTIONS)}


def result_payload(result: dict, gemini_enabled: bool) -> dict:
    cue = dict(result.get("cue_data") or {})
    return {
        "type": "result",
        "transcript": str(result.get("transcript") or ""),
        "channels": {
            "face": probability_map(result.get("p_video", np.ones(7) / 7)),
            "tone": probability_map(result.get("p_audio", np.ones(7) / 7)),
            "words": probability_map(result.get("p_semantic", np.ones(7) / 7)),
        },
        "channelStatus": dict(result.get("channel_status") or {}),
        "jsd": round(float(result.get("jsd_score", 0.0)), 4),
        "triggered": bool(result.get("is_trigger", False)),
        "cue": {
            "type": str(cue.get("social_cue_type") or "Ambiguous"),
            "confidence": str(cue.get("confidence") or "Low"),
            "explanation": str(cue.get("explanation") or "Not enough evidence yet."),
            "suggestion": str(cue.get("suggested_action") or "Keep listening for context."),
            "source": "Gemini enabled" if gemini_enabled else "Local fallback",
        },
    }


def decode_frame(encoded: str) -> Optional[np.ndarray]:
    raw = base64.b64decode(encoded, validate=True)
    if not raw or len(raw) > MAX_FRAME_BYTES:
        raise ValueError("frame size is invalid")
    try:
        import cv2
        decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB) if decoded is not None else None
    except ImportError:
        from io import BytesIO
        from PIL import Image
        return np.asarray(Image.open(BytesIO(raw)).convert("RGB"))


class BridgeRuntime:
    def __init__(self, emit: Callable[[dict], None]):
        self.emit = emit
        self.audio = InputAudioCapture()
        self.screen = InputScreenCapture()
        self.microphone = NullCapture()
        self.coordinator = PipelineCoordinator(
            audio_capture=self.audio,
            screen_capture=self.screen,
            vad_detector=LocalEnergyVAD(force_energy_vad=True),
            user_mic_capture=self.microphone,
            user_vad_detector=LocalEnergyVAD(silence_threshold_ms=450, force_energy_vad=True),
            status_callback=lambda status: emit({"type": "status", "status": status}),
            speech_state_callback=lambda active: emit({"type": "speech", "active": bool(active)}),
            result_callback=self._on_result,
        )
        self.running = False

    def _on_result(self, result):
        self.emit(result_payload(result, self.coordinator.reasoner.client is not None))

    def start(self):
        if not self.running:
            self.running = True
            self.coordinator.start()
            self.emit({"type": "ready", "gemini": self.coordinator.reasoner.client is not None})

    def handle(self, message: dict):
        kind = message.get("type")
        if kind == "start":
            self.start()
        elif kind == "audio" and self.running:
            raw = base64.b64decode(message.get("data", ""), validate=True)
            if not raw or len(raw) > MAX_AUDIO_BYTES or len(raw) % 4:
                raise ValueError("audio size is invalid")
            self.audio.push(np.frombuffer(raw, dtype="<f4"))
        elif kind == "frame" and self.running:
            frame = decode_frame(message.get("data", ""))
            if frame is not None:
                self.screen.push_frame(frame)
        elif kind == "stop":
            self.stop()
        else:
            raise ValueError("unsupported bridge message")

    def stop(self):
        if self.running:
            self.running = False
            self.coordinator.stop()


def main():
    write_lock = threading.Lock()

    def emit(message):
        with write_lock:
            sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
            sys.stdout.flush()

    runtime = None
    try:
        emit({"type": "booting"})
        runtime = BridgeRuntime(emit)
        for line in sys.stdin:
            if len(line) > 1_000_000:
                raise ValueError("bridge message is too large")
            try:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError("message must be an object")
                runtime.handle(message)
            except Exception as exc:
                logger.exception("Bridge message failed")
                emit({"type": "error", "message": f"AI input could not be processed: {str(exc)[:160]}"})
    except Exception as exc:
        logger.exception("AI bridge stopped")
        emit({"type": "error", "message": f"AI backend stopped: {str(exc)[:160]}"})
    finally:
        if runtime is not None:
            runtime.stop()


if __name__ == "__main__":
    main()
