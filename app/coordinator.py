"""
SocialLens Pipeline Coordinator
Orchestrates audio capture, VAD segmentation, multimodal perception pipelines
(Video, Audio, Semantics), JSD divergence calculation, and Gemini cognitive synthesis.
"""

from concurrent.futures import ThreadPoolExecutor
import logging
import threading
import time
from typing import Callable, Dict, List, Optional
import numpy as np

from app.agent.gemini_reasoner import GeminiReasoner
from app.capture.audio_loopback import AudioLoopbackCapture
from app.capture.screen_capture import ScreenCaptureManager
from app.capture.vad_detector import VADDetector
from app.config import JSD_THRESHOLD
from app.math_engine.jsd import analyze_cross_modal_conflict
from app.pipelines.audio_pipeline import AudioPipeline
from app.pipelines.semantic_pipeline import SemanticPipeline
from app.pipelines.video_pipeline import VideoPipeline
from app.ui.qt_compat import QT_AVAILABLE, QtCore

logger = logging.getLogger(__name__)


class PipelineCoordinator:
    """
    Central orchestration engine for SocialLens.
    Runs asynchronously, processing speech utterances from VAD, executing perception
    pipelines, checking JSD thresholds, and dispatching results to the UI.
    """

    def __init__(
        self,
        audio_capture: AudioLoopbackCapture,
        screen_capture: ScreenCaptureManager,
        vad_detector: VADDetector,
        telemetry_callback: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, float], None]] = None,
        cue_callback: Optional[Callable[[Dict[str, str]], None]] = None,
        jsd_threshold: float = JSD_THRESHOLD
    ):
        self.audio = audio_capture
        self.screen = screen_capture
        self.vad = vad_detector
        self.telemetry_cb = telemetry_callback
        self.cue_cb = cue_callback
        self.jsd_threshold = jsd_threshold

        # Perception Pipelines
        self.semantic_pipeline = SemanticPipeline(lazy_load=True)
        self.audio_pipeline = AudioPipeline(lazy_load=True)
        self.video_pipeline = VideoPipeline()
        self.reasoner = GeminiReasoner()

        # Thread pool for parallel pipeline execution
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="PerceptionWorker")
        self._running = False
        self._lock = threading.Lock()

        # Wire up VAD audio chunk listener
        self.audio.register_chunk_callback(self._on_audio_chunk)
        # Wire up VAD utterance completion listener
        self.vad.register_utterance_callback(self._on_utterance_complete)

    def start(self):
        """Starts all capture streams and processing pipeline."""
        if self._running:
            return
        self._running = True
        self.screen.start()
        self.audio.start()
        logger.info("SocialLens Pipeline Coordinator started.")

    def stop(self):
        """Gracefully shuts down all threads, capture devices, and executor pools."""
        self._running = False
        self.audio.stop()
        self.screen.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info("SocialLens Pipeline Coordinator stopped.")

    def _on_audio_chunk(self, chunk: np.ndarray):
        """Receives live 16kHz audio chunks and forwards to VAD."""
        if self._running:
            self.vad.process_chunk(chunk)

    def _on_utterance_complete(self, utterance_audio: np.ndarray):
        """Triggered when speaker pauses (>= 500ms). Submits utterance for analysis."""
        if not self._running or len(utterance_audio) == 0:
            return

        # Submit to worker pool asynchronously so VAD continues listening
        self._executor.submit(self.process_utterance, utterance_audio)

    def process_utterance(self, utterance_audio: np.ndarray) -> Dict:
        """
        Processes a single complete speech utterance across all 3 channels:
        1. Semantic Pipeline (Speech-to-Text -> Text Emotion)
        2. Audio Prosody Pipeline (Waveform -> Tone Emotion)
        3. Video Pipeline (Temporal keyframes -> Face Emotion)
        4. Math Engine (Multi-distribution JSD & Pairwise Conflict)
        5. Gated Gemini Call (if JSD >= threshold)
        """
        start_time = time.time()
        duration_sec = len(utterance_audio) / 16000.0

        # Sample aligned keyframes from screen capture buffer
        keyframes = self.screen.get_keyframes_in_window(
            start_time=start_time - duration_sec,
            end_time=start_time,
            sample_count=4
        )

        # 1. Parallel execution across all three modalities
        future_sem = self._executor.submit(self.semantic_pipeline.process, utterance_audio)
        future_aud = self._executor.submit(self.audio_pipeline.process, utterance_audio)
        future_vid = self._executor.submit(self.video_pipeline.process_keyframes, keyframes)

        transcript, p_semantic, sem_conf = future_sem.result()
        p_audio, prosody_features = future_aud.result()
        p_video, face_detected = future_vid.result()

        # If transcript is empty, fallback to brief silence handling
        if not transcript.strip():
            transcript = "[Speech detected without clear transcript]"

        # 2. Math Engine: Cross-modal conflict & JSD calculation
        conflict_data = analyze_cross_modal_conflict(
            p_v=p_video,
            p_a=p_audio,
            p_s=p_semantic,
            threshold=self.jsd_threshold
        )
        jsd_score = float(conflict_data["tri_modal_jsd"])
        is_trigger = bool(conflict_data["is_trigger"])

        logger.info(
            f"Utterance Processed | Words: '{transcript[:30]}...' | JSD: {jsd_score:.3f} | Trigger: {is_trigger}"
        )

        # 3. Notify Telemetry Listeners
        if self.telemetry_cb is not None:
            try:
                self.telemetry_cb(p_video, p_audio, p_semantic, jsd_score)
            except Exception as e:
                logger.error(f"Error in telemetry callback: {e}")

        # 4. Gated Gemini Synthesis: Only triggered on significant incongruence
        cue_data = None
        if is_trigger:
            cue_data = self.reasoner.synthesize_cue(
                transcript=transcript,
                p_video=p_video,
                p_audio=p_audio,
                p_semantic=p_semantic,
                jsd_score=jsd_score,
                conflict_pair=conflict_data.get("max_conflict_pair"),
                max_conflict_value=float(conflict_data.get("max_conflict_value", 0.0))
            )

            # Notify Cue Card Listeners
            if self.cue_cb is not None:
                try:
                    self.cue_cb(cue_data)
                except Exception as e:
                    logger.error(f"Error in cue callback: {e}")

        return {
            "transcript": transcript,
            "p_video": p_video,
            "p_audio": p_audio,
            "p_semantic": p_semantic,
            "jsd_score": jsd_score,
            "is_trigger": is_trigger,
            "cue_data": cue_data
        }


if QT_AVAILABLE:

    class QtCoordinatorBridge(QtCore.QObject):
        """
        Bridge mapping coordinator worker outputs to thread-safe PyQt Signals.
        """
        telemetry_signal = QtCore.pyqtSignal(object, object, object, float)
        cue_signal = QtCore.pyqtSignal(dict)

        def emit_telemetry(self, p_v, p_a, p_s, jsd):
            self.telemetry_signal.emit(p_v, p_a, p_s, jsd)

        def emit_cue(self, cue_data):
            self.cue_signal.emit(cue_data)
