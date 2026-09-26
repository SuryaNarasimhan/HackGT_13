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
from app.capture.mic_capture import UserMicrophoneCapture
from app.capture.screen_capture import ScreenCaptureManager
from app.capture.vad_detector import VADDetector
from app.config import JSD_THRESHOLD
from app.conversation_memory import ConversationMemory
from app.math_engine.jsd import analyze_cross_modal_conflict
from app.pipelines.audio_pipeline import AudioPipeline
from app.pipelines.semantic_pipeline import SemanticPipeline
from app.pipelines.taxonomy import get_top_emotion
from app.pipelines.video_pipeline import VideoPipeline
from app.ui.qt_compat import QT_AVAILABLE, QtCore

logger = logging.getLogger(__name__)


class PipelineCoordinator:
    """
    Central orchestration engine for SocialLens.
    Runs asynchronously, processing speech utterances from both the remote speaker (loopback)
    and the local user (mic), maintaining rolling conversational context, and dispatching
    subtext insights to the HUD.
    """

    def __init__(
        self,
        audio_capture: AudioLoopbackCapture,
        screen_capture: ScreenCaptureManager,
        vad_detector: VADDetector,
        telemetry_callback: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, float], None]] = None,
        cue_callback: Optional[Callable[[Dict[str, str]], None]] = None,
        speech_state_callback: Optional[Callable[[bool], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        user_mic_capture: Optional[UserMicrophoneCapture] = None,
        memory: Optional[ConversationMemory] = None,
        jsd_threshold: float = JSD_THRESHOLD
    ):
        self.audio = audio_capture
        self.screen = screen_capture
        self.vad = vad_detector
        self.telemetry_cb = telemetry_callback
        self.cue_cb = cue_callback
        self.speech_state_cb = speech_state_callback
        self.status_cb = status_callback
        self.jsd_threshold = jsd_threshold

        # Two-Way Conversational Memory & User Mic
        self.memory = memory or ConversationMemory(max_turns=6)
        self.user_mic = user_mic_capture or UserMicrophoneCapture()
        self.user_vad = VADDetector(silence_threshold_ms=450)

        # Perception Pipelines
        self.semantic_pipeline = SemanticPipeline(lazy_load=True)
        self.audio_pipeline = AudioPipeline(lazy_load=True)
        self.video_pipeline = VideoPipeline()
        self.reasoner = GeminiReasoner()

        # Decoupled thread pools to eliminate nested ThreadPoolExecutor starvation deadlocks
        self._orchestration_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="OrchestrationWorker")
        self._perception_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="PerceptionWorker")
        self._executor = self._orchestration_executor  # Backwards compatibility alias
        self._running = False
        self._lock = threading.Lock()

        # Wire up Remote VAD listeners (speaker loopback)
        self.audio.register_chunk_callback(self._on_audio_chunk)
        self.vad.register_utterance_callback(self._on_utterance_complete)
        self.vad.register_speech_state_callback(self._on_speech_state)

        # Wire up Local User VAD listeners (microphone)
        self.user_mic.register_chunk_callback(self._on_user_mic_chunk)
        self.user_vad.register_utterance_callback(self._on_user_utterance_complete)

    def start(self):
        """Starts all capture streams and processing pipeline."""
        if self._running:
            return
        self._running = True
        self.screen.start()
        self.audio.start()
        self.user_mic.start()
        logger.info("SocialLens Pipeline Coordinator started (Two-way Audio Active).")

    def stop(self):
        """Gracefully shuts down all threads, capture devices, and executor pools."""
        self._running = False
        self.audio.stop()
        self.user_mic.stop()
        self.screen.stop()
        self._orchestration_executor.shutdown(wait=False, cancel_futures=True)
        self._perception_executor.shutdown(wait=False, cancel_futures=True)
        logger.info("SocialLens Pipeline Coordinator stopped.")

    def _on_user_mic_chunk(self, chunk: np.ndarray):
        """Receives live 16kHz audio chunks from local user microphone."""
        if self._running:
            self.user_vad.process_chunk(chunk)

    def _on_user_utterance_complete(self, utterance_audio: np.ndarray):
        """Processes user speech completion to update two-way dialogue history."""
        if not self._running or len(utterance_audio) == 0:
            return
        self._orchestration_executor.submit(self._process_user_utterance, utterance_audio)

    def _process_user_utterance(self, utterance_audio: np.ndarray):
        """Transcribes user speech and registers turn in conversation memory."""
        try:
            transcript, p_semantic, _ = self.semantic_pipeline.process(utterance_audio)
            clean_text = transcript.strip()
            if clean_text and not clean_text.startswith("[Speech detected"):
                top_sem, _ = get_top_emotion(p_semantic)
                self.memory.add_turn(
                    speaker="User",
                    text=clean_text,
                    emotion=top_sem
                )
                logger.info(f"Two-Way Dialogue [You]: \"{clean_text}\"")
        except Exception as e:
            logger.error(f"Error processing user utterance: {e}")

    def _on_audio_chunk(self, chunk: np.ndarray):
        """Receives live 16kHz audio chunks and forwards to VAD."""
        if self._running:
            self.vad.process_chunk(chunk)

    def _on_speech_state(self, is_speaking: bool):
        """Triggered on real-time speech onset or pause detection."""
        if not self._running:
            return
        if self.speech_state_cb is not None:
            try:
                self.speech_state_cb(is_speaking)
            except Exception as e:
                logger.error(f"Error in speech state callback: {e}")
        if self.status_cb is not None:
            try:
                self.status_cb("Speaking" if is_speaking else "Listening")
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

    def _on_utterance_complete(self, utterance_audio: np.ndarray):
        """Triggered when speaker pauses (>= 500ms). Submits utterance for analysis."""
        if not self._running or len(utterance_audio) == 0:
            return

        if self.status_cb is not None:
            try:
                self.status_cb("Analyzing")
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

        # Submit to orchestration pool and attach unhandled error logger
        future = self._orchestration_executor.submit(self.process_utterance, utterance_audio)

        def _log_future_done(f):
            exc = f.exception()
            if exc is not None:
                logger.error(f"Unhandled error in process_utterance: {exc}", exc_info=exc)

        future.add_done_callback(_log_future_done)

    def process_utterance(self, utterance_audio: np.ndarray) -> Dict:
        """
        Processes a single complete speech utterance across all 3 channels:
        1. Semantic Pipeline (Speech-to-Text -> Text Emotion)
        2. Audio Prosody Pipeline (Waveform -> Tone Emotion)
        3. Video Pipeline (Temporal keyframes -> Face Emotion)
        4. Math Engine (Multi-distribution JSD & Pairwise Conflict)
        5. Reasoner Synthesis with Two-Way Conversational History Grounding
        Guaranteed to reset status back to 'Listening' in finally block.
        """
        try:
            start_time = time.time()
            duration_sec = len(utterance_audio) / 16000.0

            # Sample aligned keyframes from screen capture buffer
            keyframes = self.screen.get_keyframes_in_window(
                start_time=start_time - duration_sec,
                end_time=start_time,
                sample_count=4
            )

            # 1. Parallel execution across all three modalities in dedicated perception pool
            future_sem = self._perception_executor.submit(self.semantic_pipeline.process, utterance_audio)
            future_aud = self._perception_executor.submit(self.audio_pipeline.process, utterance_audio)
            future_vid = self._perception_executor.submit(self.video_pipeline.process_keyframes, keyframes)

            # Retrieve with modality-level timeouts to prevent any single hung model from freezing the HUD
            try:
                transcript, p_semantic, sem_conf = future_sem.result(timeout=10.0)
            except Exception as e:
                logger.warning(f"Semantic pipeline failed/timed out: {e}")
                transcript, p_semantic, sem_conf = "[Speech detected]", normalize_distribution(np.ones(7)), 0.0

            try:
                p_audio, prosody_features = future_aud.result(timeout=10.0)
            except Exception as e:
                logger.warning(f"Audio pipeline failed/timed out: {e}")
                p_audio, prosody_features = normalize_distribution(np.ones(7)), {}

            try:
                p_video, face_detected = future_vid.result(timeout=10.0)
            except Exception as e:
                logger.warning(f"Video pipeline failed/timed out: {e}")
                p_video, face_detected = normalize_distribution(np.ones(7)), False

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

            # 3. Notify Telemetry Listeners on every VAD detection
            if self.telemetry_cb is not None:
                try:
                    self.telemetry_cb(p_video, p_audio, p_semantic, jsd_score)
                except Exception as e:
                    logger.error(f"Error in telemetry callback: {e}")

            # Retrieve rolling conversation history (including user turns)
            dialogue_history = self.memory.get_formatted_history()

            # 4. Reasoner Synthesis on EVERY VAD Utterance Detection
            cue_data = self.reasoner.synthesize_cue(
                transcript=transcript,
                p_video=p_video,
                p_audio=p_audio,
                p_semantic=p_semantic,
                jsd_score=jsd_score,
                conflict_pair=conflict_data.get("max_conflict_pair"),
                max_conflict_value=float(conflict_data.get("max_conflict_value", 0.0)),
                is_trigger=is_trigger,
                dialogue_history=dialogue_history
            )

            # Record this remote speaker turn into dialogue memory
            top_tone, _ = get_top_emotion(p_audio)
            self.memory.add_turn(
                speaker="Other",
                text=transcript,
                emotion=top_tone,
                jsd_score=jsd_score
            )

            # Notify Cue Card Listeners on every VAD detection
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

        except Exception as e:
            logger.error(f"Error in process_utterance: {e}", exc_info=True)
            return {}

        finally:
            # Guarantees status pill returns to 'Listening' under all circumstances
            if self.status_cb is not None:
                try:
                    self.status_cb("Listening")
                except Exception as e:
                    logger.error(f"Error in status callback: {e}")


if QT_AVAILABLE:

    class QtCoordinatorBridge(QtCore.QObject):
        """
        Bridge mapping coordinator worker outputs to thread-safe PyQt Signals.
        """
        telemetry_signal = QtCore.pyqtSignal(object, object, object, float)
        cue_signal = QtCore.pyqtSignal(dict)
        speech_state_signal = QtCore.pyqtSignal(bool)
        status_signal = QtCore.pyqtSignal(str)

        def emit_telemetry(self, p_v, p_a, p_s, jsd):
            self.telemetry_signal.emit(p_v, p_a, p_s, jsd)

        def emit_cue(self, cue_data):
            self.cue_signal.emit(cue_data)

        def emit_speech_state(self, is_speaking: bool):
            self.speech_state_signal.emit(is_speaking)

        def emit_status(self, status: str):
            self.status_signal.emit(status)
