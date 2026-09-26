from __future__ import annotations

import asyncio
import copy
import logging
import math
import os
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from schemas import AudioCue, SpeechSegment


EventHandler = Callable[[dict[str, Any]], Awaitable[None]]
logger = logging.getLogger("subtext.transcriber")


class AudioTranscriber:
    """Transcribe microphone and selected-call-window audio locally."""

    sample_rate = 16_000
    block_size = 1_600
    # Require a clear voice-level signal before asking Whisper to decode a clip.
    # The previous microphone threshold admitted room noise and input hiss.
    microphone_speech_threshold = 0.012
    call_speech_threshold = 0.012
    silence_seconds = 0.70
    max_utterance_seconds = 12.0
    min_utterance_seconds = 0.40

    def __init__(self) -> None:
        self.state = "idle"
        self.model_loaded = False
        self.session_start_epoch = 0.0
        self._model: Any = None
        self._numpy: Any = None
        self._frames: queue.Queue[Any] = queue.Queue(maxsize=600)
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._event_handler: EventHandler | None = None
        self._last_turn: tuple[str, float] | None = None
        self._feature_baselines: dict[str, deque[dict[str, float]]] = {}
        self._context_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._source_lock = threading.Lock()
        self._microphone_enabled = True
        self._microphone_generation = 0
        self._metrics: dict[str, Any] = {
            "audio_packets_received": {"self": 0, "other": 0},
            "audio_packets_enqueued": {"self": 0, "other": 0},
            "audio_packets_dequeued": 0,
            "audio_packets_rejected_worker_unavailable": 0,
            "audio_packets_rejected_microphone_muted": 0,
            "empty_audio_packets": 0,
            "queue_dropped_packets": 0,
            "frames_processed": {"self": 0, "other": 0},
            "speech_frames": {"self": 0, "other": 0},
            "quiet_frames": {"self": 0, "other": 0},
            "utterances_started": {"self": 0, "other": 0},
            "utterances_too_short": 0,
            "transcriptions_started": 0,
            "transcriptions_completed": 0,
            "transcriptions_empty": 0,
            "transcription_errors": 0,
            "transcription_in_flight": False,
            "last_packet_at": {},
            "last_rms": {},
            "last_transcription_duration_s": None,
            "last_transcription_error": None,
        }

    def diagnostics_snapshot(self) -> dict[str, Any]:
        with self._metrics_lock:
            metrics = copy.deepcopy(self._metrics)
        return {
            **metrics,
            "state": self.state,
            "model_loaded": self.model_loaded,
            "worker_alive": bool(self._worker and self._worker.is_alive()),
            "queue_size": self._frames.qsize(),
            "queue_capacity": self._frames.maxsize,
        }

    def clear_context(self) -> None:
        with self._context_lock:
            self._last_turn = None
            self._feature_baselines.clear()

    async def start(
        self,
        event_handler: EventHandler,
        session_start_epoch: float | None = None,
    ) -> None:
        if self._worker is not None and self._worker.is_alive():
            return

        self._loop = asyncio.get_running_loop()
        self._event_handler = event_handler
        self.session_start_epoch = session_start_epoch or time.time()
        with self._source_lock:
            self._microphone_enabled = True
            self._microphone_generation += 1
        self._set_state("loading_model")
        logger.info("Starting Whisper; model=%s", os.environ.get("SUBTEXT_WHISPER_MODEL", "base.en"))
        try:
            await asyncio.to_thread(self._start_sync)
        except Exception as exc:
            self._set_state("error", detail=str(exc))
            raise

    def _start_sync(self) -> None:
        try:
            import numpy as np
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Transcription packages are missing. Run scripts/run.sh to install them."
            ) from exc

        self._numpy = np
        if self._model is None:
            model_name = os.environ.get("SUBTEXT_WHISPER_MODEL", "base.en")
            thread_count = max(1, min(4, os.cpu_count() or 2))
            logger.info("Loading Whisper model on CPU; model=%s cpu_threads=%d", model_name, thread_count)
            self._model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
                cpu_threads=thread_count,
            )
            self.model_loaded = True
            logger.info("Whisper model loaded; model=%s", model_name)

        self._stop_event.clear()
        self._frames = queue.Queue(maxsize=600)
        self.clear_context()
        self._worker = threading.Thread(
            target=self._consume_audio,
            name="subtext-transcriber",
            daemon=True,
        )
        self._worker.start()
        logger.info("Audio worker started; queue_capacity=%d", self._frames.maxsize)
        self._set_state("listening")

    def enqueue_call_audio(self, samples: Any, captured_at_epoch: float) -> None:
        """Queue one 16 kHz mono float32 call-audio packet from ScreenCaptureKit."""
        self.enqueue_audio("other", samples, captured_at_epoch)

    def enqueue_microphone_audio(self, samples: Any, captured_at_epoch: float) -> None:
        """Queue one 16 kHz mono float32 microphone packet from the macOS app."""
        self.enqueue_audio("self", samples, captured_at_epoch)

    def set_microphone_enabled(self, enabled: bool) -> None:
        """Gate local-microphone audio without affecting call-window audio."""
        with self._source_lock:
            if self._microphone_enabled == enabled:
                return
            self._microphone_enabled = enabled
            self._microphone_generation += 1
            generation = self._microphone_generation
        logger.info(
            "Microphone transcription %s; source_generation=%d",
            "enabled" if enabled else "muted",
            generation,
        )

    def enqueue_audio(self, speaker_id: str, samples: Any, captured_at_epoch: float) -> None:
        with self._metrics_lock:
            self._metrics["audio_packets_received"][speaker_id] += 1
            self._metrics["last_packet_at"][speaker_id] = time.time()
        generation = 0
        if speaker_id == "self":
            with self._source_lock:
                if not self._microphone_enabled:
                    with self._metrics_lock:
                        self._metrics["audio_packets_rejected_microphone_muted"] += 1
                    return
                generation = self._microphone_generation
        if self._worker is None or not self._worker.is_alive() or self._numpy is None:
            with self._metrics_lock:
                self._metrics["audio_packets_rejected_worker_unavailable"] += 1
            return
        audio = self._numpy.asarray(samples, dtype=self._numpy.float32).reshape(-1)
        if audio.size:
            self._enqueue(speaker_id, captured_at_epoch, audio, generation)
        else:
            with self._metrics_lock:
                self._metrics["empty_audio_packets"] += 1

    def _enqueue(
        self,
        speaker_id: str,
        captured_at_epoch: float,
        audio: Any,
        generation: int,
    ) -> None:
        try:
            self._frames.put_nowait((speaker_id, captured_at_epoch, audio, generation))
            with self._metrics_lock:
                self._metrics["audio_packets_enqueued"][speaker_id] += 1
        except queue.Full:
            try:
                self._frames.get_nowait()
                self._frames.put_nowait((speaker_id, captured_at_epoch, audio, generation))
                with self._metrics_lock:
                    self._metrics["queue_dropped_packets"] += 1
                    dropped = self._metrics["queue_dropped_packets"]
                    self._metrics["audio_packets_enqueued"][speaker_id] += 1
                if dropped == 1 or dropped % 50 == 0:
                    logger.warning("Audio queue full; dropped_oldest_total=%d queue_size=%d", dropped, self._frames.qsize())
            except queue.Empty:
                pass

    def _consume_audio(self) -> None:
        states: dict[str, dict[str, Any]] = {}
        while not self._stop_event.is_set() or not self._frames.empty():
            try:
                speaker_id, captured_at, audio, generation = self._frames.get(timeout=0.1)
            except queue.Empty:
                self._flush_silent_sources(states)
                continue

            with self._metrics_lock:
                self._metrics["audio_packets_dequeued"] += 1

            if speaker_id == "self" and not self._microphone_packet_is_current(generation):
                self._discard_active_utterance(states.get("self"))
                continue

            source_state = states.get(speaker_id)
            if (
                source_state is not None
                and source_state["active"]
                and captured_at - source_state.get("last_packet_end_epoch", captured_at) > 0.25
            ):
                self._flush_utterance(speaker_id, source_state)

            offset = 0
            while offset < len(audio):
                frame = audio[offset : offset + self.block_size]
                frame_start = captured_at + offset / self.sample_rate
                self._process_frame(states, speaker_id, frame_start, frame, generation)
                offset += len(frame)
            state = states.setdefault(
                speaker_id,
                {"chunks": [], "active": False, "start_epoch": 0.0, "last_voice_epoch": 0.0},
            )
            state["last_packet_end_epoch"] = captured_at + len(audio) / self.sample_rate

        for speaker_id, state in list(states.items()):
            if state["active"]:
                self._flush_utterance(speaker_id, state)

    def _process_frame(
        self,
        states: dict[str, dict[str, Any]],
        speaker_id: str,
        frame_start_epoch: float,
        frame: Any,
        generation: int,
    ) -> None:
        state = states.setdefault(
            speaker_id,
            {
                "chunks": [],
                "active": False,
                "start_epoch": 0.0,
                "last_voice_epoch": 0.0,
                "source_generation": 0,
            },
        )
        rms = float(self._numpy.sqrt(self._numpy.mean(frame * frame))) if len(frame) else 0.0
        frame_end_epoch = frame_start_epoch + len(frame) / self.sample_rate
        threshold = self._speech_threshold_for(speaker_id)
        is_voice = rms >= threshold
        with self._metrics_lock:
            self._metrics["frames_processed"][speaker_id] += 1
            self._metrics["last_rms"][speaker_id] = round(rms, 6)
            self._metrics["speech_frames" if is_voice else "quiet_frames"][speaker_id] += 1

        if is_voice:
            if not state["active"]:
                state["chunks"] = []
                state["start_epoch"] = frame_start_epoch
                state["active"] = True
                state["source_generation"] = generation
                with self._metrics_lock:
                    self._metrics["utterances_started"][speaker_id] += 1
                logger.info(
                    "Voice activity started; source=%s rms=%.5f threshold=%.5f",
                    speaker_id,
                    rms,
                    threshold,
                )
            state["chunks"].append(frame)
            state["last_voice_epoch"] = frame_end_epoch
        elif state["active"]:
            state["chunks"].append(frame)
            if frame_end_epoch - state["last_voice_epoch"] >= self.silence_seconds:
                self._flush_utterance(speaker_id, state)

        if state["active"]:
            duration = sum(len(chunk) for chunk in state["chunks"]) / self.sample_rate
            if duration >= self.max_utterance_seconds:
                self._flush_utterance(speaker_id, state)

    def _flush_silent_sources(self, states: dict[str, dict[str, Any]]) -> None:
        now = time.time()
        for speaker_id, state in list(states.items()):
            if speaker_id == "self" and not self._microphone_is_enabled():
                self._discard_active_utterance(state)
                continue
            if state["active"] and now - state["last_voice_epoch"] >= self.silence_seconds:
                self._flush_utterance(speaker_id, state)

    def _microphone_is_enabled(self) -> bool:
        with self._source_lock:
            return self._microphone_enabled

    def _microphone_packet_is_current(self, generation: int) -> bool:
        with self._source_lock:
            return self._microphone_enabled and generation == self._microphone_generation

    @staticmethod
    def _discard_active_utterance(state: dict[str, Any] | None) -> None:
        if state is not None:
            state["chunks"] = []
            state["active"] = False

    def _flush_utterance(self, speaker_id: str, state: dict[str, Any]) -> None:
        if not state["active"] or not state["chunks"]:
            return
        chunks = state["chunks"]
        start_epoch = float(state["start_epoch"])
        end_epoch = float(state["last_voice_epoch"])
        generation = int(state.get("source_generation", 0))
        state["chunks"] = []
        state["active"] = False

        if end_epoch - start_epoch < self.min_utterance_seconds:
            with self._metrics_lock:
                self._metrics["utterances_too_short"] += 1
            return
        logger.info(
            "Voice activity ended; source=%s duration_s=%.2f queued_audio_packets=%d",
            speaker_id,
            end_epoch - start_epoch,
            self._frames.qsize(),
        )
        self._transcribe(speaker_id, chunks, start_epoch, end_epoch, generation)

    def _transcribe(
        self,
        speaker_id: str,
        chunks: list[Any],
        start_epoch: float,
        end_epoch: float,
        generation: int,
    ) -> None:
        if not chunks:
            return

        self._set_state("transcribing")
        audio = self._numpy.concatenate(chunks).astype(self._numpy.float32, copy=False)
        started = time.monotonic()
        with self._metrics_lock:
            self._metrics["transcriptions_started"] += 1
            self._metrics["transcription_in_flight"] = True
            transcription_number = self._metrics["transcriptions_started"]
        logger.info(
            "Whisper transcription started; number=%d source=%s audio_s=%.2f samples=%d queue_size=%d",
            transcription_number,
            speaker_id,
            len(audio) / self.sample_rate,
            len(audio),
            self._frames.qsize(),
        )
        try:
            segments, _ = self._model.transcribe(
                audio,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.55,
                    "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": 300,
                    "speech_pad_ms": 120,
                },
                condition_on_previous_text=False,
                no_speech_threshold=0.45,
                log_prob_threshold=-0.65,
                compression_ratio_threshold=2.2,
            )
            accepted_segments = []
            rejected_segments = 0
            for segment in segments:
                segment_text = segment.text.strip()
                no_speech_probability = getattr(segment, "no_speech_prob", 1.0)
                average_log_probability = getattr(segment, "avg_logprob", float("-inf"))
                compression_ratio = getattr(segment, "compression_ratio", float("inf"))
                if (
                    segment_text
                    and no_speech_probability <= 0.45
                    and average_log_probability >= -0.65
                    and compression_ratio <= 2.2
                ):
                    accepted_segments.append(segment_text)
                else:
                    rejected_segments += 1
            text = " ".join(accepted_segments).strip()
            elapsed = time.monotonic() - started
            with self._metrics_lock:
                self._metrics["transcriptions_completed"] += 1
                self._metrics["transcriptions_empty"] += int(not text)
                self._metrics["transcription_in_flight"] = False
                self._metrics["last_transcription_duration_s"] = round(elapsed, 3)
                self._metrics["last_transcription_error"] = None
            logger.info(
                "Whisper transcription completed; number=%d source=%s elapsed_s=%.2f text_chars=%d empty=%s",
                transcription_number,
                speaker_id,
                elapsed,
                len(text),
                not bool(text),
            )
            if speaker_id == "self" and not self._microphone_packet_is_current(generation):
                logger.info(
                    "Whisper output suppressed because microphone was muted during decoding; number=%d",
                    transcription_number,
                )
                if not self._stop_event.is_set():
                    self._set_state("listening")
                return
            if not text:
                logger.info(
                    "Whisper output suppressed; number=%d source=%s rejected_segments=%d",
                    transcription_number,
                    speaker_id,
                    rejected_segments,
                )
                if not self._stop_event.is_set():
                    self._set_state("listening")
                return
            cue_id = str(uuid.uuid4())
            start_s = max(0.0, start_epoch - self.session_start_epoch)
            end_s = max(start_s, end_epoch - self.session_start_epoch)
            audio_cue = self._make_audio_cue(
                cue_id,
                speaker_id,
                text,
                audio,
                start_s,
                end_s,
            )
            self._emit({"type": "audio_cue", "cue": audio_cue.model_dump(exclude_none=True)})
            self._emit(
                {
                    "type": "transcript",
                    "id": cue_id,
                    "timestamp": end_epoch,
                    "speaker_id": speaker_id,
                    "start_s": start_s,
                    "end_s": end_s,
                    "text": text,
                }
            )
        except Exception as exc:
            elapsed = time.monotonic() - started
            with self._metrics_lock:
                self._metrics["transcription_errors"] += 1
                self._metrics["transcription_in_flight"] = False
                self._metrics["last_transcription_duration_s"] = round(elapsed, 3)
                self._metrics["last_transcription_error"] = str(exc)
            logger.exception(
                "Whisper transcription failed; number=%d source=%s elapsed_s=%.2f",
                transcription_number,
                speaker_id,
                elapsed,
            )
            self._set_state("error", detail=f"Whisper could not transcribe this phrase: {exc}")
            return

        if not self._stop_event.is_set():
            self._set_state("listening")

    def _make_audio_cue(
        self,
        transcript_id: str,
        speaker_id: str,
        text: str,
        audio: Any,
        start_s: float,
        end_s: float,
    ) -> AudioCue:
        duration = max(0.0, end_s - start_s)
        usable_count = min(len(audio), max(0, int(duration * self.sample_rate)))
        samples = audio[:usable_count] if usable_count else audio
        if samples.size == 0:
            samples = audio

        block = self.block_size
        count = samples.size // block
        block_rms: Any = self.numpy_empty()
        if count:
            blocks = samples[: count * block].reshape(count, block)
            block_rms = self._numpy.sqrt(self._numpy.mean(blocks * blocks, axis=1))
        active_mask = (
            block_rms >= self._speech_threshold_for(speaker_id)
            if count
            else self._numpy.array([], dtype=bool)
        )
        active_indexes = self._numpy.flatnonzero(active_mask) if count else []
        segment_specs: list[tuple[int, int]] = []
        if len(active_indexes):
            begin = previous = int(active_indexes[0])
            for raw_index in active_indexes[1:]:
                index = int(raw_index)
                if index - previous > 2:
                    segment_specs.append((begin, previous + 1))
                    begin = index
                previous = index
            segment_specs.append((begin, previous + 1))

        speech_segments = [
            SpeechSegment(
                start_s=min(end_s, start_s + first * block / self.sample_rate),
                end_s=min(end_s, start_s + last * block / self.sample_rate),
            )
            for first, last in segment_specs
            if start_s + first * block / self.sample_rate < end_s
        ]
        active_seconds = sum(segment.end_s - segment.start_s for segment in speech_segments)
        pause_ms = max(0.0, duration - active_seconds) * 1000.0
        pitch_values: list[float] = []
        frame_size = int(self.sample_rate * 0.04)
        frame_step = int(self.sample_rate * 0.02)
        min_lag = max(1, int(self.sample_rate / 400))
        max_lag = int(self.sample_rate / 80)
        for frame_start in range(0, max(0, samples.size - frame_size + 1), frame_step):
            frame = samples[frame_start : frame_start + frame_size]
            frame = frame - self._numpy.mean(frame)
            if float(self._numpy.sqrt(self._numpy.mean(frame * frame))) < 0.006:
                continue
            correlation = self._numpy.correlate(frame, frame, mode="full")[frame_size - 1 :]
            upper = min(max_lag, correlation.size - 1)
            if upper <= min_lag or correlation[0] <= 0:
                continue
            section = correlation[min_lag : upper + 1]
            lag = min_lag + int(self._numpy.argmax(section))
            if float(correlation[lag] / correlation[0]) >= 0.35:
                pitch_values.append(self.sample_rate / lag)

        loudness_dbfs: float | None = None
        if len(active_indexes):
            rms_values = block_rms[active_mask]
            loudness_dbfs = float(20.0 * self._numpy.log10(float(self._numpy.median(rms_values)) + 1e-8))
        pitch_median = float(self._numpy.median(pitch_values)) if pitch_values else None
        pitch_range: float | None = None
        if len(pitch_values) >= 2:
            pitch_range = float(
                12.0
                * self._numpy.log2(
                    max(float(self._numpy.percentile(pitch_values, 90)), 1.0)
                    / max(float(self._numpy.percentile(pitch_values, 10)), 1.0)
                )
            )

        words_per_minute = None
        word_count = len(text.split())
        if active_seconds > 0 and word_count:
            words_per_minute = word_count / active_seconds * 60.0

        features: dict[str, float] = {}
        if pitch_median is not None:
            features["pitch_median_hz"] = pitch_median
        if loudness_dbfs is not None:
            features["loudness_dbfs"] = loudness_dbfs
        if words_per_minute is not None:
            features["speaking_rate_wpm"] = words_per_minute

        with self._context_lock:
            previous_turn = self._last_turn
            raw_gap_ms = (
                (start_s - previous_turn[1]) * 1000.0
                if previous_turn is not None and previous_turn[0] != speaker_id
                else None
            )
            baseline = self._feature_baselines.setdefault(speaker_id, deque(maxlen=12))
            changes: dict[str, float] = {}
            if len(baseline) >= 3:
                for key, value in features.items():
                    prior = [entry[key] for entry in baseline if key in entry]
                    if not prior:
                        continue
                    prior_median = float(self._numpy.median(prior))
                    if key == "pitch_median_hz" and value > 0 and prior_median > 0:
                        changes["pitch_semitones"] = 12.0 * math.log2(value / prior_median)
                    else:
                        changes[f"{key}_delta"] = value - prior_median
            if features:
                baseline.append(features)
            self._last_turn = (speaker_id, end_s)
        return AudioCue(
            id=f"audio-{transcript_id}",
            speaker_id=speaker_id,
            start_s=start_s,
            end_s=end_s,
            speech_segments=speech_segments,
            response_gap_ms=max(0.0, raw_gap_ms) if raw_gap_ms is not None else None,
            pause_ms=pause_ms,
            overlap_ms=max(0.0, -raw_gap_ms) if raw_gap_ms is not None else None,
            pitch_median_hz=pitch_median,
            pitch_range_semitones=pitch_range,
            loudness_dbfs=loudness_dbfs,
            speaking_rate_wpm=words_per_minute,
            change_from_baseline=changes,
            vocal_events=[],
            quality={
                "source": "selected_microphone" if speaker_id == "self" else "call_window_audio",
                "voice_activity": "energy_threshold",
                "prosody_is_approximate": True,
                "pitch_available": bool(pitch_values),
            },
        )

    def numpy_empty(self) -> Any:
        return self._numpy.array([], dtype=self._numpy.float32)

    def _speech_threshold_for(self, speaker_id: str) -> float:
        return (
            self.microphone_speech_threshold
            if speaker_id == "self"
            else self.call_speech_threshold
        )

    async def stop(self) -> None:
        worker = self._worker
        if worker is None or not worker.is_alive():
            self._set_state("idle")
            return

        self._stop_event.set()
        await asyncio.to_thread(worker.join, 30)
        if worker.is_alive():
            logger.error(
                "Audio worker did not stop within 30 seconds; queue_size=%d diagnostics=%s",
                self._frames.qsize(),
                self.diagnostics_snapshot(),
            )
            self._set_state("error", detail="The current phrase is still being transcribed.")
            return

        self._worker = None
        logger.info("Audio worker stopped cleanly")
        self._set_state("idle")

    def _set_state(self, state: str, detail: str | None = None) -> None:
        previous = self.state
        self.state = state
        if detail:
            logger.warning("Transcriber state changed; %s -> %s detail=%s", previous, state, detail)
        elif previous != state:
            logger.info("Transcriber state changed; %s -> %s", previous, state)
        event: dict[str, Any] = {"type": "status", "state": state}
        if detail:
            event["detail"] = detail
        self._emit(event)

    def _emit(self, event: dict[str, Any]) -> None:
        if self._loop is None or self._event_handler is None or self._loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self._event_handler(event), self._loop)
