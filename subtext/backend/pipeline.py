from __future__ import annotations

import asyncio
import logging
import statistics
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from gemini import GeminiError, GeminiInterpreter
from schemas import (
    AlignedStep,
    AlignedSequence,
    AudioCue,
    LLMInterpretation,
    TranscriptCue,
    VisualCue,
    VisualSubjectCue,
)


EventHandler = Callable[[dict[str, Any]], Awaitable[None]]

_gemini_logger = logging.getLogger("subtext.gemini")
if not _gemini_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [Gemini] %(message)s", "%H:%M:%S"))
    _gemini_logger.addHandler(_handler)
    _gemini_logger.propagate = False
_gemini_logger.setLevel(logging.INFO)


class ConversationPipeline:
    """Collect in-memory cues, align them, and request occasional Gemini analysis."""

    window_seconds = 14.0
    analysis_spacing_seconds = 9.0
    maximum_cues_per_kind = 600

    def __init__(self) -> None:
        self.interpreter = GeminiInterpreter()
        self.session_id = ""
        self.started_at_epoch = 0.0
        self._transcript: deque[TranscriptCue] = deque(maxlen=self.maximum_cues_per_kind)
        self._audio: deque[AudioCue] = deque(maxlen=self.maximum_cues_per_kind)
        self._visual: deque[VisualCue] = deque(maxlen=self.maximum_cues_per_kind)
        self._visual_baselines: dict[str, deque[dict[str, float]]] = {}
        self._next_analysis_at = 8.0
        self._analysis_in_flight = False
        self._generation = 0

    def begin(self, started_at_epoch: float | None = None) -> str:
        self.session_id = str(uuid.uuid4())
        self._generation += 1
        self.started_at_epoch = started_at_epoch or time.time()
        self._transcript.clear()
        self._audio.clear()
        self._visual.clear()
        self._visual_baselines.clear()
        self._next_analysis_at = 8.0
        self._analysis_in_flight = False
        return self.session_id

    def clear(self) -> None:
        self._generation += 1
        self._analysis_in_flight = False
        self._transcript.clear()
        self._audio.clear()
        self._visual.clear()
        self._visual_baselines.clear()
        self._next_analysis_at = 8.0

    @property
    def cue_counts(self) -> dict[str, int]:
        return {
            "transcript": len(self._transcript),
            "audio": len(self._audio),
            "visual": len(self._visual),
        }

    def add_transcript_event(self, event: dict[str, Any]) -> None:
        cue = TranscriptCue.model_validate(
            {
                "id": event["id"],
                "speaker_id": event["speaker_id"],
                "start_s": event["start_s"],
                "end_s": event["end_s"],
                "text": event["text"],
                "confidence": event.get("confidence"),
            }
        )
        self._transcript.append(cue)
        self._trim_old_cues(cue.end_s)

    def add_audio_event(self, event: dict[str, Any]) -> None:
        self._audio.append(AudioCue.model_validate(event["cue"]))
        self._trim_old_cues(self._audio[-1].end_s)

    def add_visual_frame(self, message: dict[str, Any]) -> None:
        timestamp = float(message.get("timestamp", time.time()))
        relative_s = max(0.0, timestamp - self.started_at_epoch)
        subjects: list[VisualSubjectCue] = []

        for raw in message.get("subjects", []):
            track_id = str(raw.get("track_id", "window_face_1"))
            measured = {
                key: float(raw[key])
                for key in (
                    "head_yaw",
                    "head_pitch",
                    "head_roll",
                    "mouth_aperture",
                    "left_eye_aperture",
                    "right_eye_aperture",
                )
                if raw.get(key) is not None
            }
            baseline = self._visual_baselines.setdefault(track_id, deque(maxlen=40))
            changes: dict[str, float] = {}
            if len(baseline) >= 3:
                for key, value in measured.items():
                    prior_values = [sample[key] for sample in baseline if key in sample]
                    if prior_values:
                        changes[key] = value - statistics.median(prior_values)
            if measured:
                baseline.append(measured)

            subjects.append(
                VisualSubjectCue.model_validate(
                    {
                        **raw,
                        "track_id": track_id,
                        "face_present": True,
                        "change_from_baseline": changes,
                    }
                )
            )

        cue = VisualCue.model_validate(
            {
                "id": str(message.get("id") or f"visual-{uuid.uuid4()}"),
                "start_s": relative_s,
                "end_s": relative_s,
                "subjects": subjects,
                "quality": {
                    "source": "selected_call_window",
                    "frame_processed_locally": True,
                    "face_count": len(subjects),
                    **message.get("quality", {}),
                },
            }
        )
        self._visual.append(cue)
        self._trim_old_cues(relative_s)

    async def maybe_analyze(self, publish: EventHandler) -> None:
        if not self.interpreter.configured or self._analysis_in_flight:
            return
        if not self._transcript or not self._audio or not self._visual:
            return

        current_s = max(
            self._transcript[-1].end_s,
            self._audio[-1].end_s,
            self._visual[-1].end_s,
        )
        if current_s < self._next_analysis_at:
            return

        start_s = max(0.0, current_s - self.window_seconds)
        transcript = sorted(
            (cue for cue in self._transcript if cue.end_s >= start_s and cue.start_s <= current_s),
            key=lambda cue: cue.start_s,
        )
        audio = sorted(
            (cue for cue in self._audio if cue.end_s >= start_s and cue.start_s <= current_s),
            key=lambda cue: cue.start_s,
        )
        visual = sorted(
            (cue for cue in self._visual if cue.end_s >= start_s and cue.start_s <= current_s),
            key=lambda cue: cue.start_s,
        )
        timeline: list[AlignedStep] = []
        step_start = start_s
        while step_start < current_s:
            step_end = min(current_s, step_start + 1.0)
            timeline.append(
                AlignedStep(
                    start_s=step_start,
                    end_s=step_end,
                    transcript_ids=[
                        cue.id for cue in transcript
                        if cue.start_s < step_end and cue.end_s >= step_start
                    ],
                    audio_ids=[
                        cue.id for cue in audio
                        if cue.start_s < step_end and cue.end_s >= step_start
                    ],
                    visual_ids=[
                        cue.id for cue in visual
                        if step_start <= cue.start_s < step_end
                    ],
                )
            )
            step_start = step_end

        sequence = AlignedSequence(
            session_id=self.session_id,
            start_s=start_s,
            end_s=current_s,
            transcript=transcript,
            audio=audio,
            visual=visual,
            timeline=timeline,
        )
        if not sequence.transcript or not sequence.audio or not sequence.visual:
            return

        self._next_analysis_at = current_s + self.analysis_spacing_seconds
        self._analysis_in_flight = True
        _gemini_logger.info(
            "sending to %s (%.1fs window; transcript=%d, audio=%d, visual=%d cues)",
            self.interpreter.model,
            sequence.end_s - sequence.start_s,
            len(sequence.transcript),
            len(sequence.audio),
            len(sequence.visual),
        )
        asyncio.create_task(self._interpret_and_publish(sequence, publish, self._generation))

    async def _interpret_and_publish(
        self,
        sequence: AlignedSequence,
        publish: EventHandler,
        generation: int,
    ) -> None:
        request_started = time.monotonic()
        try:
            result: LLMInterpretation = await self.interpreter.interpret(sequence)
        except GeminiError as exc:
            if generation != self._generation:
                return
            _gemini_logger.error("request failed on %s: %s", self.interpreter.model, exc)
            await publish(
                {
                    "type": "llm_status",
                    "state": "error",
                    "detail": str(exc),
                }
            )
        except Exception:
            if generation != self._generation:
                return
            _gemini_logger.exception(
                "unexpected analysis failure on %s", self.interpreter.model
            )
            await publish(
                {
                    "type": "llm_status",
                    "state": "error",
                    "detail": "Subtext could not interpret this conversation window.",
                }
            )
        else:
            if generation != self._generation:
                return
            _gemini_logger.info(
                "response from %s in %.2fs:\n%s",
                self.interpreter.model,
                time.monotonic() - request_started,
                result.model_dump_json(exclude_none=True, indent=2),
            )
            await publish(
                {
                    "type": "llm_analysis",
                    "id": str(uuid.uuid4()),
                    "timestamp": time.time(),
                    "start_s": sequence.start_s,
                    "end_s": sequence.end_s,
                    "analysis": result.model_dump(exclude_none=True),
                }
            )
        finally:
            if generation == self._generation:
                self._analysis_in_flight = False

    def _trim_old_cues(self, current_s: float) -> None:
        cutoff = max(0.0, current_s - 90.0)
        for collection in (self._transcript, self._audio, self._visual):
            while collection and collection[0].end_s < cutoff:
                collection.popleft()
