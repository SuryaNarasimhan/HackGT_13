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
    MismatchCandidate,
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
_mismatch_logger = logging.getLogger("subtext.mismatch")
if not _mismatch_logger.handlers:
    _mismatch_handler = logging.StreamHandler()
    _mismatch_handler.setFormatter(logging.Formatter("%(asctime)s [Mismatch] %(message)s", "%H:%M:%S"))
    _mismatch_logger.addHandler(_mismatch_handler)
    _mismatch_logger.propagate = False
_mismatch_logger.setLevel(logging.INFO)


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
        self._mismatch_candidates: deque[MismatchCandidate] = deque(maxlen=self.maximum_cues_per_kind)
        self._reviewed_mismatch_candidates: set[str] = set()
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
        self._mismatch_candidates.clear()
        self._reviewed_mismatch_candidates.clear()
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
        self._mismatch_candidates.clear()
        self._reviewed_mismatch_candidates.clear()
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

    def add_mismatch_candidate(self, event: dict[str, Any]) -> None:
        candidate = MismatchCandidate.model_validate(
            {key: value for key, value in event.items() if key != "type"}
        )
        self._mismatch_candidates.append(candidate)
        self._trim_old_cues(candidate.end_s)
        _mismatch_logger.info(
            "local candidate queued | id=%s source=%s trigger=%s time=%.1f-%.1fs "
            "gemini_configured=%s",
            candidate.id,
            candidate.speaker_id,
            candidate.trigger,
            candidate.start_s,
            candidate.end_s,
            self.interpreter.configured,
        )

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

    def add_valence_update(self, event: dict[str, Any]) -> None:
        visual_id = event.get("visual_id")
        if not isinstance(visual_id, str) or not visual_id:
            return

        scores: dict[str, tuple[float | None, int]] = {}
        for score in event.get("scores", []):
            if not isinstance(score, dict) or not isinstance(score.get("track_id"), str):
                continue
            try:
                valence = float(score["valence"]) if score.get("valence") is not None else None
                frames_seen = max(0, int(score.get("frames_seen", 0)))
                if valence is not None and not -1.0 <= valence <= 1.0:
                    continue
            except (TypeError, ValueError):
                continue
            scores[score["track_id"]] = (valence, frames_seen)

        if not scores:
            return
        for index in range(len(self._visual) - 1, -1, -1):
            cue = self._visual[index]
            if cue.id != visual_id:
                continue
            subjects = [
                VisualSubjectCue.model_validate(
                    {
                        **subject.model_dump(exclude_none=True),
                        "facial_valence": scores[subject.track_id][0],
                        "valence_frames_seen": scores[subject.track_id][1],
                    }
                )
                if subject.track_id in scores
                else subject
                for subject in cue.subjects
            ]
            self._visual[index] = cue.model_copy(update={"subjects": subjects})
            return

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
        mismatch_candidates = sorted(
            (
                candidate
                for candidate in self._mismatch_candidates
                if candidate.end_s >= start_s and candidate.start_s <= current_s
            ),
            key=lambda candidate: candidate.start_s,
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
            mismatch_candidates=mismatch_candidates,
            timeline=timeline,
        )
        if not sequence.transcript or not sequence.audio or not sequence.visual:
            return

        self._next_analysis_at = current_s + self.analysis_spacing_seconds
        self._analysis_in_flight = True
        _gemini_logger.info(
            "sending to %s (%.1fs window; transcript=%d, audio=%d, visual=%d cues, mismatch_candidates=%d)",
            self.interpreter.model,
            sequence.end_s - sequence.start_s,
            len(sequence.transcript),
            len(sequence.audio),
            len(sequence.visual),
            len(sequence.mismatch_candidates),
        )
        if sequence.mismatch_candidates:
            _mismatch_logger.info(
                "candidate review scheduled in existing Gemini window | candidates=%s "
                "window=%.1f-%.1fs spacing=%.1fs",
                [candidate.id for candidate in sequence.mismatch_candidates],
                sequence.start_s,
                sequence.end_s,
                self.analysis_spacing_seconds,
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
            if sequence.mismatch_candidates:
                _mismatch_logger.warning(
                    "candidate review failed | candidate_count=%d error_type=%s",
                    len(sequence.mismatch_candidates),
                    type(exc).__name__,
                )
            await publish(
                {
                    "type": "llm_status",
                    "state": "error",
                    "detail": str(exc),
                }
            )
            await self._publish_unavailable_mismatch_reviews(sequence, publish)
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
            await self._publish_unavailable_mismatch_reviews(sequence, publish)
        else:
            if generation != self._generation:
                return
            _gemini_logger.info(
                "response from %s in %.2fs:\n%s",
                self.interpreter.model,
                time.monotonic() - request_started,
                result.model_dump_json(indent=2, exclude_none=True),
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
            await self._publish_mismatch_alerts(sequence, result, publish)
        finally:
            if generation == self._generation:
                self._analysis_in_flight = False

    async def _publish_mismatch_alerts(
        self,
        sequence: AlignedSequence,
        result: LLMInterpretation,
        publish: EventHandler,
    ) -> None:
        candidates = sequence.mismatch_candidates
        if not candidates:
            return

        signal = result.implied_meaning
        evidence_ids = set(signal.evidence_ids)
        can_confirm = (
            signal.detected
            and signal.kind == "sarcasm"
            and signal.confidence >= 70
            and bool(signal.quote.strip())
            and bool(signal.meaning.strip())
        )

        for candidate in candidates:
            if candidate.id in self._reviewed_mismatch_candidates:
                continue

            candidate_is_supported = (
                can_confirm
                and candidate.transcript_id in evidence_ids
                and candidate.audio_id in evidence_ids
            )
            self._reviewed_mismatch_candidates.add(candidate.id)
            if not candidate_is_supported:
                _mismatch_logger.info(
                    "candidate review processed | candidate_id=%s outcome=unclear detected=%s "
                    "kind=%s confidence=%d evidence_ids=%s",
                    candidate.id,
                    signal.detected,
                    signal.kind,
                    signal.confidence,
                    signal.evidence_ids,
                )
                await publish(
                    {
                        "type": "mismatch_review",
                        "id": candidate.id,
                        "timestamp": time.time(),
                        "outcome": "unclear",
                    }
                )
                continue

            _mismatch_logger.info(
                "candidate review processed | candidate_id=%s outcome=confirmed confidence=%d "
                "evidence_ids=%s",
                candidate.id,
                signal.confidence,
                signal.evidence_ids,
            )
            await publish(
                {
                    "type": "mismatch_alert",
                    "id": candidate.id,
                    "timestamp": time.time(),
                    "quote": signal.quote,
                    "interpretation": signal.meaning,
                }
            )

    async def _publish_unavailable_mismatch_reviews(
        self,
        sequence: AlignedSequence,
        publish: EventHandler,
    ) -> None:
        for candidate in sequence.mismatch_candidates:
            if candidate.id in self._reviewed_mismatch_candidates:
                continue
            _mismatch_logger.info(
                "candidate review processed | candidate_id=%s outcome=unavailable confidence=n/a",
                candidate.id,
            )
            await publish(
                {
                    "type": "mismatch_review",
                    "id": candidate.id,
                    "timestamp": time.time(),
                    "outcome": "unavailable",
                }
            )

    def _trim_old_cues(self, current_s: float) -> None:
        cutoff = max(0.0, current_s - 90.0)
        for collection in (
            self._transcript,
            self._audio,
            self._visual,
            self._mismatch_candidates,
        ):
            while collection and collection[0].end_s < cutoff:
                collection.popleft()
