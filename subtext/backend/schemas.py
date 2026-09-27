"""Typed, timestamped evidence exchanged by Subtext's local pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TranscriptCue(StrictModel):
    id: str
    speaker_id: str = Field(description="Temporary source label, currently self or other.")
    start_s: float = Field(ge=0, description="Session-relative start time in seconds.")
    end_s: float = Field(ge=0, description="Session-relative end time in seconds.")
    text: str
    confidence: float | None = Field(default=None, ge=0, le=1, description="Recognizer confidence when available.")


class SpeechSegment(StrictModel):
    start_s: float = Field(ge=0, description="Session-relative start time in seconds.")
    end_s: float = Field(ge=0, description="Session-relative end time in seconds.")


class AudioCue(StrictModel):
    id: str
    speaker_id: str = Field(description="Temporary source label, currently self or other.")
    start_s: float = Field(ge=0, description="Session-relative start time in seconds.")
    end_s: float = Field(ge=0, description="Session-relative end time in seconds.")
    speech_segments: list[SpeechSegment] = Field(default_factory=list)
    response_gap_ms: float | None = Field(default=None, ge=0, description="Quiet time after the other source's last utterance.")
    pause_ms: float | None = Field(default=None, ge=0, description="Estimated non-speech time inside this utterance.")
    overlap_ms: float | None = Field(default=None, ge=0, description="Estimated overlap with the other source's last utterance.")
    pitch_median_hz: float | None = Field(default=None, gt=0, description="Approximate median voice pitch in hertz.")
    pitch_range_semitones: float | None = Field(default=None, ge=0, description="Approximate 90th-to-10th percentile pitch range in semitones.")
    loudness_dbfs: float | None = Field(default=None, description="Median voice-active level in dBFS.")
    speaking_rate_wpm: float | None = Field(default=None, ge=0, description="Estimated words per active minute.")
    change_from_baseline: dict[str, float] = Field(default_factory=dict, description="Approximate difference from recent measurements of this source.")
    vocal_events: list[dict[str, str | float | bool]] = Field(default_factory=list)
    quality: dict[str, str | float | bool] = Field(default_factory=dict)


class VisualSubjectCue(StrictModel):
    track_id: str
    face_present: bool
    face_box: list[float] | None = Field(default=None, description="Normalized image [x, y, width, height], each in the 0–1 frame space.")
    head_yaw: float | None = Field(default=None, description="Head yaw in degrees.")
    head_pitch: float | None = Field(default=None, description="Head pitch in degrees.")
    head_roll: float | None = Field(default=None, description="Head roll in degrees.")
    mouth_aperture: float | None = Field(default=None, description="Local landmark height-to-width ratio for the inner-lip region.")
    left_eye_aperture: float | None = Field(default=None, description="Local landmark height-to-width ratio for the left-eye region.")
    right_eye_aperture: float | None = Field(default=None, description="Local landmark height-to-width ratio for the right-eye region.")
    change_from_baseline: dict[str, float] = Field(default_factory=dict)
    facial_valence: float | None = Field(default=None, ge=-1, le=1, description="Smoothed local face-model valence estimate; not a probability.")
    valence_frames_seen: int | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)


class VisualCue(StrictModel):
    id: str
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    subjects: list[VisualSubjectCue] = Field(default_factory=list)
    quality: dict[str, str | float | bool] = Field(default_factory=dict)


class MismatchCandidate(StrictModel):
    """Local text-and-voice gate result awaiting multimodal review."""

    id: str
    transcript_id: str
    audio_id: str
    speaker_id: str
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    trigger: str = "local_text_voice_incongruity"


class AlignedStep(StrictModel):
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    transcript_ids: list[str] = Field(default_factory=list)
    audio_ids: list[str] = Field(default_factory=list)
    visual_ids: list[str] = Field(default_factory=list)


class AlignedSequence(StrictModel):
    session_id: str
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    transcript: list[TranscriptCue] = Field(default_factory=list)
    audio: list[AudioCue] = Field(default_factory=list)
    visual: list[VisualCue] = Field(default_factory=list)
    mismatch_candidates: list[MismatchCandidate] = Field(default_factory=list)
    timeline: list[AlignedStep] = Field(default_factory=list)


class InterpretationHypothesis(StrictModel):
    interpretation: str
    evidence_ids: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "low"
    possible_check_in: str | None = None


class InteractionObservation(StrictModel):
    observation: str
    evidence_ids: list[str] = Field(default_factory=list)


class ImpliedMeaningSignal(StrictModel):
    detected: bool = False
    kind: Literal["idiom", "figurative", "indirect", "sarcasm", "other", "none"] = "none"
    quote: str = ""
    meaning: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: int = Field(default=0, ge=0, le=100)


class LLMInterpretation(StrictModel):
    summary: str
    notable_observations: list[InteractionObservation] = Field(default_factory=list)
    hypotheses: list[InterpretationHypothesis] = Field(default_factory=list)
    no_clear_signal: bool = False
    implied_meaning: ImpliedMeaningSignal = Field(default_factory=ImpliedMeaningSignal)
