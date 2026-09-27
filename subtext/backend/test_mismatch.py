"""Focused checks for the local mismatch gate and its quiet alert path."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from gemini import GeminiInterpreter
from pipeline import ConversationPipeline
from sarcasm import SarcasmDetector
from schemas import (
    AlignedSequence,
    AudioCue,
    ImpliedMeaningSignal,
    LLMInterpretation,
    MismatchCandidate,
    SpeechSegment,
    TranscriptCue,
    VisualCue,
    VisualSubjectCue,
)


class LocalMismatchGateTests(unittest.TestCase):
    def test_reports_diagnostics_and_candidate_only_when_both_modalities_pass(self) -> None:
        detector = SarcasmDetector()
        with (
            patch.object(
                detector,
                "_text_cues",
                return_value=(0.78, ["sarcasm-like wording"], 0.5),
            ),
            patch.object(detector, "_audio_features", return_value={"pitch_spread": 9.5}),
            patch.object(
                detector,
                "_audio_cues",
                return_value=(0.68, ["strong pitch emphasis"]),
            ),
        ):
            candidate, diagnostics = detector.analyze_with_diagnostics(
                "Yeah right, perfect timing.", [], 16_000, None
            )

        self.assertEqual(candidate["kind"], "possible_sarcasm")
        self.assertTrue(diagnostics["detected"])
        self.assertEqual(diagnostics["text_score"], 0.78)
        self.assertEqual(diagnostics["audio_score"], 0.68)


class MismatchPipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.pipeline = ConversationPipeline()
        self.candidate = MismatchCandidate(
            id="mismatch-1",
            transcript_id="transcript-1",
            audio_id="audio-transcript-1",
            speaker_id="other",
            start_s=1.0,
            end_s=2.0,
            trigger="possible_sarcasm",
        )
        self.sequence = AlignedSequence(
            session_id="test-session",
            start_s=0,
            end_s=4,
            mismatch_candidates=[self.candidate],
        )

    async def test_confirmed_candidate_publishes_one_minimal_alert(self) -> None:
        result = LLMInterpretation(
            summary="The wording and delivery may be incongruent.",
            implied_meaning=ImpliedMeaningSignal(
                detected=True,
                kind="sarcasm",
                quote="Yeah, perfect timing.",
                meaning="They may be expressing frustration sarcastically.",
                evidence_ids=[
                    "transcript-1",
                    "audio-transcript-1",
                    "visual-1",
                ],
                confidence=84,
            ),
        )
        events: list[dict] = []

        async def publish(event: dict) -> None:
            events.append(event)

        await self.pipeline._publish_mismatch_alerts(self.sequence, result, publish)
        await self.pipeline._publish_mismatch_alerts(self.sequence, result, publish)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "mismatch_alert")
        self.assertEqual(events[0]["interpretation"], result.implied_meaning.meaning)
        self.assertNotIn("confidence", events[0])
        self.assertNotIn("evidence_ids", events[0])

    async def test_alert_is_suppressed_when_model_does_not_cite_candidate(self) -> None:
        result = LLMInterpretation(
            summary="The meaning is unclear.",
            implied_meaning=ImpliedMeaningSignal(
                detected=True,
                kind="sarcasm",
                quote="Another phrase.",
                meaning="Possibly sarcastic.",
                evidence_ids=["different-transcript", "different-audio", "visual-1"],
                confidence=90,
            ),
        )
        events: list[dict] = []

        async def publish(event: dict) -> None:
            events.append(event)

        await self.pipeline._publish_mismatch_alerts(self.sequence, result, publish)

        self.assertEqual(
            [(event["type"], event.get("outcome")) for event in events],
            [("mismatch_review", "unclear")],
        )

    async def test_candidate_uses_existing_nine_second_analysis_spacing(self) -> None:
        class FakeInterpreter:
            configured = True
            model = "test-model"

            def __init__(self) -> None:
                self.sequences: list[AlignedSequence] = []

            async def interpret(self, sequence: AlignedSequence) -> LLMInterpretation:
                self.sequences.append(sequence)
                return LLMInterpretation(summary="No supported sarcasm.", no_clear_signal=True)

        fake_interpreter = FakeInterpreter()
        self.pipeline.interpreter = fake_interpreter
        self.pipeline.begin(started_at_epoch=100.0)
        events: list[dict] = []

        async def publish(event: dict) -> None:
            events.append(event)

        def add_cues(end_s: float) -> None:
            start_s = end_s - 0.5
            transcript_id = f"transcript-{end_s}"
            audio_id = f"audio-{transcript_id}"
            self.pipeline.add_transcript_event(
                {
                    "id": transcript_id,
                    "speaker_id": "other",
                    "start_s": start_s,
                    "end_s": end_s,
                    "text": "Yeah right, perfect timing.",
                }
            )
            self.pipeline.add_audio_event(
                {
                    "cue": AudioCue(
                        id=audio_id,
                        speaker_id="other",
                        start_s=start_s,
                        end_s=end_s,
                        speech_segments=[SpeechSegment(start_s=start_s, end_s=end_s)],
                    ).model_dump()
                }
            )
            self.pipeline.add_visual_frame(
                {
                    "id": f"visual-{end_s}",
                    "timestamp": 100.0 + end_s,
                    "subjects": [{"track_id": "face-1", "head_yaw": end_s}],
                }
            )

        add_cues(8.0)
        await self.pipeline.maybe_analyze(publish)
        await asyncio.sleep(0.01)
        self.assertEqual(len(fake_interpreter.sequences), 1)

        add_cues(8.5)
        self.pipeline.add_mismatch_candidate(
            {
                "id": "mismatch-1",
                "transcript_id": "transcript-8.5",
                "audio_id": "audio-transcript-8.5",
                "speaker_id": "other",
                "start_s": 8.0,
                "end_s": 8.5,
            }
        )
        await self.pipeline.maybe_analyze(publish)
        await asyncio.sleep(0.01)
        self.assertEqual(len(fake_interpreter.sequences), 1)

        add_cues(17.0)
        await self.pipeline.maybe_analyze(publish)
        await asyncio.sleep(0.01)

        self.assertEqual(len(fake_interpreter.sequences), 2)
        self.assertEqual(
            [candidate.id for candidate in fake_interpreter.sequences[1].mismatch_candidates],
            ["mismatch-1"],
        )

    def test_local_candidate_event_and_face_valence_attach_to_existing_visual_cue(self) -> None:
        self.pipeline.started_at_epoch = 100.0
        self.pipeline.add_visual_frame(
            {
                "id": "visual-1",
                "timestamp": 101.5,
                "subjects": [{"track_id": "face-1", "head_yaw": 3.0}],
            }
        )
        self.pipeline.add_valence_update(
            {
                "visual_id": "visual-1",
                "scores": [{"track_id": "face-1", "valence": -0.6, "frames_seen": 10}],
            }
        )
        self.pipeline.add_mismatch_candidate(
            {
                "type": "mismatch_candidate",
                "id": "mismatch-1",
                "transcript_id": "transcript-1",
                "audio_id": "audio-transcript-1",
                "speaker_id": "other",
                "start_s": 1.0,
                "end_s": 2.0,
                "trigger": "possible_sarcasm",
            }
        )

        subject = self.pipeline._visual[0].subjects[0]
        self.assertEqual(subject.facial_valence, -0.6)
        self.assertEqual(subject.valence_frames_seen, 10)
        self.assertEqual(self.pipeline._mismatch_candidates[0].id, "mismatch-1")

    def test_face_model_valence_can_ground_the_visual_evidence(self) -> None:
        sequence = AlignedSequence(
            session_id="test-session",
            start_s=1,
            end_s=2,
            transcript=[
                TranscriptCue(
                    id="transcript-1",
                    speaker_id="other",
                    start_s=1,
                    end_s=2,
                    text="Yeah, perfect timing.",
                )
            ],
            audio=[
                AudioCue(
                    id="audio-1",
                    speaker_id="other",
                    start_s=1,
                    end_s=2,
                    speech_segments=[SpeechSegment(start_s=1, end_s=2)],
                )
            ],
            visual=[
                VisualCue(
                    id="visual-1",
                    start_s=1.5,
                    end_s=1.5,
                    subjects=[
                        VisualSubjectCue(
                            track_id="face-1",
                            face_present=True,
                            facial_valence=-0.6,
                            valence_frames_seen=10,
                        )
                    ],
                )
            ],
        )
        signal = ImpliedMeaningSignal(
            detected=True,
            kind="sarcasm",
            quote="Yeah, perfect timing.",
            meaning="They may be expressing frustration sarcastically.",
            evidence_ids=["transcript-1", "audio-1", "visual-1"],
            confidence=84,
        )

        self.assertTrue(GeminiInterpreter._implied_meaning_is_grounded(signal, sequence))


if __name__ == "__main__":
    unittest.main()
