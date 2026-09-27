"""Regression coverage for local and selected-window transcript delivery."""

from __future__ import annotations

import asyncio
import base64
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import app as app_module
from transcriber import AudioTranscriber


class AudioActivityTests(unittest.TestCase):
    def test_quiet_speech_from_both_sources_starts_with_preroll(self) -> None:
        for speaker_id in ("self", "other"):
            with self.subTest(speaker_id=speaker_id):
                transcriber = AudioTranscriber()
                transcriber._numpy = np
                states: dict[str, dict] = {}

                quiet_rms, voice_rms = (
                    (0.0015, 0.004)
                    if speaker_id == "self"
                    else (0.004, 0.010)
                )

                # Three quiet frames leave a 200 ms tail in the pre-roll. The
                # next frame is voice-level but quieter than the old mic gate.
                for index in range(3):
                    transcriber._process_frame(
                        states,
                        speaker_id,
                        100.0 + index * 0.1,
                        np.full(1_600, quiet_rms, dtype=np.float32),
                        0,
                    )
                transcriber._process_frame(
                    states,
                    speaker_id,
                    100.3,
                    np.full(1_600, voice_rms, dtype=np.float32),
                    0,
                )

                state = states[speaker_id]
                self.assertTrue(state["active"])
                self.assertAlmostEqual(state["start_epoch"], 100.1)
                self.assertEqual(sum(len(frame) for frame in state["chunks"]), 4_800)

    def test_microphone_gate_accepts_quiet_speech_without_lowering_call_gate(self) -> None:
        transcriber = AudioTranscriber()
        transcriber._numpy = np
        microphone_states: dict[str, dict] = {}
        call_states: dict[str, dict] = {}
        quiet_voice = np.full(1_600, 0.004, dtype=np.float32)

        transcriber._process_frame(microphone_states, "self", 100.0, quiet_voice, 0)
        transcriber._process_frame(call_states, "other", 100.0, quiet_voice, 0)

        self.assertTrue(microphone_states["self"]["active"])
        self.assertFalse(call_states["other"]["active"])

    def test_preroll_does_not_make_a_short_noise_burst_a_full_utterance(self) -> None:
        transcriber = AudioTranscriber()
        transcriber._numpy = np
        states: dict[str, dict] = {}
        quiet = np.full(1_600, 0.004, dtype=np.float32)
        voice = np.full(1_600, 0.010, dtype=np.float32)
        for index in range(2):
            transcriber._process_frame(states, "other", 100.0 + index * 0.1, quiet, 0)
        for index in range(2):
            transcriber._process_frame(states, "other", 100.2 + index * 0.1, voice, 0)

        transcriber._flush_utterance("other", states["other"])

        self.assertEqual(transcriber._metrics["utterances_too_short"], 1)
        self.assertFalse(states["other"]["active"])


class TranscriptEmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_transcribes_queued_audio_for_both_sources(self) -> None:
        class FakeWhisper:
            def __init__(self) -> None:
                self.call_count = 0
                self.complete = threading.Event()

            def transcribe(self, _audio, **_options):
                self.call_count += 1
                if self.call_count == 2:
                    self.complete.set()
                return [
                    SimpleNamespace(
                        text="spoken words",
                        no_speech_prob=0.1,
                        avg_logprob=-0.2,
                        compression_ratio=1.1,
                    )
                ], None

        transcriber = AudioTranscriber()
        transcriber._numpy = np
        transcriber._model = FakeWhisper()
        transcriber.session_start_epoch = 100.0
        transcriber._loop = asyncio.get_running_loop()
        events: list[dict] = []

        async def capture(event: dict) -> None:
            events.append(event)

        transcriber._event_handler = capture
        quiet = np.full(6_400, 0.004, dtype=np.float32)
        voice = np.full(9_600, 0.010, dtype=np.float32)
        silence = np.zeros(12_800, dtype=np.float32)
        samples = np.concatenate((quiet, voice, silence))
        worker = threading.Thread(target=transcriber._consume_audio, daemon=True)
        transcriber._worker = worker
        worker.start()
        transcriber.enqueue_microphone_audio(samples, 100.0)
        transcriber.enqueue_call_audio(samples, 100.0)

        try:
            completed = await asyncio.wait_for(
                asyncio.to_thread(transcriber._model.complete.wait, 2),
                timeout=3,
            )
            self.assertTrue(completed, "both source utterances should reach Whisper")
        finally:
            transcriber._stop_event.set()
            await asyncio.to_thread(worker.join, 2)
        await asyncio.sleep(0.02)

        transcripts = [event for event in events if event.get("type") == "transcript"]
        self.assertEqual(
            [(event["speaker_id"], event["text"]) for event in transcripts],
            [("self", "spoken words"), ("other", "spoken words")],
        )

    async def test_moderate_confidence_text_is_published_for_each_source(self) -> None:
        class FakeWhisper:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def transcribe(self, _audio, **options):
                self.calls.append(options)
                # This valid segment was rejected by the former overly strict
                # -0.65 log-probability and 0.45 no-speech cutoffs.
                segment = SimpleNamespace(
                    text="  A clear test phrase.  ",
                    no_speech_prob=0.55,
                    avg_logprob=-0.90,
                    compression_ratio=2.3,
                )
                return [segment], None

        transcriber = AudioTranscriber()
        transcriber._numpy = np
        model = FakeWhisper()
        transcriber._model = model
        transcriber.session_start_epoch = 100.0
        events: list[dict] = []

        async def capture(event: dict) -> None:
            events.append(event)

        transcriber._loop = asyncio.get_running_loop()
        transcriber._event_handler = capture
        audio = np.full(8_000, 0.02, dtype=np.float32)
        for speaker_id in ("self", "other"):
            transcriber._transcribe(
                speaker_id,
                [audio],
                100.0,
                100.5,
                transcriber._microphone_generation if speaker_id == "self" else 0,
            )

        await asyncio.sleep(0.02)
        transcripts = [event for event in events if event.get("type") == "transcript"]
        self.assertEqual(
            [(event["speaker_id"], event["text"]) for event in transcripts],
            [
                ("self", "A clear test phrase."),
                ("other", "A clear test phrase."),
            ],
        )
        self.assertEqual(len(model.calls), 2)
        for options in model.calls:
            self.assertEqual(options["no_speech_threshold"], 0.60)
            self.assertEqual(options["log_prob_threshold"], -1.0)
            self.assertEqual(options["compression_ratio_threshold"], 2.4)

    async def test_muting_self_does_not_block_other_source(self) -> None:
        class FakeWhisper:
            def transcribe(self, _audio, **_options):
                return [
                    SimpleNamespace(
                        text="remote voice",
                        no_speech_prob=0.1,
                        avg_logprob=-0.2,
                        compression_ratio=1.1,
                    )
                ], None

        transcriber = AudioTranscriber()
        transcriber._numpy = np
        transcriber._model = FakeWhisper()
        transcriber.session_start_epoch = 100.0
        transcriber.set_microphone_enabled(False)
        events: list[dict] = []

        async def capture(event: dict) -> None:
            events.append(event)

        transcriber._loop = asyncio.get_running_loop()
        transcriber._event_handler = capture
        audio = np.full(8_000, 0.02, dtype=np.float32)
        transcriber._transcribe("self", [audio], 100.0, 100.5, 0)
        transcriber._transcribe("other", [audio], 100.0, 100.5, 0)
        await asyncio.sleep(0.02)

        transcripts = [event for event in events if event.get("type") == "transcript"]
        self.assertEqual([event["speaker_id"] for event in transcripts], ["other"])


class IngestSourceRoutingTests(unittest.TestCase):
    def test_websocket_routes_microphone_and_call_audio_to_distinct_sources(self) -> None:
        class CapturingTranscriber:
            sample_rate = 16_000
            state = "listening"

            def __init__(self) -> None:
                self.microphone_packets: list[tuple[np.ndarray, float]] = []
                self.call_packets: list[tuple[np.ndarray, float]] = []
                self.microphone_enabled = True

            async def stop(self) -> None:
                self.state = "idle"

            def set_microphone_enabled(self, enabled: bool) -> None:
                self.microphone_enabled = enabled

            def enqueue_microphone_audio(self, samples: np.ndarray, timestamp: float) -> None:
                self.microphone_packets.append((samples, timestamp))

            def enqueue_call_audio(self, samples: np.ndarray, timestamp: float) -> None:
                self.call_packets.append((samples, timestamp))

        fake = CapturingTranscriber()
        pcm = np.full(1_600, 0.02, dtype="<f4").tobytes()
        payload = base64.b64encode(pcm).decode("ascii")
        with patch.object(app_module, "transcriber", fake):
            with TestClient(app_module.app) as client:
                with client.websocket_connect("/ingest") as websocket:
                    websocket.send_json(
                        {
                            "type": "microphone_chunk",
                            "timestamp": 101.0,
                            "sample_rate": 16_000,
                            "pcm_format": "float32",
                            "samples": payload,
                        }
                    )
                    websocket.send_json(
                        {
                            "type": "audio_chunk",
                            "timestamp": 102.0,
                            "sample_rate": 16_000,
                            "pcm_format": "float32",
                            "samples": payload,
                        }
                    )

        self.assertEqual(len(fake.microphone_packets), 1)
        self.assertEqual(len(fake.call_packets), 1)
        self.assertEqual(fake.microphone_packets[0][1], 101.0)
        self.assertEqual(fake.call_packets[0][1], 102.0)
        np.testing.assert_allclose(fake.microphone_packets[0][0], 0.02)
        np.testing.assert_allclose(fake.call_packets[0][0], 0.02)


if __name__ == "__main__":
    unittest.main()
