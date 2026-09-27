"""Backend smoke checks that do not open the microphone or download Whisper."""

from __future__ import annotations

import asyncio
import unittest

from fastapi.testclient import TestClient

from app import app, history
from transcriber import AudioTranscriber


class TranscriberSmokeTests(unittest.TestCase):
    def test_stopping_when_idle_is_safe(self) -> None:
        transcriber = AudioTranscriber()

        asyncio.run(transcriber.stop())

        self.assertEqual(transcriber.state, "idle")

    def test_model_is_lazy_until_listening_starts(self) -> None:
        transcriber = AudioTranscriber()

        self.assertFalse(transcriber.model_loaded)
        self.assertEqual(transcriber.state, "idle")


class ServiceSmokeTests(unittest.TestCase):
    def test_health_endpoint_reports_idle_without_loading_whisper(self) -> None:
        with TestClient(app) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"], "idle")
        self.assertFalse(response.json()["model_loaded"])

    def test_websocket_sends_initial_state(self) -> None:
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                event = websocket.receive_json()

        self.assertEqual(event["type"], "status")
        self.assertEqual(event["state"], "idle")
        self.assertIsInstance(event["gemini_configured"], bool)

    def test_clear_endpoint_removes_reconnect_history(self) -> None:
        history.append({"type": "transcript", "text": "temporary test phrase"})

        with TestClient(app) as client:
            response = client.delete("/transcript")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(history), 0)


if __name__ == "__main__":
    unittest.main()
