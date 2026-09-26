"""Focused tests for Gemini structured-output handling."""

from __future__ import annotations

import json
import io
import os
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from gemini import GeminiError, GeminiInterpreter
from schemas import AlignedSequence


class FakeResponse:
    def __init__(self, body: dict) -> None:
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def candidate(*parts: dict, finish_reason: str = "STOP") -> dict:
    return {
        "candidates": [
            {
                "finishReason": finish_reason,
                "content": {"parts": list(parts)},
            }
        ]
    }


class GeminiStructuredOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sequence = AlignedSequence(session_id="test", start_s=0, end_s=1)
        self.valid_json = json.dumps(
            {
                "summary": "The exchange is brief.",
                "notable_observations": [],
                "hypotheses": [],
                "no_clear_signal": True,
            }
        )

    def interpret_with_response(self, body: dict):
        response = FakeResponse(body)
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "unit-test-key"}),
            patch("gemini.urlopen", return_value=response) as mocked_urlopen,
        ):
            result = GeminiInterpreter()._interpret_sync(self.sequence)
        request = mocked_urlopen.call_args.args[0]
        return result, json.loads(request.data)

    def test_skips_thought_parts_and_reads_final_json(self) -> None:
        result, request_body = self.interpret_with_response(
            candidate(
                {"text": "private reasoning is not the answer", "thought": True},
                {"text": self.valid_json},
            )
        )

        self.assertEqual(result.summary, "The exchange is brief.")
        generation_config = request_body["generationConfig"]
        self.assertEqual(
            generation_config["responseFormat"]["text"]["mimeType"],
            "APPLICATION_JSON",
        )
        self.assertEqual(generation_config["thinkingConfig"]["thinkingLevel"], "low")
        self.assertEqual(generation_config["maxOutputTokens"], 1024)
        self.assertEqual(
            generation_config["responseFormat"]["text"]["schema"]["properties"]["hypotheses"]["items"]["properties"]["possible_check_in"]["type"],
            ["string", "null"],
        )

    def test_joins_split_final_text_parts(self) -> None:
        split_at = len(self.valid_json) // 2
        result, _ = self.interpret_with_response(
            candidate(
                {"text": self.valid_json[:split_at]},
                {"text": self.valid_json[split_at:]},
            )
        )

        self.assertEqual(result.summary, "The exchange is brief.")

    def test_nullable_optional_check_in_matches_model_schema(self) -> None:
        response = json.loads(self.valid_json)
        response["hypotheses"] = [
            {
                "interpretation": "A brief pause may have several explanations.",
                "evidence_ids": [],
                "alternatives": [],
                "confidence": "low",
                "possible_check_in": None,
            }
        ]
        result, _ = self.interpret_with_response(candidate({"text": json.dumps(response)}))

        self.assertIsNone(result.hypotheses[0].possible_check_in)

    def test_retries_temporary_503_with_exponential_backoff(self) -> None:
        response_body = json.loads(self.valid_json)
        responses = [
            self.http_error(503, "temporarily overloaded"),
            self.http_error(503, "temporarily overloaded"),
            FakeResponse(candidate({"text": self.valid_json})),
        ]

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "unit-test-key"}),
            patch("gemini.urlopen", side_effect=responses) as mocked_urlopen,
            patch("gemini.random.uniform", return_value=0.1),
            patch("gemini.time.sleep") as mocked_sleep,
        ):
            result = GeminiInterpreter()._interpret_sync(self.sequence)

        self.assertEqual(result.summary, response_body["summary"])
        self.assertEqual(mocked_urlopen.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in mocked_sleep.call_args_list],
            [1.1, 2.1],
        )

    def test_reports_provider_message_after_503_retries_are_exhausted(self) -> None:
        responses = [
            self.http_error(
                503,
                "This model is currently experiencing high demand.",
            )
            for _ in range(3)
        ]

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "unit-test-key"}),
            patch("gemini.urlopen", side_effect=responses) as mocked_urlopen,
            patch("gemini.random.uniform", return_value=0),
            patch("gemini.time.sleep"),
        ):
            with self.assertRaisesRegex(
                GeminiError,
                "This model is currently experiencing high demand",
            ):
                GeminiInterpreter()._interpret_sync(self.sequence)

        self.assertEqual(mocked_urlopen.call_count, 3)

    @staticmethod
    def http_error(code: int, detail: str):
        body = json.dumps({"error": {"message": detail}}).encode("utf-8")
        return HTTPError(
            "https://generativelanguage.googleapis.com/test",
            code,
            "Service Unavailable",
            {},
            io.BytesIO(body),
        )

    def test_reports_output_limit_instead_of_schema_mismatch(self) -> None:
        with self.assertRaisesRegex(GeminiError, "output token limit was reached"):
            self.interpret_with_response(
                candidate({"text": "{"}, finish_reason="MAX_TOKENS")
            )

    def test_reports_exact_schema_field_for_invalid_response(self) -> None:
        invalid = json.dumps(
            {
                "summary": "The exchange is brief.",
                "notable_observations": [],
                "hypotheses": [
                    {
                        "interpretation": "Maybe.",
                        "evidence_ids": [],
                        "alternatives": [],
                        "confidence": "certain",
                    }
                ],
                "no_clear_signal": False,
            }
        )

        with self.assertRaisesRegex(
            GeminiError,
            r"hypotheses\.0\.confidence: Input should be 'low', 'medium' or 'high'",
        ):
            self.interpret_with_response(candidate({"text": invalid}))

    def test_reports_malformed_json_location(self) -> None:
        with self.assertRaisesRegex(GeminiError, r"invalid JSON \(line 1, column"):
            self.interpret_with_response(candidate({"text": '{"summary":'}))

    def test_reports_safety_block_when_there_is_no_candidate(self) -> None:
        with self.assertRaisesRegex(GeminiError, r"blocked the request \(SAFETY\)"):
            self.interpret_with_response(
                {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}
            )


if __name__ == "__main__":
    unittest.main()
