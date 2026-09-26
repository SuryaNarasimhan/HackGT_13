from __future__ import annotations

import re
import threading
from collections import deque
from typing import Any


_WORD_PATTERN = re.compile(r"[a-z']+")
_SARCASTIC_PHRASES = (
    "yeah right",
    "oh sure",
    "as if",
    "what a surprise",
    "well that went well",
    "just what i needed",
    "love that for me",
)
_POSITIVE_REACTION_WORDS = {"amazing", "awesome", "excellent", "fantastic", "great", "perfect", "wonderful"}
_CONTRAST_WORDS = {"again", "another", "broken", "crash", "crashed", "crashing", "failed", "late", "waiting"}
_POSITIVE_WORDS = {
    "amazing", "awesome", "brilliant", "excellent", "fantastic", "good",
    "great", "happy", "love", "perfect", "wonderful",
}
_NEGATIVE_WORDS = {
    "annoying", "awful", "bad", "broken", "failed", "hate", "horrible",
    "late", "missed", "problem", "terrible", "wrong",
}


class SarcasmDetector:
    """Conservative, local text-and-prosody cue detector for one speaker.

    This is a heuristic MVP rather than a trained sarcasm model. It looks for
    wording/context incongruity and unusual vocal delivery relative to the
    speaker's recent baseline. Audio is analyzed in memory and never retained.
    """

    def __init__(self) -> None:
        self._context: deque[float] = deque(maxlen=6)
        self._audio_baseline: deque[dict[str, float]] = deque(maxlen=8)
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._context.clear()
            self._audio_baseline.clear()

    def analyze(
        self,
        text: str,
        audio: Any,
        sample_rate: int,
        numpy: Any,
    ) -> dict[str, Any] | None:
        normalized = " ".join(text.lower().split())
        if not normalized:
            return None

        with self._lock:
            text_score, text_evidence, sentiment = self._text_cues(normalized)
            features = self._audio_features(audio, text, sample_rate, numpy)
            audio_score, audio_evidence = self._audio_cues(features)

            # Require evidence from both the words/context and the voice. The
            # score is only a ranking aid; it is not a calibrated probability.
            combined_score = 0.52 * text_score + 0.48 * audio_score
            detected = (
                text_score >= 0.62
                and audio_score >= 0.58
                and combined_score >= 0.64
            )

            self._context.append(sentiment)
            if features:
                self._audio_baseline.append(features)

            if not detected:
                return None

            return {
                "kind": "possible_sarcasm",
                "label": "Possible sarcasm",
                "evidence": list(dict.fromkeys(text_evidence + audio_evidence)),
            }

    def _text_cues(self, text: str) -> tuple[float, list[str], float]:
        tokens = _WORD_PATTERN.findall(text)
        positive_count = sum(token in _POSITIVE_WORDS for token in tokens)
        negative_count = sum(token in _NEGATIVE_WORDS for token in tokens)
        sentiment = (positive_count - negative_count) / max(1, positive_count + negative_count)
        score = 0.0
        evidence: list[str] = []

        if any(phrase in text for phrase in _SARCASTIC_PHRASES):
            score = max(score, 0.78)
            evidence.append("sarcasm-like wording")

        if positive_count and negative_count:
            score = max(score, 0.82)
            evidence.append("mixed positive and negative wording")

        if any(token in _POSITIVE_REACTION_WORDS for token in tokens) and any(
            token in _CONTRAST_WORDS for token in tokens
        ):
            score = max(score, 0.74)
            evidence.append("positive wording paired with a setback")

        if len(self._context) >= 2 and sentiment:
            prior_sentiment = sum(self._context) / len(self._context)
            if sentiment * prior_sentiment < -0.35:
                score = max(score, 0.66)
                evidence.append("wording differs from the recent context")

        return score, evidence, sentiment

    def _audio_features(
        self,
        audio: Any,
        text: str,
        sample_rate: int,
        numpy: Any,
    ) -> dict[str, float]:
        samples = numpy.asarray(audio, dtype=numpy.float32).reshape(-1)
        if samples.size == 0:
            return {}

        # Whisper's segment includes the silence used to close the utterance.
        # Trim that tail so speech rate and prosody describe spoken audio only.
        block = max(1, int(sample_rate * 0.1))
        block_count = samples.size // block
        if block_count:
            blocks = samples[: block_count * block].reshape(block_count, block)
            block_rms = numpy.sqrt(numpy.mean(blocks * blocks, axis=1))
            voiced_blocks = numpy.flatnonzero(block_rms >= 0.008)
            if voiced_blocks.size:
                samples = samples[: min(samples.size, int((voiced_blocks[-1] + 1) * block))]

        duration = samples.size / sample_rate
        if duration <= 0:
            return {}

        frame_size = int(sample_rate * 0.04)
        frame_step = int(sample_rate * 0.02)
        rms_values: list[float] = []
        pitch_values: list[float] = []
        min_lag = max(1, int(sample_rate / 400))
        max_lag = int(sample_rate / 80)

        for start in range(0, max(0, samples.size - frame_size + 1), frame_step):
            frame = samples[start : start + frame_size]
            frame = frame - numpy.mean(frame)
            rms = float(numpy.sqrt(numpy.mean(frame * frame)))
            if rms < 0.006:
                continue
            rms_values.append(rms)

            correlation = numpy.correlate(frame, frame, mode="full")[frame_size - 1 :]
            upper = min(max_lag, correlation.size - 1)
            if upper <= min_lag or correlation[0] <= 0:
                continue
            section = correlation[min_lag : upper + 1]
            lag = min_lag + int(numpy.argmax(section))
            strength = float(correlation[lag] / correlation[0])
            if strength >= 0.35:
                pitch_values.append(sample_rate / lag)

        words = _WORD_PATTERN.findall(text.lower())
        features: dict[str, float] = {
            "speech_rate": len(words) / duration,
        }
        if rms_values:
            db_values = 20.0 * numpy.log10(numpy.asarray(rms_values) + 1e-8)
            features["energy_spread"] = float(numpy.percentile(db_values, 90) - numpy.percentile(db_values, 10))
        if pitch_values:
            semitones = 12.0 * numpy.log2(numpy.asarray(pitch_values) / 220.0)
            features["pitch_median"] = float(numpy.median(semitones))
            features["pitch_spread"] = float(numpy.percentile(semitones, 90) - numpy.percentile(semitones, 10))
        return features

    def _audio_cues(self, features: dict[str, float]) -> tuple[float, list[str]]:
        if not features:
            return 0.0, []

        scores: list[float] = []
        evidence: list[str] = []
        baseline = list(self._audio_baseline)

        if "pitch_spread" in features:
            pitch_spread = features["pitch_spread"]
            if len(baseline) >= 3:
                prior_spreads = [item["pitch_spread"] for item in baseline if "pitch_spread" in item]
                if prior_spreads:
                    prior = float(sorted(prior_spreads)[len(prior_spreads) // 2])
                    if pitch_spread >= max(7.0, prior * 1.45):
                        scores.append(0.78)
                        evidence.append("unusually wide pitch movement")
                    elif prior >= 4.0 and pitch_spread <= prior * 0.55:
                        scores.append(0.65)
                        evidence.append("flatter delivery than usual")
            elif pitch_spread >= 9.0:
                scores.append(0.68)
                evidence.append("strong pitch emphasis")

        if "pitch_median" in features and len(baseline) >= 3:
            prior_pitch = [item["pitch_median"] for item in baseline if "pitch_median" in item]
            if prior_pitch:
                prior = float(sorted(prior_pitch)[len(prior_pitch) // 2])
                if abs(features["pitch_median"] - prior) >= 3.5:
                    scores.append(0.62)
                    evidence.append("pitch shifted from your usual range")

        if "energy_spread" in features:
            spread = features["energy_spread"]
            if len(baseline) >= 3:
                prior_energy = [item["energy_spread"] for item in baseline if "energy_spread" in item]
                if prior_energy:
                    prior = float(sorted(prior_energy)[len(prior_energy) // 2])
                    if spread >= max(9.0, prior + 4.0):
                        scores.append(0.68)
                        evidence.append("unusual vocal emphasis")
            elif spread >= 13.0:
                scores.append(0.62)
                evidence.append("strong vocal emphasis")

        if "speech_rate" in features and len(baseline) >= 3:
            prior_rates = [item["speech_rate"] for item in baseline if item.get("speech_rate", 0) > 0]
            if prior_rates:
                prior = float(sorted(prior_rates)[len(prior_rates) // 2])
                rate = features["speech_rate"]
                if prior > 0 and (rate <= prior * 0.62 or rate >= prior * 1.55):
                    scores.append(0.60)
                    evidence.append("speech rate shifted from your usual pace")

        return (max(scores), evidence) if scores else (0.0, [])
