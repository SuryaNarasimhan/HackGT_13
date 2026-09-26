"""
Semantic Perception Pipeline for SocialLens
Transcribes incoming audio buffers via faster-whisper and classifies
speech sentiment/emotion into the canonical 7-emotion simplex.
"""

import logging
from typing import Dict, Optional, Tuple, Union
import numpy as np

from app.config import WHISPER_MODEL_SIZE, TEXT_EMOTION_MODEL
from app.pipelines.taxonomy import (
    CANONICAL_EMOTIONS,
    NUM_EMOTIONS,
    dict_to_vector,
    normalize_distribution,
    get_top_emotion,
)

logger = logging.getLogger(__name__)

# Fallback semantic lexicon mapping for offline / zero-dependency resilience
KEYWORD_EMOTION_MAP: Dict[str, str] = {
    # Joy / Praise
    "fantastic": "joy",
    "great": "joy",
    "awesome": "joy",
    "brilliant": "joy",
    "wonderful": "joy",
    "amazing": "joy",
    "love": "joy",
    "happy": "joy",
    "congratulations": "joy",
    "excellent": "joy",
    # Surprise
    "wow": "surprise",
    "unbelievable": "surprise",
    "really": "surprise",
    "shocking": "surprise",
    "unexpected": "surprise",
    # Anger / Frustration
    "crashed": "anger",
    "terrible": "anger",
    "broken": "anger",
    "hate": "anger",
    "annoying": "anger",
    "stupid": "anger",
    "awful": "anger",
    "horrible": "anger",
    # Sadness
    "sad": "sadness",
    "depressing": "sadness",
    "sorry": "sadness",
    "unfortunate": "sadness",
    "disappointed": "sadness",
    # Fear
    "scared": "fear",
    "nervous": "fear",
    "worried": "fear",
    "panicking": "fear",
    # Disgust
    "gross": "disgust",
    "nasty": "disgust",
    "disgusting": "disgust",
}


class SemanticPipeline:
    """
    Orchestrates Speech-to-Text (STT) and Semantic Emotion Classification.
    """

    def __init__(
        self,
        whisper_model_size: str = WHISPER_MODEL_SIZE,
        text_model_name: str = TEXT_EMOTION_MODEL,
        lazy_load: bool = True
    ):
        self.whisper_model_size = whisper_model_size
        self.text_model_name = text_model_name
        self._stt_model = None
        self._emotion_classifier = None

        if not lazy_load:
            self._load_stt()
            self._load_emotion_classifier()

    def _load_stt(self):
        """Loads faster-whisper model lazily."""
        if self._stt_model is not None:
            return
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading faster-whisper model ({self.whisper_model_size})...")
            # Run on CPU with INT8 compute type for lightweight real-time latency
            self._stt_model = WhisperModel(
                self.whisper_model_size,
                device="cpu",
                compute_type="int8"
            )
            logger.info("faster-whisper model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load faster-whisper ({e}). STT will use fallback.")
            self._stt_model = None

    def _load_emotion_classifier(self):
        """Loads Hugging Face text emotion pipeline lazily."""
        if self._emotion_classifier is not None:
            return
        try:
            from transformers import pipeline
            logger.info(f"Loading text emotion model ({self.text_model_name})...")
            self._emotion_classifier = pipeline(
                "text-classification",
                model=self.text_model_name,
                top_k=None,
                device=-1  # CPU
            )
            logger.info("Text emotion classifier loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load transformers pipeline ({e}). Semantic pipeline will use heuristic lexicon."
            )
            self._emotion_classifier = None

    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        initial_prompt: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Transcribes a 1D float32 audio buffer.
        Returns: (transcript_text, average_confidence)
        """
        if len(audio_data) == 0:
            return "", 0.0

        self._load_stt()
        if self._stt_model is None:
            # Fallback stub for unit testing without weights downloaded
            return "", 0.0

        try:
            # Ensure float32 format normalized in [-1.0, 1.0]
            audio_f32 = audio_data.astype(np.float32)
            if np.max(np.abs(audio_f32)) > 1.0:
                audio_f32 = audio_f32 / 32768.0

            transcribe_kwargs = {
                "beam_size": 1,
                "language": "en",
                "vad_filter": False,
            }
            if initial_prompt and initial_prompt.strip():
                transcribe_kwargs["initial_prompt"] = initial_prompt.strip()

            segments, info = self._stt_model.transcribe(
                audio_f32,
                **transcribe_kwargs
            )

            texts = []
            probs = []
            for seg in segments:
                texts.append(seg.text.strip())
                probs.append(float(getattr(seg, "avg_logprob", -0.5)))

            full_text = " ".join(texts).strip()
            conf = float(np.exp(np.mean(probs))) if probs else 0.5
            return full_text, conf
        except Exception as e:
            logger.error(f"Error during audio transcription: {e}")
            return "", 0.0

    def classify_text_emotion(self, text: str) -> np.ndarray:
        """
        Classifies input text into a canonical 7-D emotion probability vector.
        """
        clean_text = text.strip()
        if not clean_text:
            # Return canonical neutral distribution if text is empty
            neutral_vec = np.zeros(NUM_EMOTIONS, dtype=np.float64)
            neutral_vec[CANONICAL_EMOTIONS.index("neutral")] = 1.0
            return neutral_vec

        self._load_emotion_classifier()

        if self._emotion_classifier is not None:
            try:
                # Output format: [[{'label': 'joy', 'score': 0.88}, ...]]
                outputs = self._emotion_classifier(clean_text)
                if outputs and isinstance(outputs[0], list):
                    raw_dict = {item["label"].lower(): float(item["score"]) for item in outputs[0]}
                    return dict_to_vector(raw_dict)
            except Exception as e:
                logger.warning(f"Transformers classification failed ({e}). Reverting to lexicon.")

        # Heuristic Lexicon Fallback
        return self._heuristic_lexicon_classify(clean_text)

    def _heuristic_lexicon_classify(self, text: str) -> np.ndarray:
        """
        Rule-based lexicon sentiment scoring when transformers model is offline.
        """
        words = text.lower().split()
        counts: Dict[str, float] = {e: 0.05 for e in CANONICAL_EMOTIONS}
        counts["neutral"] = 0.20  # Baseline neutral prior

        for word in words:
            # Strip punctuation
            clean_word = word.strip(".,!?:;\"'()[]{}")
            if clean_word in KEYWORD_EMOTION_MAP:
                detected_emotion = KEYWORD_EMOTION_MAP[clean_word]
                counts[detected_emotion] += 1.0

        return dict_to_vector(counts)

    def process(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        context_prompt: Optional[str] = None
    ) -> Tuple[str, np.ndarray, float]:
        """
        End-to-end execution: Transcribes audio buffer and computes 7-D emotion distribution.
        Returns: (transcript, p_semantic_vector, confidence)
        """
        transcript, conf = self.transcribe(audio_data, sample_rate, initial_prompt=context_prompt)
        p_semantic = self.classify_text_emotion(transcript)
        return transcript, p_semantic, conf
