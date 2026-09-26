"""
Audio Prosody & Acoustic Tone Perception Pipeline for SocialLens
Ingests 16kHz mono audio waveform segments, analyzes acoustic prosody
(pitch variance, energy contour, harmonic stability), and classifies vocal tone
into the canonical 7-emotion discrete simplex using a deep Speech Emotion Recognition (SER) model.
"""

import logging
from typing import Dict, Optional, Tuple, Union
import numpy as np

from app.config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_EMOTION_MODEL,
    MIN_VOICED_FRAMES,
    MONOTONE_SPREAD_SEMITONES,
)
from app.pipelines.speaker_baseline import STATUS_TOO_QUIET, STATUS_USED, SpeakerBaseline
from app.pipelines.taxonomy import (
    CANONICAL_EMOTIONS,
    EMOTION_TO_IDX,
    NUM_EMOTIONS,
    dict_to_vector,
    normalize_distribution,
    softmax,
    get_top_emotion,
)

logger = logging.getLogger(__name__)

# Mapping from common SER model label names to the canonical 7-emotion simplex
SER_LABEL_TO_CANONICAL: Dict[str, str] = {
    "happy": "joy",
    "happiness": "joy",
    "joy": "joy",
    "hap": "joy",
    "sad": "sadness",
    "sadness": "sadness",
    "angry": "anger",
    "anger": "anger",
    "ang": "anger",
    "fearful": "fear",
    "fear": "fear",
    "fea": "fear",
    "disgust": "disgust",
    "disgusted": "disgust",
    "dis": "disgust",
    "surprised": "surprise",
    "surprise": "surprise",
    "sur": "surprise",
    "neutral": "neutral",
    "neu": "neutral",
    "calm": "neutral",
}


class AudioPipeline:
    """
    Acoustic Tone & Prosody Pipeline.
    Combines deep Speech Emotion Recognition (DistilHuBERT / Wav2Vec2) with
    digital signal processing (DSP) prosodic feature extraction.
    """

    def __init__(
        self,
        model_name: str = AUDIO_EMOTION_MODEL,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        lazy_load: bool = True
    ):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self._hf_pipeline = None

        if not lazy_load:
            self._load_hf_model()

    def _load_hf_model(self):
        """Loads Hugging Face audio classification model lazily if installed."""
        if self._hf_pipeline is not None:
            return
        try:
            from transformers import pipeline
            logger.info(f"Loading audio emotion model ({self.model_name})...")
            self._hf_pipeline = pipeline(
                "audio-classification",
                model=self.model_name,
                top_k=None,
                device=-1  # CPU for predictable real-time latency
            )
            logger.info("Audio classification model loaded successfully.")
        except Exception as e:
            logger.info(
                f"Audio deep model not loaded ({e}). Utilizing native DSP acoustic prosody extractor."
            )
            self._hf_pipeline = None

    def extract_prosody_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extracts fundamental acoustic prosody features using NumPy & SciPy:
        - RMS Energy: Overall loudness and vocal effort
        - Energy Variance: Expressive dynamic range vs. flat delivery
        - Pitch (F0) Estimation via normalized autocorrelation
        - Pitch Mean and Standard Deviation: Monotone vs. melodic inflection
        - Pitch Spread in semitones: voice-independent inflection used for the monotone check
        - Zero Crossing Rate (ZCR): Voice roughness / tension
        """
        if len(audio) == 0:
            return {
                "rms": 0.0,
                "energy_var": 0.0,
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "pitch_spread_st": 0.0,
                "voiced_frames": 0,
                "is_monotone": True,
                "zcr": 0.0
            }

        audio_f = audio.astype(np.float64)
        # Normalize amplitude
        max_val = np.max(np.abs(audio_f))
        if max_val > 1e-6:
            audio_f = audio_f / max_val

        # 1. Frame-based RMS Energy
        frame_len = int(0.030 * self.sample_rate)  # 30ms window
        hop_len = int(0.015 * self.sample_rate)    # 15ms hop
        
        num_frames = max(1, (len(audio_f) - frame_len) // hop_len)
        energies = []
        pitches = []

        for i in range(num_frames):
            start = i * hop_len
            frame = audio_f[start : start + frame_len]
            if len(frame) < frame_len:
                break
            
            # RMS energy
            rms = np.sqrt(np.mean(frame**2))
            energies.append(rms)

            # Pitch estimation via Autocorrelation (searching 60 Hz to 400 Hz)
            if rms > 0.02:  # Voiced frame threshold
                min_lag = int(self.sample_rate / 400.0)  # 400 Hz
                max_lag = int(self.sample_rate / 60.0)   # 60 Hz
                
                corr = np.correlate(frame, frame, mode="full")
                corr = corr[len(frame) - 1 :]
                
                if len(corr) > max_lag:
                    peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
                    if corr[peak_lag] > 0.3 * corr[0]:
                        f0 = float(self.sample_rate / peak_lag)
                        pitches.append(f0)

        rms_mean = float(np.mean(energies)) if energies else 0.0
        energy_var = float(np.std(energies)) if energies else 0.0
        
        pitch_mean = float(np.mean(pitches)) if pitches else 140.0
        pitch_std = float(np.std(pitches)) if pitches else 0.0

        # Pitch spread in semitones, so the same melody scores the same in low and high voices
        # (a fixed Hz cutoff reads low voices as flatter). Median absolute deviation instead of
        # std, because the autocorrelation tracker occasionally jumps an octave.
        semitones = 12.0 * np.log2(pitches) if pitches else np.zeros(1)
        pitch_spread_st = float(1.4826 * np.median(np.abs(semitones - np.median(semitones))))

        # Monotone detection heuristic: low pitch spread
        is_monotone = bool(pitch_spread_st < MONOTONE_SPREAD_SEMITONES)

        # Zero-Crossing Rate
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio_f)))) / 2.0)

        return {
            "rms": rms_mean,
            "energy_var": energy_var,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pitch_spread_st": pitch_spread_st,
            "voiced_frames": len(pitches),
            "is_monotone": is_monotone,
            "zcr": zcr
        }

    def classify_speech_emotion(self, audio: np.ndarray) -> np.ndarray:
        """
        Classifies speech waveform into a canonical 7-D emotion distribution:
        ['joy', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'neutral']
        Uses pre-trained SER deep learning model with fallbacks.
        """
        if len(audio) == 0:
            neutral_vec = np.zeros(NUM_EMOTIONS, dtype=np.float64)
            neutral_vec[CANONICAL_EMOTIONS.index("neutral")] = 1.0
            return neutral_vec

        self._load_hf_model()

        if self._hf_pipeline is not None:
            try:
                # Ensure float32 1D audio normalized in [-1.0, 1.0]
                audio_f32 = audio.astype(np.float32)
                max_abs = float(np.max(np.abs(audio_f32)))
                if max_abs > 1.0:
                    audio_f32 = audio_f32 / 32768.0

                outputs = self._hf_pipeline(audio_f32)
                if outputs and isinstance(outputs, list):
                    vec = np.zeros(NUM_EMOTIONS, dtype=np.float64)
                    for item in outputs:
                        raw_label = str(item.get("label", "")).strip().lower()
                        score = float(item.get("score", 0.0))
                        canonical = SER_LABEL_TO_CANONICAL.get(raw_label)
                        if canonical and canonical in EMOTION_TO_IDX:
                            vec[EMOTION_TO_IDX[canonical]] += score

                    if np.sum(vec) > 1e-6:
                        return normalize_distribution(vec)
            except Exception as e:
                logger.warning(f"HF Audio inference failed ({e}). Using DSP prosody extractor.")

        # DSP Prosody-based Emotion Classification Fallback
        features = self.extract_prosody_features(audio)
        return self._prosody_features_to_simplex(features)

    def classify_prosody(self, audio: np.ndarray) -> np.ndarray:
        """Alias for classify_speech_emotion for backward compatibility."""
        return self.classify_speech_emotion(audio)

    def _prosody_features_to_simplex(self, f: Dict[str, float]) -> np.ndarray:
        """
        Fallback heuristic mapping acoustic prosodic features into the canonical 7-simplex.
        """
        logits = np.zeros(NUM_EMOTIONS, dtype=np.float64)
        
        idx_joy = CANONICAL_EMOTIONS.index("joy")
        idx_surprise = CANONICAL_EMOTIONS.index("surprise")
        idx_sadness = CANONICAL_EMOTIONS.index("sadness")
        idx_anger = CANONICAL_EMOTIONS.index("anger")
        idx_disgust = CANONICAL_EMOTIONS.index("disgust")
        idx_fear = CANONICAL_EMOTIONS.index("fear")
        idx_neutral = CANONICAL_EMOTIONS.index("neutral")

        pitch_std = f["pitch_std"]
        rms = f["rms"]
        is_monotone = f["is_monotone"]
        zcr = f["zcr"]

        # 1. Monotone / Deadpan Delivery -> Strong Neutral
        if is_monotone:
            logits[idx_neutral] += 3.5
            logits[idx_sadness] += 0.5
        # 2. High Energy + Harsh Spectral Turbulence -> Anger
        elif rms > 0.20 and zcr > 0.12:
            logits[idx_anger] += 2.5
            logits[idx_disgust] += 1.0
        # 3. Dynamic Pitch Variance with Moderate Energy
        elif pitch_std > 25.0:
            logits[idx_surprise] += 1.5
            logits[idx_joy] += 1.2
            logits[idx_sadness] += 1.0
        else:
            logits[idx_neutral] += 2.0

        # Baseline prior so no probability is absolute zero
        logits += 0.2
        return normalize_distribution(softmax(logits))

    def process(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Processes audio waveform and returns (p_audio_vector, prosody_features).
        """
        features = self.extract_prosody_features(audio)
        p_audio = self.classify_speech_emotion(audio)
        return p_audio, features

    def process_relative(
        self,
        audio: np.ndarray,
        baseline: SpeakerBaseline
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Processes audio and compares the tone with this speaker's usual delivery.
        Returns (p_display, p_compare, prosody_features) with features["status"] set.
        p_display is the raw reading for the HUD. p_compare holds the emotions stronger than
        usual when status is STATUS_USED; any other status means the tone must not count.
        """
        p_audio, features = self.process(audio)
        if features.get("voiced_frames", 0) < MIN_VOICED_FRAMES:
            # Too little voice to judge, or to learn this speaker's style from
            features["status"] = STATUS_TOO_QUIET
            return p_audio, p_audio, features

        status, p_relative = baseline.compare_and_update(p_audio)
        features["status"] = status
        return p_audio, (p_relative if status == STATUS_USED else p_audio), features
