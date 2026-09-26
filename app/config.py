"""
Global Configuration for SocialLens
"""

import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file if present
    BASE_DIR = Path(__file__).resolve().parent.parent
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Audio & Capture Settings
# ---------------------------------------------------------------------------
AUDIO_SAMPLE_RATE = 16000          # 16 kHz mono standard for speech models
AUDIO_CHANNELS = 1                 # Mono
AUDIO_BUFFER_SECONDS = 8.0         # Circular buffer length (seconds)
AUDIO_CHUNK_SIZE = 512             # VAD processing chunk size (samples)

# VAD (Voice Activity Detection) Parameters
VAD_SILENCE_THRESHOLD_MS = 500     # Silence duration to mark end of utterance
VAD_MIN_SPEECH_DURATION_MS = 600   # Minimum speech duration to trigger pipeline
VAD_CONFIDENCE_THRESHOLD = 0.5     # Silero VAD positive detection threshold

# Screen Capture Settings
SCREEN_FPS = 30
FRAME_SAMPLE_COUNT = 4             # Number of keyframes sampled across utterance

# ---------------------------------------------------------------------------
# Mathematics & Gating Thresholds
# ---------------------------------------------------------------------------
JSD_THRESHOLD = float(os.getenv("JSD_THRESHOLD", "0.40"))
JSD_PAIRWISE_THRESHOLD = float(os.getenv("JSD_PAIRWISE_THRESHOLD", "0.65"))
JSD_EPSILON = 1e-12                 # Numerical stability constant for log2

# Subtext Divergence Levels
JSD_LEVEL_BALANCED = 0.30          # Below this is considered congruent
JSD_LEVEL_NUANCE = 0.40            # Between 0.30 and 0.40 is subtle nuance
                                   # Above 0.40 or pairwise >= 0.65 is cue trigger


# ---------------------------------------------------------------------------
# Machine Learning & AI Models
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base.en")
TEXT_EMOTION_MODEL = os.getenv(
    "TEXT_EMOTION_MODEL",
    "j-hartmann/emotion-english-distilroberta-base"
)
AUDIO_EMOTION_MODEL = os.getenv(
    "AUDIO_EMOTION_MODEL",
    "amnesiackid/distilhubert-finetuned-ravdess"
)
VIDEO_EMOTION_MODEL = os.getenv(
    "VIDEO_EMOTION_MODEL",
    "dima806/facial_emotions_image_detection"
)

# Google Gemini Reasoner
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# ---------------------------------------------------------------------------
# UI & Overlay Styling
# ---------------------------------------------------------------------------
UI_WINDOW_WIDTH = 340
UI_WINDOW_HEIGHT = 480
UI_CARD_AUTO_DISMISS_SECONDS = 10
UI_UPDATE_INTERVAL_MS = 50          # Telemetry update rate (20 FPS)
