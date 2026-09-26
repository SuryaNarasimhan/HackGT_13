"""
Video Perception Pipeline for SocialLens
Detects speaker face bounding box from screen frames, classifies facial emotion
into the canonical 7-emotion simplex, and aggregates across keyframe windows.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from app.config import FRAME_SAMPLE_COUNT
from app.pipelines.taxonomy import (
    CANONICAL_EMOTIONS,
    NUM_EMOTIONS,
    dict_to_vector,
    normalize_distribution,
    softmax,
    get_top_emotion,
)

logger = logging.getLogger(__name__)


class VideoPipeline:
    """
    Perception pipeline for analyzing speaker facial expressions from screen capture frames.
    """

    def __init__(self, sample_count: int = FRAME_SAMPLE_COUNT):
        self.sample_count = sample_count
        self._cv2 = None
        self._face_cascade = None
        self._smile_cascade = None
        self._init_vision_models()

    def _init_vision_models(self):
        """Initializes OpenCV cascades if cv2 is installed."""
        try:
            import cv2
            self._cv2 = cv2
            cascades_path = cv2.data.haarcascades
            self._face_cascade = cv2.CascadeClassifier(
                cascades_path + "haarcascade_frontalface_default.xml"
            )
            self._smile_cascade = cv2.CascadeClassifier(
                cascades_path + "haarcascade_smile.xml"
            )
            logger.info("OpenCV face and smile cascades loaded successfully.")
        except ImportError:
            logger.info("OpenCV (cv2) not installed. Using pure-NumPy vision fallback.")
            self._cv2 = None
        except Exception as e:
            logger.warning(f"Error loading OpenCV cascades ({e}). Using vision fallback.")
            self._cv2 = None

    def detect_face(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """
        Detects primary face bounding box (x, y, w, h) in an RGB or BGR frame.
        Returns: (face_detected, bbox_tuple)
        """
        if frame is None or frame.size == 0:
            return False, None

        if self._cv2 is not None and self._face_cascade is not None:
            try:
                # Convert to grayscale
                if len(frame.shape) == 3:
                    gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                
                faces = self._face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(40, 40)
                )
                if len(faces) > 0:
                    # Select largest face by area
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    return True, tuple(int(v) for v in largest_face)
            except Exception as e:
                logger.warning(f"Face cascade detection failed: {e}")

        # Fallback heuristic: check if frame has reasonable dimensions & non-zero variance
        if len(frame.shape) >= 2 and frame.shape[0] >= 30 and frame.shape[1] >= 30:
            std = float(np.std(frame))
            if std > 10.0:
                h, w = frame.shape[:2]
                return True, (0, 0, w, h)

        return False, None

    def classify_face_crop(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Classifies facial emotion of a cropped face region into canonical 7-emotion simplex:
        ['joy', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'neutral']
        """
        logits = np.zeros(NUM_EMOTIONS, dtype=np.float64)
        idx_joy = CANONICAL_EMOTIONS.index("joy")
        idx_neutral = CANONICAL_EMOTIONS.index("neutral")
        idx_surprise = CANONICAL_EMOTIONS.index("surprise")
        idx_anger = CANONICAL_EMOTIONS.index("anger")
        idx_sadness = CANONICAL_EMOTIONS.index("sadness")

        if self._cv2 is not None and self._smile_cascade is not None and face_crop.size > 0:
            try:
                if len(face_crop.shape) == 3:
                    gray = self._cv2.cvtColor(face_crop, self._cv2.COLOR_BGR2GRAY)
                else:
                    gray = face_crop
                
                # Detect smile in lower half of face
                h, w = gray.shape[:2]
                lower_face = gray[int(h * 0.5) : h, :]
                
                smiles = self._smile_cascade.detectMultiScale(
                    lower_face,
                    scaleFactor=1.3,
                    minNeighbors=15,
                    minSize=(20, 20)
                )
                
                if len(smiles) > 0:
                    # Smile detected -> Joy
                    logits[idx_joy] += 3.5
                    logits[idx_surprise] += 0.5
                else:
                    # Deadpan / neutral expression (no smile)
                    logits[idx_neutral] += 3.0
                    logits[idx_sadness] += 0.5
            except Exception as e:
                logger.warning(f"Smile cascade error: {e}")
                logits[idx_neutral] += 2.5
        else:
            # Fallback neutral distribution
            logits[idx_neutral] += 2.5

        # Baseline smoothing
        logits += 0.2
        return normalize_distribution(softmax(logits))

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, Optional[Tuple[int, int, int, int]]]:
        """
        Processes a single video frame.
        Returns: (p_video_vector, face_detected, bbox)
        """
        face_found, bbox = self.detect_face(frame)
        if not face_found or bbox is None:
            # Camera off or occluded: return uniform neutral fallback
            neutral_vec = np.zeros(NUM_EMOTIONS, dtype=np.float64)
            neutral_vec[CANONICAL_EMOTIONS.index("neutral")] = 0.85
            neutral_vec[CANONICAL_EMOTIONS.index("joy")] = 0.05
            neutral_vec[CANONICAL_EMOTIONS.index("surprise")] = 0.05
            neutral_vec[CANONICAL_EMOTIONS.index("sadness")] = 0.05
            return normalize_distribution(neutral_vec), False, None

        x, y, w, h = bbox
        face_crop = frame[y : y + h, x : x + w]
        p_video = self.classify_face_crop(face_crop)
        return p_video, True, bbox

    def process_keyframes(self, frames: List[np.ndarray]) -> Tuple[np.ndarray, bool]:
        """
        Aggregates predictions across sampled keyframes (e.g., 3-5 frames over the speech window).
        Averages the probability vectors across all frames where a face was detected.
        Returns: (aggregated_p_video_vector, face_detected)
        """
        if not frames:
            neutral_vec = np.zeros(NUM_EMOTIONS, dtype=np.float64)
            neutral_vec[CANONICAL_EMOTIONS.index("neutral")] = 1.0
            return neutral_vec, False

        detected_vectors = []
        for frame in frames:
            p_vec, detected, _ = self.process_frame(frame)
            if detected:
                detected_vectors.append(p_vec)

        if not detected_vectors:
            # No face detected in any of the sampled frames
            fallback_vec = np.zeros(NUM_EMOTIONS, dtype=np.float64)
            fallback_vec[CANONICAL_EMOTIONS.index("neutral")] = 1.0
            return fallback_vec, False

        # Mean probability vector across keyframes
        mean_vec = np.mean(detected_vectors, axis=0)
        return normalize_distribution(mean_vec), True
