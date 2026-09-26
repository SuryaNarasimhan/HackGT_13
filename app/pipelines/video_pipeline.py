"""
Video Perception Pipeline for SocialLens
Detects speaker face bounding box from screen frames, classifies facial emotion
into the canonical 7-emotion simplex using a Facial Emotion Recognition (FER) model,
and aggregates across keyframe windows.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from app.config import FRAME_SAMPLE_COUNT, VIDEO_EMOTION_MODEL
from app.pipelines.speaker_baseline import (
    STATUS_NO_FACE,
    STATUS_UNAVAILABLE,
    STATUS_USED,
    SpeakerBaseline,
)
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

# Mapping from common FER model label names to the canonical 7-emotion simplex
FER_LABEL_TO_CANONICAL: Dict[str, str] = {
    "happy": "joy",
    "happiness": "joy",
    "joy": "joy",
    "sad": "sadness",
    "sadness": "sadness",
    "angry": "anger",
    "anger": "anger",
    "surprise": "surprise",
    "surprised": "surprise",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "scared": "fear",
    "neutral": "neutral",
}


class VideoPipeline:
    """
    Perception pipeline for analyzing speaker facial expressions from screen capture frames.
    Uses OpenCV for face localization and a Deep Learning FER model for 7-class emotion classification.
    """

    def __init__(
        self,
        model_name: str = VIDEO_EMOTION_MODEL,
        sample_count: int = FRAME_SAMPLE_COUNT,
        lazy_load: bool = True,
    ):
        self.model_name = model_name
        self.sample_count = sample_count
        self._cv2 = None
        self._face_cascade = None
        self._emotion_classifier = None

        self._init_vision_models()
        if not lazy_load:
            self._load_emotion_classifier()

    def _init_vision_models(self):
        """Initializes OpenCV face detector cascade if cv2 is installed."""
        try:
            import cv2
            self._cv2 = cv2
            cascades_path = cv2.data.haarcascades
            self._face_cascade = cv2.CascadeClassifier(
                cascades_path + "haarcascade_frontalface_default.xml"
            )
            logger.info("OpenCV face cascade loaded successfully.")
        except ImportError:
            logger.info("OpenCV (cv2) not installed. Using pure-NumPy vision fallback.")
            self._cv2 = None
        except Exception as e:
            logger.warning(f"Error loading OpenCV cascades ({e}). Using vision fallback.")
            self._cv2 = None

    def _load_emotion_classifier(self):
        """Loads Hugging Face FER image classification model lazily."""
        if self._emotion_classifier is not None:
            return
        try:
            from transformers import pipeline
            logger.info(f"Loading facial emotion recognition model ({self.model_name})...")
            self._emotion_classifier = pipeline(
                "image-classification",
                model=self.model_name,
                top_k=None,
                device=-1  # CPU for predictable latency
            )
            logger.info("Facial emotion recognition model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load FER model ({e}). Video pipeline will use fallback distribution."
            )
            self._emotion_classifier = None

    def detect_face(
        self,
        frame: np.ndarray,
        color_order: str = "RGB"
    ) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """
        Detects primary face bounding box (x, y, w, h) in an RGB or BGR frame.
        Uses OpenCV Haar cascades with histogram equalization when cv2 is available,
        falling back to a biometric skin-color and aspect-ratio geometry validator only
        when the cascade cannot run.
        Returns: (face_detected, bbox_tuple)
        """
        if frame is None or frame.size == 0:
            return False, None

        if self._cv2 is not None and self._face_cascade is not None:
            try:
                # Convert to grayscale matching the input color order
                if len(frame.shape) == 3:
                    if color_order == "RGB":
                        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2GRAY)
                    else:
                        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame

                # Histogram equalization normalizes lighting differences across webcam feeds
                if len(gray.shape) == 2:
                    gray_eq = self._cv2.equalizeHist(gray)
                else:
                    gray_eq = gray
                
                faces = self._face_cascade.detectMultiScale(
                    gray_eq,
                    scaleFactor=1.08,
                    minNeighbors=3,
                    minSize=(30, 30)
                )
                if len(faces) == 0:
                    # Trust the detector's "no face". The skin-color fallback also matches beige
                    # walls, wood and noise, and a wrong face is worse than a missing one.
                    return False, None
                # Select largest face by area
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                return True, tuple(int(v) for v in largest_face)
            except Exception as e:
                logger.warning(f"Face cascade detection failed: {e}")

        # Biometric fallback when the cascade cannot run: checks for human skin cluster and
        # face aspect-ratio geometry. Replaces fragile `std > 10` that falsely flagged
        # desktop UI/wallpaper as faces.
        return self._detect_face_heuristic(frame, color_order=color_order)

    def _detect_face_heuristic(
        self,
        frame: np.ndarray,
        color_order: str = "RGB"
    ) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """
        Robust non-cascade fallback that detects face presence based on human skin chrominance
        clustering and elliptical face aspect ratio (0.6 to 2.2).
        Prevents non-face desktop text, buttons, and backgrounds from triggering false face detections.
        """
        if frame is None or len(frame.shape) < 3 or frame.shape[0] < 30 or frame.shape[1] < 30:
            return False, None

        img = frame[:, :, :3]
        if color_order == "BGR":
            b = img[:, :, 0].astype(np.int32)
            g = img[:, :, 1].astype(np.int32)
            r = img[:, :, 2].astype(np.int32)
        else:
            r = img[:, :, 0].astype(np.int32)
            g = img[:, :, 1].astype(np.int32)
            b = img[:, :, 2].astype(np.int32)

        # Standard daylight human skin chrominance rules
        skin_mask = (
            (r > 70) &
            (g > 35) &
            (b > 15) &
            (r > g) &
            (r > b) &
            (np.maximum(r, np.maximum(g, b)) - np.minimum(r, np.minimum(g, b)) > 15) &
            (np.abs(r - g) > 12)
        )

        skin_count = int(np.sum(skin_mask))
        total_pixels = frame.shape[0] * frame.shape[1]
        skin_ratio = skin_count / total_pixels

        # Require at least 4% skin pixel density to avoid small random noise
        if skin_ratio < 0.04:
            return False, None

        # Find bounding box around skin pixel cluster
        ys, xs = np.where(skin_mask)
        if len(ys) == 0 or len(xs) == 0:
            return False, None

        y_min, y_max = int(np.min(ys)), int(np.max(ys))
        x_min, x_max = int(np.min(xs)), int(np.max(xs))
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        if bbox_w < 25 or bbox_h < 25:
            return False, None

        # Human faces generally have height/width aspect ratios between 0.6 and 2.2
        aspect_ratio = bbox_h / float(bbox_w)
        if 0.5 <= aspect_ratio <= 2.5:
            return True, (x_min, y_min, bbox_w, bbox_h)

        return False, None

    def classify_face_crop(
        self,
        face_crop: np.ndarray,
        color_order: str = "RGB"
    ) -> np.ndarray:
        """
        Classifies facial emotion of a cropped face region into canonical 7-emotion simplex:
        ['joy', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'neutral']
        Guarantees genuine RGB color channel orientation so models receive accurate skin & lip tones.
        """
        if face_crop is None or face_crop.size == 0:
            return self._get_fallback_distribution()

        self._load_emotion_classifier()

        if self._emotion_classifier is not None:
            try:
                from PIL import Image

                # Ensure 3-channel RGB format for Pillow / FER Transformer
                if len(face_crop.shape) == 2:
                    rgb_crop = np.stack([face_crop] * 3, axis=-1)
                elif face_crop.shape[2] == 4:
                    rgb_crop = face_crop[:, :, :3]
                else:
                    rgb_crop = face_crop.copy()

                if color_order == "BGR":
                    # Convert explicit BGR input to RGB
                    if self._cv2 is not None:
                        rgb_crop = self._cv2.cvtColor(rgb_crop, self._cv2.COLOR_BGR2RGB)
                    else:
                        rgb_crop = rgb_crop[:, :, ::-1]
                else:
                    # Input is already RGB.
                    # Defensive auto-check: human skin in RGB has Red > Blue.
                    # If Blue is heavily dominant over Red on bright pixels,
                    # the caller likely passed BGR by mistake; auto-swap to protect the FER model.
                    if rgb_crop.shape[2] == 3 and rgb_crop.size > 150:
                        r_mean = float(np.mean(rgb_crop[:, :, 0]))
                        b_mean = float(np.mean(rgb_crop[:, :, 2]))
                        if b_mean > 60 and b_mean > r_mean * 1.4:
                            logger.debug("Detected inverted BGR channels in face crop; auto-correcting to RGB.")
                            rgb_crop = rgb_crop[:, :, ::-1]

                # Clip safely to uint8 range
                rgb_crop = np.clip(rgb_crop, 0, 255).astype(np.uint8)

                pil_image = Image.fromarray(rgb_crop)
                predictions = self._emotion_classifier(pil_image)

                # Initialize distribution vector
                vec = np.zeros(NUM_EMOTIONS, dtype=np.float64)

                for item in predictions:
                    raw_label = str(item.get("label", "")).strip().lower()
                    score = float(item.get("score", 0.0))
                    canonical = FER_LABEL_TO_CANONICAL.get(raw_label)
                    if canonical and canonical in EMOTION_TO_IDX:
                        vec[EMOTION_TO_IDX[canonical]] += score

                # If scores were populated, return normalized simplex
                if np.sum(vec) > 1e-6:
                    return normalize_distribution(vec)

            except Exception as e:
                logger.warning(f"FER inference failed ({e}). Using fallback distribution.")

        return self._get_fallback_distribution()

    def _get_fallback_distribution(self) -> np.ndarray:
        """Graceful fallback distribution when model is unavailable or inference fails."""
        vec = np.zeros(NUM_EMOTIONS, dtype=np.float64)
        vec[CANONICAL_EMOTIONS.index("neutral")] = 0.70
        vec[CANONICAL_EMOTIONS.index("sadness")] = 0.05
        vec[CANONICAL_EMOTIONS.index("joy")] = 0.05
        vec[CANONICAL_EMOTIONS.index("surprise")] = 0.05
        vec[CANONICAL_EMOTIONS.index("anger")] = 0.05
        vec[CANONICAL_EMOTIONS.index("fear")] = 0.05
        vec[CANONICAL_EMOTIONS.index("disgust")] = 0.05
        return normalize_distribution(vec)

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

    def process_keyframes_relative(
        self,
        frames: List[np.ndarray],
        baseline: SpeakerBaseline
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Aggregates keyframes and compares the expression with this speaker's usual face.
        Returns (p_display, p_compare, status). No face means the channel is missing, not a
        deadpan face. p_compare holds the emotions stronger than usual when status is
        STATUS_USED; any other status means the face must not count.
        """
        p_video, face_detected = self.process_keyframes(frames)
        if not face_detected:
            return p_video, p_video, STATUS_NO_FACE

        if self._emotion_classifier is None:
            # Without the FER model every face reads as the same placeholder distribution
            return p_video, p_video, STATUS_UNAVAILABLE

        status, p_relative = baseline.compare_and_update(p_video)
        return p_video, (p_relative if status == STATUS_USED else p_video), status
