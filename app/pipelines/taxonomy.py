"""
Canonical 7-Emotion Taxonomy and Simplex Projection Utilities
"""

from typing import Dict, List, Tuple, Union
import numpy as np

# Canonical 7 emotions (Ekman 6 + Neutral)
CANONICAL_EMOTIONS: List[str] = [
    "joy",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "neutral",
]

NUM_EMOTIONS: int = len(CANONICAL_EMOTIONS)
EMOTION_TO_IDX: Dict[str, int] = {e: i for i, e in enumerate(CANONICAL_EMOTIONS)}
IDX_TO_EMOTION: Dict[int, str] = {i: e for i, e in enumerate(CANONICAL_EMOTIONS)}


def get_canonical_emotions() -> List[str]:
    """Returns the ordered list of canonical emotion labels."""
    return list(CANONICAL_EMOTIONS)


def emotion_to_index(emotion: str) -> int:
    """Returns index of the given emotion string (case-insensitive)."""
    norm = emotion.strip().lower()
    if norm not in EMOTION_TO_IDX:
        raise ValueError(
            f"Unknown emotion '{emotion}'. Must be one of {CANONICAL_EMOTIONS}"
        )
    return EMOTION_TO_IDX[norm]


def index_to_emotion(index: int) -> str:
    """Returns emotion name corresponding to the index."""
    if index < 0 or index >= NUM_EMOTIONS:
        raise IndexError(
            f"Index {index} out of bounds for {NUM_EMOTIONS} canonical emotions"
        )
    return IDX_TO_EMOTION[index]


def softmax(x: np.ndarray) -> np.ndarray:
    """Computes softmax over 1D numpy array with numerical stability."""
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


def normalize_distribution(vec: Union[np.ndarray, List[float]]) -> np.ndarray:
    """
    Normalizes any 7-element vector into a valid probability distribution
    on the 7-simplex: non-negative and summing to 1.0.
    """
    arr = np.asarray(vec, dtype=np.float64)
    if arr.shape != (NUM_EMOTIONS,):
        raise ValueError(f"Expected shape ({NUM_EMOTIONS},), got {arr.shape}")

    # Clip negative values
    arr = np.clip(arr, 0.0, None)
    total = np.sum(arr)
    if total <= 1e-12:
        # Fallback to uniform distribution if zero vector
        return uniform_distribution()
    return arr / total


def validate_distribution(
    dist: Union[np.ndarray, List[float]], tolerance: float = 1e-3
) -> bool:
    """Validates that a vector is a proper probability distribution on the 7-simplex."""
    arr = np.asarray(dist, dtype=np.float64)
    if arr.shape != (NUM_EMOTIONS,):
        return False
    if np.any(arr < -1e-6):
        return False
    return bool(np.isclose(np.sum(arr), 1.0, atol=tolerance))


def dict_to_vector(emotion_dict: Dict[str, float]) -> np.ndarray:
    """
    Converts a dictionary of {emotion_name: probability} into a canonical 7-D vector.
    Missing keys default to 0.0 before normalization.
    """
    vec = np.zeros(NUM_EMOTIONS, dtype=np.float64)
    for k, v in emotion_dict.items():
        norm_k = k.strip().lower()
        if norm_k in EMOTION_TO_IDX:
            vec[EMOTION_TO_IDX[norm_k]] = float(v)
    return normalize_distribution(vec)


def vector_to_dict(vec: np.ndarray) -> Dict[str, float]:
    """Converts a 7-D probability vector into an emotion -> probability dictionary."""
    norm_vec = normalize_distribution(vec)
    return {emotion: float(norm_vec[i]) for i, emotion in enumerate(CANONICAL_EMOTIONS)}


def get_top_emotion(vec: np.ndarray) -> Tuple[str, float]:
    """Returns (top_emotion_name, probability) from a 7-D probability vector."""
    norm_vec = normalize_distribution(vec)
    idx = int(np.argmax(norm_vec))
    return IDX_TO_EMOTION[idx], float(norm_vec[idx])


def uniform_distribution() -> np.ndarray:
    """Returns a uniform 7-D probability distribution (1/7 each)."""
    return np.full(NUM_EMOTIONS, 1.0 / NUM_EMOTIONS, dtype=np.float64)
