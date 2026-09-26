"""
Per-Speaker Baseline for SocialLens
Learns how one speaker usually comes across on one channel (voice or face) during a session,
so readings are compared with that person's own normal instead of with the actors the
emotion models were trained on. A consistently flat voice or rarely-smiling face is that
person's usual style, not a social signal.
"""

from collections import deque
import threading
from typing import Tuple

import numpy as np

from app.config import BASELINE_SHIFT_JSD, BASELINE_WARMUP_UTTERANCES, BASELINE_WINDOW_UTTERANCES
from app.math_engine.jsd import compute_pairwise_jsd
from app.pipelines.taxonomy import normalize_distribution

# Channel status values shared by the pipelines, coordinator, reasoner and HUD.
# Only STATUS_USED channels count toward cross-modal mismatch.
STATUS_USED = "used"                # Differs from this speaker's usual; carries information
STATUS_USUAL = "usual"              # Matches this speaker's usual delivery; carries no information
STATUS_WARMING_UP = "warming_up"    # Not enough history yet to know what is usual
STATUS_NO_FACE = "no_face"          # No face found; missing, not a "deadpan" face
STATUS_TOO_QUIET = "too_quiet"      # Too little voiced speech to judge
STATUS_NO_WORDS = "no_words"        # No usable transcript
STATUS_UNAVAILABLE = "unavailable"  # Model missing, failed, or timed out

STATUS_LABELS = {
    STATUS_USED: "different from usual",
    STATUS_USUAL: "usual for them",
    STATUS_WARMING_UP: "learning their style",
    STATUS_NO_FACE: "no face found",
    STATUS_TOO_QUIET: "not enough voice",
    STATUS_NO_WORDS: "no clear words",
    STATUS_UNAVAILABLE: "no reading",
}


class SpeakerBaseline:
    """
    Rolling record of one speaker's emotion readings on one channel, kept in memory for one session.
    Thread-safe, since utterances are processed on worker threads.
    """

    def __init__(
        self,
        warmup: int = BASELINE_WARMUP_UTTERANCES,
        window: int = BASELINE_WINDOW_UTTERANCES,
        shift_threshold: float = BASELINE_SHIFT_JSD,
    ):
        self.warmup = warmup
        self.shift_threshold = shift_threshold
        self._history = deque(maxlen=window)
        self._lock = threading.Lock()

    def compare_and_update(self, p: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Compares a reading with this speaker's usual, then adds it to their history.
        Returns (status, p_relative). When status is STATUS_USED, p_relative holds only the
        emotions that are stronger than usual; otherwise it is the reading unchanged.
        """
        p = normalize_distribution(p)
        with self._lock:
            if len(self._history) < self.warmup:
                status, p_relative = STATUS_WARMING_UP, p
            else:
                usual = normalize_distribution(np.mean(self._history, axis=0))
                if compute_pairwise_jsd(p, usual) < self.shift_threshold:
                    status, p_relative = STATUS_USUAL, p
                else:
                    status = STATUS_USED
                    p_relative = normalize_distribution(np.clip(p - usual, 0.0, None))
            self._history.append(p)
        return status, p_relative

    def reset(self) -> None:
        """Forgets this speaker's history (end of session)."""
        with self._lock:
            self._history.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._history)
