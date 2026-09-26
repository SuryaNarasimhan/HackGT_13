"""
Multi-Distribution Jensen-Shannon Divergence (JSD) Engine for SocialLens
Computes cross-modal divergence across Video, Audio, and Semantic emotion vectors.
"""

from typing import Dict, Tuple, Union
import numpy as np
from app.config import JSD_EPSILON, JSD_THRESHOLD, JSD_PAIRWISE_THRESHOLD
from app.pipelines.taxonomy import normalize_distribution, NUM_EMOTIONS


def shannon_entropy(p: np.ndarray, eps: float = JSD_EPSILON) -> float:
    """
    Computes Shannon entropy H(P) using base-2 logarithm.
    H(P) = -sum(p_i * log2(p_i))
    
    Uses epsilon-clamping to ensure numerical stability and prevent log2(0).
    """
    p_norm = normalize_distribution(p)
    # Clip probabilities to avoid log2(0)
    p_safe = np.clip(p_norm, eps, 1.0)
    # Re-normalize slightly after clipping
    p_safe = p_safe / np.sum(p_safe)
    return float(-np.sum(p_safe * np.log2(p_safe)))


def compute_pairwise_jsd(
    p: Union[np.ndarray, list],
    q: Union[np.ndarray, list],
    eps: float = JSD_EPSILON
) -> float:
    """
    Computes pairwise Jensen-Shannon Divergence between two distributions P and Q:
        M = 0.5 * (P + Q)
        JSD(P, Q) = H(M) - 0.5 * [H(P) + H(Q)]
        
    Bounded strictly in [0, 1] when using base-2 logarithm.
    """
    p_arr = normalize_distribution(p)
    q_arr = normalize_distribution(q)
    
    m = 0.5 * (p_arr + q_arr)
    h_m = shannon_entropy(m, eps)
    h_p = shannon_entropy(p_arr, eps)
    h_q = shannon_entropy(q_arr, eps)
    
    jsd = h_m - 0.5 * (h_p + h_q)
    # Numerical clip to exactly [0, 1]
    return float(np.clip(jsd, 0.0, 1.0))


def compute_tri_modal_jsd(
    p_v: Union[np.ndarray, list],
    p_a: Union[np.ndarray, list],
    p_s: Union[np.ndarray, list],
    normalized: bool = True,
    eps: float = JSD_EPSILON
) -> float:
    """
    Computes generalized Jensen-Shannon Divergence across three distributions:
        M = (1/3) * (P_v + P_a + P_s)
        JSD_raw = H(M) - (1/3) * [H(P_v) + H(P_a) + H(P_s)]
        
    Theoretical maximum of JSD_raw for 3 distributions using base-2 log is log2(3) ~ 1.58496.
    When normalized=True, the result is divided by log2(3), bounding it strictly in [0.0, 1.0].
    """
    v = normalize_distribution(p_v)
    a = normalize_distribution(p_a)
    s = normalize_distribution(p_s)
    
    m = (v + a + s) / 3.0
    h_m = shannon_entropy(m, eps)
    h_v = shannon_entropy(v, eps)
    h_a = shannon_entropy(a, eps)
    h_s = shannon_entropy(s, eps)
    
    jsd_raw = h_m - (h_v + h_a + h_s) / 3.0
    
    if normalized:
        # Maximum possible JSD for 3 distributions in base-2 is log2(3)
        max_jsd = np.log2(3.0)
        jsd_norm = jsd_raw / max_jsd
        return float(np.clip(jsd_norm, 0.0, 1.0))
    
    return float(np.clip(jsd_raw, 0.0, np.log2(3.0)))


def analyze_cross_modal_conflict(
    p_v: Union[np.ndarray, list],
    p_a: Union[np.ndarray, list],
    p_s: Union[np.ndarray, list],
    threshold: float = JSD_THRESHOLD,
    pairwise_threshold: float = JSD_PAIRWISE_THRESHOLD,
    eps: float = JSD_EPSILON
) -> Dict[str, Union[float, Dict[str, float], Tuple[str, str], bool]]:
    """
    Performs comprehensive cross-modal divergence analysis:
    - Calculates tri-modal normalized JSD.
    - Calculates all three pairwise JSDs (words vs tone, words vs face, tone vs face).
    - Identifies the pair with highest friction.
    - Determines if the trigger threshold is exceeded (either tri-modal or max-pairwise).
    """
    v = normalize_distribution(p_v)
    a = normalize_distribution(p_a)
    s = normalize_distribution(p_s)
    
    tri_jsd = compute_tri_modal_jsd(v, a, s, normalized=True, eps=eps)
    
    pw_words_tone = compute_pairwise_jsd(s, a, eps=eps)
    pw_words_face = compute_pairwise_jsd(s, v, eps=eps)
    pw_tone_face = compute_pairwise_jsd(a, v, eps=eps)
    
    pairwise = {
        "words_vs_tone": pw_words_tone,
        "words_vs_face": pw_words_face,
        "tone_vs_face": pw_tone_face
    }
    
    # Identify maximum conflict pair
    max_pair_name = max(pairwise, key=pairwise.get)
    max_conflict_val = pairwise[max_pair_name]
    pair_mapping = {
        "words_vs_tone": ("words", "tone"),
        "words_vs_face": ("words", "face"),
        "tone_vs_face": ("tone", "face")
    }
    
    # Trigger if tri-modal divergence is high OR a strong pairwise clash exists
    is_trigger = bool(tri_jsd >= threshold or max_conflict_val >= pairwise_threshold)
    
    return {
        "tri_modal_jsd": tri_jsd,
        "pairwise_jsd": pairwise,
        "max_conflict_pair": pair_mapping[max_pair_name],
        "max_conflict_value": max_conflict_val,
        "is_trigger": is_trigger
    }

