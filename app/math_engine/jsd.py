"""
Multi-Distribution Jensen-Shannon Divergence (JSD) Engine for SocialLens
Computes cross-modal divergence across Video, Audio, and Semantic emotion vectors.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from app.config import JSD_EPSILON, JSD_THRESHOLD, JSD_PAIRWISE_THRESHOLD
from app.pipelines.taxonomy import normalize_distribution, NUM_EMOTIONS

# Channel pairs in reporting order (ties resolve to the earlier pair)
CHANNEL_PAIRS = (("words", "tone"), ("words", "face"), ("tone", "face"))


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


def compute_multi_jsd(
    distributions: Sequence[Union[np.ndarray, list]],
    eps: float = JSD_EPSILON
) -> float:
    """
    Generalized Jensen-Shannon Divergence across two or more distributions:
        M = mean(P_i)
        JSD = [H(M) - mean(H(P_i))] / log2(n)
    Normalized to [0, 1]. Equals compute_pairwise_jsd for two inputs and
    compute_tri_modal_jsd(normalized=True) for three.
    """
    arrs = [normalize_distribution(d) for d in distributions]
    if len(arrs) < 2:
        return 0.0

    m = np.mean(arrs, axis=0)
    jsd_raw = shannon_entropy(m, eps) - float(np.mean([shannon_entropy(a, eps) for a in arrs]))
    return float(np.clip(jsd_raw / np.log2(len(arrs)), 0.0, 1.0))


def analyze_cross_modal_conflict(
    p_v: Union[np.ndarray, list],
    p_a: Union[np.ndarray, list],
    p_s: Union[np.ndarray, list],
    threshold: float = JSD_THRESHOLD,
    pairwise_threshold: float = JSD_PAIRWISE_THRESHOLD,
    eps: float = JSD_EPSILON,
    informative: Optional[Dict[str, bool]] = None
) -> Dict[str, Union[float, Dict[str, float], Optional[Tuple[str, str]], bool, List[str]]]:
    """
    Performs comprehensive cross-modal divergence analysis:
    - Leaves out channels marked not informative in `informative` ({"face"|"tone"|"words": bool};
      missing keys count as informative). A missing face or a delivery that matches the
      speaker's usual style carries no information and must not count as a clash.
    - Calculates the normalized JSD across the remaining channels (tri-modal when all three count).
    - Calculates the pairwise JSDs between the remaining channels and finds the strongest clash.
    - Triggers only when the words are part of the strongest clash: tone vs. face alone never triggers.
      With fewer than three channels, only the stricter pairwise threshold applies.
    """
    informative = informative or {}
    channels = {
        name: normalize_distribution(dist)
        for name, dist in (("face", p_v), ("tone", p_a), ("words", p_s))
        if informative.get(name, True)
    }

    if len(channels) < 2:
        return {
            "tri_modal_jsd": 0.0,
            "pairwise_jsd": {},
            "max_conflict_pair": None,
            "max_conflict_value": 0.0,
            "is_trigger": False,
            "channels_used": sorted(channels)
        }

    # Keeps its historical key name; covers only the channels used
    jsd = compute_multi_jsd(list(channels.values()), eps=eps)

    pairwise = {
        f"{a}_vs_{b}": compute_pairwise_jsd(channels[a], channels[b], eps=eps)
        for a, b in CHANNEL_PAIRS
        if a in channels and b in channels
    }

    # Identify maximum conflict pair
    max_pair_name = max(pairwise, key=pairwise.get)
    max_conflict_val = pairwise[max_pair_name]
    max_pair = tuple(max_pair_name.split("_vs_"))

    strong = max_conflict_val >= pairwise_threshold or (len(channels) == 3 and jsd >= threshold)
    is_trigger = bool(strong and "words" in max_pair)

    return {
        "tri_modal_jsd": jsd,
        "pairwise_jsd": pairwise,
        "max_conflict_pair": max_pair,
        "max_conflict_value": max_conflict_val,
        "is_trigger": is_trigger,
        "channels_used": sorted(channels)
    }

