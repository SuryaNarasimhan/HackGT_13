"""
Deterministic Hackathon Demo Runner for SocialLens
Enables reliable, reproducible live judging presentations and demo video recordings.
Provides 3 pre-configured multimodal scenarios triggered via keyboard hotkeys (1, 2, 3) or buttons.
"""

import logging
import time
from typing import Callable, Dict, List, Optional
import numpy as np

from app.agent.gemini_reasoner import GeminiReasoner
from app.config import JSD_THRESHOLD
from app.math_engine.jsd import analyze_cross_modal_conflict

logger = logging.getLogger(__name__)

# Canonical Emotion Order: ['joy', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'neutral']

DEMO_SCENARIOS = {
    1: {
        "name": "Deadpan Sarcasm",
        "description": "Positive words with monotone voice and deadpan face after an announced setback",
        "context_history": '- You (User): "I just accidentally dropped our production staging database right before the demo." [fear]',
        "transcript": "Yeah, that's just fantastic.",
        "p_semantic": np.array([0.90, 0.02, 0.02, 0.02, 0.01, 0.01, 0.02]),  # Joy
        "p_audio": np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.88]),     # Neutral / Monotone
        "p_video": np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.88]),     # Neutral / Deadpan
    },
    2: {
        "name": "Sincere Praise",
        "description": "All channels in harmony expressing genuine gratitude for completed work",
        "context_history": '- You (User): "I finished refactoring the entire audio DSP pipeline and all tests pass." [joy]',
        "transcript": "Thank you so much, this is great!",
        "p_semantic": np.array([0.88, 0.04, 0.02, 0.02, 0.01, 0.01, 0.02]),  # Joy
        "p_audio": np.array([0.82, 0.05, 0.03, 0.03, 0.02, 0.02, 0.03]),     # Joy
        "p_video": np.array([0.91, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01]),     # Joy / Smile
    },
    3: {
        "name": "Concealed Frustration",
        "description": "Polite words masking vocal tension and facial stress after an overburdening request",
        "context_history": '- You (User): "Could you also handle the 15 client tickets and the emergency deploy by 5 PM?" [neutral]',
        "transcript": "No, it's totally fine, don't worry about it.",
        "p_semantic": np.array([0.08, 0.02, 0.05, 0.05, 0.02, 0.02, 0.76]),  # Neutral / Polite
        "p_audio": np.array([0.02, 0.02, 0.02, 0.80, 0.06, 0.05, 0.03]),     # Anger / Tense
        "p_video": np.array([0.02, 0.02, 0.02, 0.05, 0.80, 0.04, 0.05]),     # Disgust / Grimace
    }
}


class MockDemoRunner:
    """
    Manages deterministic playback of conversational scenarios for demo presentations.
    """

    def __init__(
        self,
        telemetry_callback: Callable[[np.ndarray, np.ndarray, np.ndarray, float], None],
        cue_callback: Callable[[Dict[str, str]], None],
        reasoner: Optional[GeminiReasoner] = None,
        threshold: float = JSD_THRESHOLD
    ):
        self.telemetry_cb = telemetry_callback
        self.cue_cb = cue_callback
        self.reasoner = reasoner or GeminiReasoner()
        self.threshold = threshold

    def trigger_scenario(self, scenario_id: int) -> Dict:
        """
        Executes a pre-configured scenario, updates HUD telemetry, computes JSD,
        and triggers Gemini reasoning with grounded conversational dialogue history.
        """
        if scenario_id not in DEMO_SCENARIOS:
            logger.warning(f"Unknown scenario ID: {scenario_id}. Choose 1, 2, or 3.")
            return {}

        scenario = DEMO_SCENARIOS[scenario_id]
        name = scenario["name"]
        transcript = scenario["transcript"]
        context_history = scenario.get("context_history", "")
        p_v = scenario["p_video"]
        p_a = scenario["p_audio"]
        p_s = scenario["p_semantic"]

        logger.info(f"--- TRIGGERING DEMO SCENARIO [{scenario_id}]: {name} ---")

        # 1. Compute JSD divergence
        conflict_data = analyze_cross_modal_conflict(
            p_v=p_v,
            p_a=p_a,
            p_s=p_s,
            threshold=self.threshold
        )
        jsd_score = float(conflict_data["tri_modal_jsd"])
        is_trigger = bool(conflict_data["is_trigger"])

        # 2. Update HUD Telemetry
        self.telemetry_cb(p_v, p_a, p_s, jsd_score)

        # 3. Reasoner Synthesis on every scenario execution with conversational history
        try:
            cue_data = self.reasoner.synthesize_cue(
                transcript=transcript,
                p_video=p_v,
                p_audio=p_a,
                p_semantic=p_s,
                jsd_score=jsd_score,
                conflict_pair=conflict_data.get("max_conflict_pair"),
                max_conflict_value=float(conflict_data.get("max_conflict_value", 0.0)),
                is_trigger=is_trigger,
                dialogue_history=context_history
            )
        except Exception as e:
            logger.error(f"Error synthesizing cue in scenario [{scenario_id}]: {e}", exc_info=True)
            cue_data = self.reasoner._fallback_heuristic_reasoner(
                transcript=transcript,
                p_video=p_v,
                p_audio=p_a,
                p_semantic=p_s,
                jsd_score=jsd_score,
                conflict_pair=conflict_data.get("max_conflict_pair"),
                max_conflict_value=float(conflict_data.get("max_conflict_value", 0.0)),
                is_trigger=is_trigger,
                dialogue_history=context_history
            )

        self.cue_cb(cue_data)

        logger.info(
            f"Scenario [{scenario_id}] Executed | JSD: {jsd_score:.3f} | Trigger: {is_trigger}"
        )
        return {
            "scenario_id": scenario_id,
            "name": name,
            "transcript": transcript,
            "jsd_score": jsd_score,
            "is_trigger": is_trigger,
            "cue_data": cue_data
        }
