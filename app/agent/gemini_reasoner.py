"""
Cognitive Reasoning Engine using Google Gemini 2.5 Flash
Interprets cross-modal incongruence into empathetic, clear social cues.
Includes robust fallback/mock reasoning when offline or without an API key.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import logging
from typing import Any, Dict, List, Optional
import numpy as np

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.pipelines.taxonomy import (
    get_top_emotion,
    vector_to_dict,
    CANONICAL_EMOTIONS
)

logger = logging.getLogger(__name__)

# Structured JSON Schema for SocialLens Insights
SOCIAL_CUE_SCHEMA = {
    "type": "object",
    "properties": {
        "social_cue_type": {
            "type": "string",
            "enum": [
                "In Sync / Authentic",
                "Dry Sarcasm / Irony",
                "Playful Teasing",
                "Concealed Frustration",
                "Polite Agreement",
                "Understated Humor",
                "Defensive Hesitation",
                "Ambiguous"
            ],
            "description": "Category of the detected social subtext or cue."
        },
        "confidence": {
            "type": "string",
            "enum": ["High", "Medium", "Low"],
            "description": "Confidence level in the social cue interpretation."
        },
        "explanation": {
            "type": "string",
            "description": "1-2 concise, empathetic sentences explaining the cross-modal mismatch."
        },
        "suggested_action": {
            "type": "string",
            "description": "A supportive, actionable suggestion for how the user can respond."
        }
    },
    "required": ["social_cue_type", "confidence", "explanation", "suggested_action"]
}


class GeminiReasoner:
    """
    Cognitive reasoner that synthesizes transcript, vocal tone, facial expression,
    and divergence scores into actionable social cue insights for neurodivergent users.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = GEMINI_MODEL):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        self.client = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="GeminiCallWorker")
        self._init_client()

    def _init_client(self):
        """Initializes the google-genai client if installed and configured."""
        if not self.api_key:
            logger.info("No GEMINI_API_KEY found. Reasoner running in intelligent fallback mode.")
            return

        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"Initialized Google Gemini client with model: {self.model_name}")
        except ImportError:
            logger.warning(
                "google-genai package not found. Running in intelligent fallback mode."
            )
            self.client = None
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini client: {e}. Running in fallback mode.")
            self.client = None

    def synthesize_cue(
        self,
        transcript: str,
        p_video: np.ndarray,
        p_audio: np.ndarray,
        p_semantic: np.ndarray,
        jsd_score: float,
        conflict_pair: Optional[tuple] = None,
        max_conflict_value: float = 0.0,
        is_trigger: Optional[bool] = None,
        dialogue_history: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Synthesizes multimodal cue information.
        Attempts live Gemini 2.5 Flash structured generation first,
        falling back seamlessly to rule-based heuristics if offline or timed out (>4s).
        """
        if is_trigger is None:
            is_trigger = bool(jsd_score >= 0.40 or max_conflict_value >= 0.65)

        if self.client:
            try:
                future = self._executor.submit(
                    self._call_gemini,
                    transcript=transcript,
                    p_video=p_video,
                    p_audio=p_audio,
                    p_semantic=p_semantic,
                    jsd_score=jsd_score,
                    conflict_pair=conflict_pair,
                    max_conflict_value=max_conflict_value,
                    is_trigger=is_trigger,
                    dialogue_history=dialogue_history
                )
                return future.result(timeout=4.0)
            except Exception as e:
                logger.warning(f"Gemini API call timed out or failed ({e}). Reverting to fallback reasoner.")

        # Fallback heuristic reasoner
        return self._fallback_heuristic_reasoner(
            transcript=transcript,
            p_video=p_video,
            p_audio=p_audio,
            p_semantic=p_semantic,
            jsd_score=jsd_score,
            conflict_pair=conflict_pair,
            max_conflict_value=max_conflict_value,
            is_trigger=is_trigger,
            dialogue_history=dialogue_history
        )

    def _build_prompt(
        self,
        transcript: str,
        p_video: np.ndarray,
        p_audio: np.ndarray,
        p_semantic: np.ndarray,
        jsd_score: float,
        conflict_pair: Optional[tuple],
        max_conflict_value: float,
        is_trigger: bool,
        dialogue_history: Optional[str] = None
    ) -> str:
        """Constructs an empathetic, context-rich prompt for Gemini with multi-turn dialogue history."""
        top_face, face_prob = get_top_emotion(p_video)
        top_tone, tone_prob = get_top_emotion(p_audio)
        top_words, words_prob = get_top_emotion(p_semantic)

        conflict_desc = ""
        if conflict_pair:
            conflict_desc = (
                f"- Primary Friction: Channel '{conflict_pair[0]}' conflicts with '{conflict_pair[1]}' "
                f"(Pairwise Divergence: {max_conflict_value:.2f})"
            )

        history_section = ""
        if dialogue_history and dialogue_history.strip() and not dialogue_history.startswith("(No previous"):
            history_section = f"""
RECENT CONVERSATION HISTORY (Context from both parties):
{dialogue_history.strip()}
""".strip()

        if not is_trigger or jsd_score < 0.35:
            context_header = (
                "Spoken words, vocal prosody, and facial expression are in HARMONY. "
                "No significant cross-modal mismatch or subtext detected."
            )
            task_rules = """
- Categorize as 'In Sync / Authentic' or 'Polite Agreement'.
- Confirm why the channels align in the context of the conversation.
- Provide a brief, supportive tip confirming the user can take the statement at face value.
""".strip()
        else:
            context_header = (
                "A cross-modal incongruence was detected between what was said, how it was said, and facial expression."
            )
            task_rules = """
- Consider the conversational context (what the user or other person said before) to interpret the subtext accurately.
- Focus on why the contrast between channels indicates sarcasm, teasing, concealed frustration, or polite masking.
- Avoid judgmental or pathologizing language. Present interpretations as supportive possibilities.
- Provide a brief, practical tip for how the user can comfortably respond or navigate this moment.
""".strip()

        history_block = f"\n{history_section}\n" if history_section else ""

        return f"""
You are SocialLens, an empathetic assistive communication assistant designed to support neurodivergent individuals during video calls.
{context_header}
{history_block}
CURRENT UTTERANCE UNDER ANALYSIS:
- Spoken Words: "{transcript}"
- Semantic Sentiment: Top='{top_words}' ({words_prob:.0%}) | Full={vector_to_dict(p_semantic)}
- Vocal Tone / Prosody: Top='{top_tone}' ({tone_prob:.0%}) | Full={vector_to_dict(p_audio)}
- Facial Expression: Top='{top_face}' ({face_prob:.0%}) | Full={vector_to_dict(p_video)}
- Cross-Modal Divergence (JSD): {jsd_score:.3f} (0=identical, 1=completely conflicting)
{conflict_desc}

TASK:
Interpret the social context for the user:
{task_rules}
- You must output valid JSON matching the required schema.
""".strip()

    def _call_gemini(
        self,
        transcript: str,
        p_video: np.ndarray,
        p_audio: np.ndarray,
        p_semantic: np.ndarray,
        jsd_score: float,
        conflict_pair: Optional[tuple],
        max_conflict_value: float,
        is_trigger: bool,
        dialogue_history: Optional[str] = None
    ) -> Dict[str, str]:
        """Calls Gemini with structured JSON output configuration."""
        from google.genai import types

        prompt = self._build_prompt(
            transcript=transcript,
            p_video=p_video,
            p_audio=p_audio,
            p_semantic=p_semantic,
            jsd_score=jsd_score,
            conflict_pair=conflict_pair,
            max_conflict_value=max_conflict_value,
            is_trigger=is_trigger,
            dialogue_history=dialogue_history
        )

        config = types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=SOCIAL_CUE_SCHEMA,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            system_instruction=(
                "You are an empathetic social communication assistant for neurodivergent individuals. "
                "Output concise, highly supportive JSON explaining non-verbal social cues with conversational context."
            )
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )

        raw_text = response.text.strip() if response.text else ""
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_text = "\n".join(lines).strip()

        data = json.loads(raw_text)
        return self._validate_response(data, transcript)

    def _fallback_heuristic_reasoner(
        self,
        transcript: str,
        p_video: np.ndarray,
        p_audio: np.ndarray,
        p_semantic: np.ndarray,
        jsd_score: float,
        conflict_pair: Optional[tuple],
        max_conflict_value: float,
        is_trigger: bool = True,
        dialogue_history: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Rule-based heuristic reasoner that mirrors Gemini's output schema when offline.
        Uses affective incongruence rules (Mehrabian principle) and handles in-sync communication.
        """
        top_face, _ = get_top_emotion(p_video)
        top_tone, _ = get_top_emotion(p_audio)
        top_words, _ = get_top_emotion(p_semantic)

        # Baseline: In-Sync / Congruent communication
        if not is_trigger or jsd_score < 0.35:
            return {
                "social_cue_type": "In Sync / Authentic",
                "confidence": "High",
                "explanation": (
                    f"Spoken words, vocal tone ({top_tone}), and facial expression ({top_face}) are aligned in harmony. "
                    "No conflicting subtext detected; communication is direct and genuine."
                ),
                "suggested_action": "Take the message at face value and engage naturally with confidence.",
                "transcript": transcript
            }

        # Rule 1: Words express Joy/Praise, but Tone/Face are Neutral or Negative -> Sarcasm/Irony
        if top_words in ["joy", "surprise"] and (top_tone in ["neutral", "disgust", "anger"] or top_face in ["neutral", "disgust"]):
            return {
                "social_cue_type": "Dry Sarcasm / Irony",
                "confidence": "High" if jsd_score > 0.40 else "Medium",
                "explanation": (
                    f"Spoken words express {top_words}, but vocal tone is {top_tone} and expression is {top_face}. "
                    "This contrast often signals dry sarcasm, irony, or self-deprecating humor."
                ),
                "suggested_action": "Acknowledge the underlying irony or setback lightly rather than taking the literal praise at face value.",
                "transcript": transcript
            }

        # Rule 2: Words are Polite/Neutral, but Tone or Face indicates Anger/Frustration -> Concealed Frustration
        if top_words in ["neutral", "joy"] and (top_tone in ["anger", "disgust"] or top_face in ["anger", "disgust"]):
            return {
                "social_cue_type": "Concealed Frustration",
                "confidence": "High" if max_conflict_value > 0.60 else "Medium",
                "explanation": (
                    f"While the words seem {top_words}, subtle vocal tension ({top_tone}) and facial cues ({top_face}) "
                    "suggest mild frustration or unspoken stress."
                ),
                "suggested_action": "Offer reassurance or gently invite them to share if they have concerns ('Is everything looking alright on your end?').",
                "transcript": transcript
            }

        # Rule 3: Words indicate Negative/Fear, but Tone or Face is Joyful -> Playful Teasing / Banter
        if top_words in ["anger", "fear", "sadness"] and (top_tone == "joy" or top_face == "joy"):
            return {
                "social_cue_type": "Playful Teasing",
                "confidence": "Medium",
                "explanation": (
                    f"The speaker used words denoting {top_words}, but their smiling expression ({top_face}) "
                    f"and upbeat tone ({top_tone}) indicate playful banter rather than genuine hostility."
                ),
                "suggested_action": "Respond in a friendly, lighthearted tone; they are likely engaging in friendly teasing.",
                "transcript": transcript
            }

        # Rule 4: General High Divergence
        return {
            "social_cue_type": "Understated Humor",
            "confidence": "Low",
            "explanation": (
                f"There is a notable difference between their words ({top_words}), tone ({top_tone}), "
                f"and facial expression ({top_face}), which may indicate understated humor or mixed feelings."
            ),
            "suggested_action": "Give a brief pause or check in with a clarifying question to confirm their intent.",
            "transcript": transcript
        }

    def _validate_response(self, data: Dict[str, Any], transcript: str = "") -> Dict[str, str]:
        """Ensures all required keys are present and clean strings."""
        return {
            "social_cue_type": str(data.get("social_cue_type", "In Sync / Authentic")),
            "confidence": str(data.get("confidence", "Medium")),
            "explanation": str(data.get("explanation", "Spoken words, tone, and expression were analyzed.")),
            "suggested_action": str(data.get("suggested_action", "Proceed naturally.")),
            "transcript": transcript or str(data.get("transcript", ""))
        }
