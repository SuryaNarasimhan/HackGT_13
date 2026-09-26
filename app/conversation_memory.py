"""
Rolling Conversational Dialogue Memory for SocialLens
Tracks multi-turn context between the local user and remote video call participants.
Enables conversational grounding for social subtext and actionable suggestions.
"""

from collections import deque
from dataclasses import dataclass, field
import logging
import threading
import time
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DialogueTurn:
    """Represents a single conversational turn from either the User or Other speaker."""
    speaker: str  # "User" or "Other"
    text: str
    timestamp: float = field(default_factory=time.time)
    emotion: Optional[str] = None
    jsd_score: Optional[float] = None

    def formatted(self) -> str:
        """Formatted string for reasoning prompts."""
        label = "You (User)" if self.speaker.lower() == "user" else "Other Person"
        emotion_tag = f" [{self.emotion}]" if self.emotion else ""
        return f"- {label}: \"{self.text}\"{emotion_tag}"


class ConversationMemory:
    """
    Thread-safe circular queue holding the last N dialogue turns.
    Allows Gemini Reasoner to understand contextual intent across turns.
    """

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._history: deque[DialogueTurn] = deque(maxlen=max_turns)
        self._lock = threading.Lock()

    def add_turn(
        self,
        speaker: str,
        text: str,
        emotion: Optional[str] = None,
        jsd_score: Optional[float] = None
    ) -> DialogueTurn:
        """Appends a new turn into the rolling history."""
        cleaned_text = text.strip()
        if not cleaned_text:
            return None

        turn = DialogueTurn(
            speaker=speaker,
            text=cleaned_text,
            timestamp=time.time(),
            emotion=emotion,
            jsd_score=jsd_score
        )
        with self._lock:
            self._history.append(turn)

        logger.debug(f"Dialogue Memory [{speaker}]: {cleaned_text[:40]}...")
        return turn

    def get_history(self) -> List[DialogueTurn]:
        """Returns a snapshot of the current dialogue turns."""
        with self._lock:
            return list(self._history)

    def get_formatted_history(self, exclude_last: bool = False) -> str:
        """
        Formats dialogue history into a clean script for prompt ingestion.
        Optionally excludes the most recent turn if it is currently under analysis.
        """
        with self._lock:
            turns = list(self._history)

        if exclude_last and len(turns) > 0:
            turns = turns[:-1]

        if not turns:
            return "(No previous conversational turns recorded yet)"

        return "\n".join(turn.formatted() for turn in turns)

    def clear(self):
        """Resets conversational memory."""
        with self._lock:
            self._history.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._history)
