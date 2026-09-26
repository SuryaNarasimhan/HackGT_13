"""
Unit Tests for Rolling Conversation Memory & Two-Way Context Window
Verifies thread-safe turn tracking, FIFO eviction, and prompt formatting.
"""

import unittest
from app.conversation_memory import ConversationMemory, DialogueTurn
from app.agent.gemini_reasoner import GeminiReasoner
import numpy as np


class TestConversationMemory(unittest.TestCase):

    def setUp(self):
        self.memory = ConversationMemory(max_turns=4)

    def test_add_turns_and_fifo_eviction(self):
        """Verifies turns append and oldest turns are evicted past max_turns."""
        self.memory.add_turn(speaker="User", text="Hey, how's the presentation coming along?", emotion="joy")
        self.memory.add_turn(speaker="Other", text="It's going well, almost done with the slides.", emotion="neutral")
        self.memory.add_turn(speaker="User", text="Could you also finish the financial analysis before noon?", emotion="neutral")
        self.memory.add_turn(speaker="Other", text="Yeah, totally fine.", emotion="anger", jsd_score=0.62)

        self.assertEqual(len(self.memory), 4)

        # Add 5th turn -> should evict the 1st turn
        self.memory.add_turn(speaker="User", text="Thanks, really appreciate it!", emotion="joy")
        self.assertEqual(len(self.memory), 4)

        history = self.memory.get_history()
        self.assertEqual(history[0].speaker, "Other")
        self.assertIn("almost done with the slides", history[0].text)
        self.assertEqual(history[-1].speaker, "User")
        self.assertIn("appreciate it", history[-1].text)

    def test_formatted_history_output(self):
        """Verifies clean script formatting for AI reasoning prompt ingestion."""
        self.memory.add_turn(speaker="User", text="Can you deploy the fix to production?", emotion="neutral")
        self.memory.add_turn(speaker="Other", text="Right now?", emotion="surprise")

        formatted = self.memory.get_formatted_history()
        self.assertIn('You (User): "Can you deploy the fix to production?"', formatted)
        self.assertIn('Other Person: "Right now?"', formatted)

    def test_empty_turns_ignored(self):
        """Ensures blank or whitespace-only utterances are not saved."""
        res = self.memory.add_turn(speaker="User", text="   ")
        self.assertIsNone(res)
        self.assertEqual(len(self.memory), 0)

    def test_gemini_reasoner_with_dialogue_history(self):
        """Verifies Gemini Reasoner incorporates preceding conversational history."""
        reasoner = GeminiReasoner()
        
        preceding_context = '- You (User): "I just accidentally dropped our production staging database right before the demo." [fear]'
        transcript = "Yeah, that's just fantastic."
        
        # Deadpan Sarcasm distributions
        p_semantic = np.array([0.90, 0.02, 0.02, 0.02, 0.01, 0.01, 0.02])  # Joy
        p_audio = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.88])     # Neutral
        p_video = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.88])     # Neutral
        
        cue = reasoner.synthesize_cue(
            transcript=transcript,
            p_video=p_video,
            p_audio=p_audio,
            p_semantic=p_semantic,
            jsd_score=0.72,
            conflict_pair=("words", "tone"),
            max_conflict_value=0.78,
            is_trigger=True,
            dialogue_history=preceding_context
        )

        self.assertEqual(cue["social_cue_type"], "Dry Sarcasm / Irony")
        self.assertIn("sarcasm", cue["explanation"].lower())
        print("\n[Dialogue-Grounded Reasoner Output]:")
        print(f"Context: {preceding_context}")
        print(f"Cue: {cue['social_cue_type']} | Tip: {cue['suggested_action']}")


if __name__ == "__main__":
    unittest.main()
