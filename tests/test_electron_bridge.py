import base64

import numpy as np
import pytest

from app.electron_bridge import InputAudioCapture, decode_frame, probability_map, result_payload


def test_audio_adapter_emits_fixed_chunks_without_opening_a_device():
    capture = InputAudioCapture()
    chunks = []
    capture.register_chunk_callback(chunks.append)
    capture.start()
    capture.push(np.ones(700, dtype=np.float32))
    assert [len(chunk) for chunk in chunks] == [512, 512]
    assert np.all(chunks[1][188:] == 0)


def test_result_payload_is_json_safe_and_labels_model_scores():
    vector = np.arange(1, 8, dtype=np.float64)
    vector /= vector.sum()
    payload = result_payload({
        "transcript": "That was great.", "p_video": vector, "p_audio": vector,
        "p_semantic": vector, "jsd_score": .21, "is_trigger": True,
        "channel_status": {"face": "used", "tone": "used", "words": "used"},
        "cue_data": {"social_cue_type": "Possible sarcasm", "confidence": "Medium",
                     "explanation": "Signals differ.", "suggested_action": "Check gently."},
    }, gemini_enabled=False)
    assert payload["type"] == "result"
    assert payload["cue"]["source"] == "Local fallback"
    assert payload["channels"]["face"] == probability_map(vector)
    assert sum(payload["channels"]["words"].values()) == pytest.approx(1, abs=.001)


def test_invalid_frame_payload_is_rejected():
    with pytest.raises(ValueError):
        decode_frame(base64.b64encode(b"").decode())
