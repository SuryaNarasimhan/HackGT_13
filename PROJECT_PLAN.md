# Project Blueprint: "SocialLens" — Real-Time Cross-Modal Social Cue HUD
**HackGT 2026 | Meta Track: Bringing People Closer Together with AI**

---

## 1. Executive Summary & Vision

### Problem
Video calls flatten human interaction. Micro-expressions, subtle prosodic cues, sarcasm, and dry humor are easily lost or misunderstood—creating intense anxiety and cognitive fatigue for neurodivergent individuals (ASD, ADHD, social anxiety). 

### Solution
**SocialLens** is an intelligent, non-intrusive desktop overlay that serves as **"subtitles for social subtext."**
It continuously monitors a video call (screen and incoming system audio), evaluates emotion probability distributions across **three independent channels** (Video, Audio Tone, and Semantics), and computes **cross-modal incongruence** using **Jensen–Shannon Divergence (JSD)**.

When a statistically significant mismatch is detected (e.g., enthusiastic words delivered in a deadpan tone with a neutral face), the system triggers **Google Gemini 2.5 Flash** to translate the multimodal mismatch into an empathetic, clear, and actionable social cue tip.

---

## 2. System Architecture & Data Flow

```
[ Video Call (Zoom/Teams/Meet) ]
         │
         ├──► Screen Capture (mss / bettercam) ────► Video Emotion Pipeline (ONNX/DeepFace) ──┐
         │                                                                                    │
         └──► System Audio (WASAPI Loopback)                                                  │
                   │                                                                          │
                   ├──► Silero VAD (Speech Segmenter)                                         │
                   │         │                                                                │
                   │         ├──► faster-whisper (STT) ─► Semantic Emotion (RoBERTa) ─────────┼──► [ 7-D Emotion Vectors ]
                   │         │                                                                │       P_v, P_a, P_s
                   │         └──► Audio Prosody Feature Extractor (Wav2Vec2 / HuBERT) ────────┘
                   │                                                                          │
                   └──────────────────────────────────────────────────────────────────────────┘
                                                              │
                                                              ▼
                                            [ Multi-Distribution JSD Engine ]
                                               JSD(P_v, P_a, P_s) >= Tau ?
                                                              │
                                            ┌─────────────────┴─────────────────┐
                                      NO (Congruent)                       YES (Mismatch)
                                            │                                   │
                                      (Skip LLM call,                           ▼
                                      update live meters)           [ Gemini 2.5 Flash Reasoner ]
                                                                       Structured JSON Synthesis
                                                                                │
                                                                                ▼
                                                                  [ Transparent Desktop HUD ]
                                                                       (PyQt6 Overlay)
```

---

## 3. Core Tech Stack

| Layer | Technology | Role & Rationale |
| :--- | :--- | :--- |
| **GUI / Overlay** | **PyQt6** (`QWindow`, `Qt.FramelessWindowHint`, `Qt.WA_TranslucentBackground`) | Ultra-responsive desktop HUD; stays transparent, draggable, and pinned on top of any video call without window manager lag. |
| **Screen Ingestion** | **`mss`** or **`bettercam`** | 60+ FPS screen grab with near-zero CPU footprint; crops speaker video tile. |
| **System Audio Capture**| **`soundcard`** (Windows WASAPI Loopback) | Captures clean output from Zoom/speakers without requiring virtual audio cables. |
| **Voice Activity Detection** | **Silero VAD** (PyTorch / ONNX) | Lightweight, sub-millisecond voice segmentation to trigger evaluation on natural speech pauses rather than arbitrary 1s slicing. |
| **Speech-to-Text (STT)**| **`faster-whisper`** (`base.en` or `small.en`) | Transcribes speech chunks with word-level accuracy in <150ms. |
| **Semantic Emotion** | **`j-hartmann/emotion-english-distilroberta-base`** (Hugging Face / ONNX) | Ultra-fast text emotion classification into canonical classes. |
| **Audio Tone Emotion** | **`superb/hubert-base-superb-er`** or **`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`** | Captures acoustic prosody (pitch variance, energy, speaking rate). |
| **Visual Face Emotion** | **DeepFace** (Fast backend) or **AffectNet ONNX** | Detects speaker face bounding box and outputs emotion probabilities. |
| **Cognitive Reasoner** | **Google Gemini (`gemini-2.5-flash`)** | Interprets cross-modal conflicts and produces empathetic explanations via structured JSON schema. |

---

## 4. Canonical Emotion Taxonomy (Refinement A)

To mathematically compare distributions across three independent models, all outputs must be mapped to a unified 7-dimensional discrete probability simplex:

$$\mathcal{E} = \{ \text{Joy}, \text{Surprise}, \text{Sadness}, \text{Anger}, \text{Disgust}, \text{Fear}, \text{Neutral} \}$$

### Modality Projection & Normalization:
Each pipeline produces a raw probability vector:
- $P_{\text{video}} \in \mathbb{R}^7, \quad \sum_{i=1}^7 P_{\text{video}, i} = 1$
- $P_{\text{audio}} \in \mathbb{R}^7, \quad \sum_{i=1}^7 P_{\text{audio}, i} = 1$
- $P_{\text{semantic}} \in \mathbb{R}^7, \quad \sum_{i=1}^7 P_{\text{semantic}, i} = 1$

*Rule: If a pre-trained model outputs extra classes (e.g. GoEmotions with 28 classes), collapse them into the canonical 7 using a weighted mapping matrix, followed by softmax normalization.*

---

## 5. Mathematical Foundation: Multi-Distribution JSD

### Why JSD?
Kullback-Leibler (KL) divergence is unbounded ($D_{KL} \to \infty$ when $Q(x) = 0$). 
**Jensen–Shannon Divergence (JSD)** is symmetric, smooth, and strictly bounded between $[0, 1]$ (when using base-2 logarithm).

### Generalized Formulation for 3 Distributions:
Given $P_V$ (video), $P_A$ (audio), and $P_S$ (semantic):

$$M = \frac{1}{3}(P_V + P_A + P_S)$$

$$JSD(P_V, P_A, P_S) = H(M) - \frac{1}{3}\Big[H(P_V) + H(P_A) + H(P_S)\Big]$$

Where Shannon Entropy $H(P)$ is defined as:
$$H(P) = -\sum_{i=1}^{7} p_i \log_2(p_i + \epsilon)$$
*(with $\epsilon = 10^{-12}$ for numerical stability)*.

### Gating Threshold:
- Let $\tau \approx 0.40 - 0.48$ (calibrated during testing).
- **If $JSD < \tau$:** The channels are congruent (e.g., both smiling and saying "Great job!" with happy tone). **No LLM invocation.** Live meters update quietly.
- **If $JSD \ge \tau$:** High cross-modal friction detected. **Trigger Gemini reasoning.**

---

## 6. Execution Pipeline & Cadence (Refinement B & C)

1. **Continuous Audio Loopback (`soundcard`)**:
   - Audio stream captures 16kHz mono audio from Windows default speaker output.
   - Audio feeds into a rolling 8-second circular buffer.
2. **VAD Chunking (Silero VAD)**:
   - When a pause of $\ge 500\text{ ms}$ is detected following speech, finalize the utterance segment.
3. **Concurrent Parallel Evaluation**:
   - `Task 1 (STT + Semantics)`: Transcribe audio segment via `faster-whisper`, pass transcript to RoBERTa $\to P_S$.
   - `Task 2 (Audio Prosody)`: Pass raw audio waveform segment to HuBERT/Wav2Vec2 $\to P_A$.
   - `Task 3 (Video Frames)`: Sample 3–5 representative frames during the speech window from the screen capture, detect face, run emotion ONNX $\to$ average $\to P_V$.
4. **Divergence Check**:
   - Compute $JSD(P_V, P_A, P_S)$.
5. **Gated Gemini Synthesis**:
   - If $JSD \ge \tau$, send payload to Gemini.
   - Update PyQt6 overlay signals via thread-safe Qt Signals.

---

## 7. Gemini Reasoning Schema & Prompt (Refinement D)

### Model:
`gemini-2.5-flash` with `response_mime_type="application/json"`.

### Prompt Template:
```text
You are SocialLens, an empathetic assistive communication assistant helping a neurodivergent person read subtle social cues during a live conversation.

A multimodal incongruence was detected between the speaker's words, voice tone, and facial expression:
- Spoken Words: "{transcript}"
- Semantic Sentiment Distribution: {p_semantic}
- Vocal Acoustic Tone Distribution: {p_audio}
- Facial Expression Distribution: {p_video}
- Cross-Modal Divergence Score (JSD): {jsd_score:.3f} (Scale: 0=identical, 1=completely conflicting)

Explain the social subtext concisely and constructively. Avoid pathologizing or accusatory language. Do not state assumptions as hard facts; offer supportive interpretations.
```

### Response JSON Schema:
```json
{
  "social_cue_type": "Dry Sarcasm / Irony | Playful Teasing | Concealed Frustration | Polite Agreement | Understated Humor | Ambiguous",
  "confidence": "High | Medium | Low",
  "explanation": "Spoken words express enthusiasm, but flat vocal pitch and a neutral deadpan face suggest sarcastic banter rather than genuine praise.",
  "suggested_action": "Acknowledge the shared frustration lightly or reply with playful humor."
}
```

---

## 8. Desktop HUD User Interface (PyQt6)

### Visual Style:
* **Glassmorphism / Dark Modern**: Semi-transparent dark slate backdrop (`rgba(20, 24, 33, 0.85)`), rounded corners (16px), subtle border glow.
* **Always-On-Top & Draggable**: Floats seamlessly on top of Zoom/Google Meet.
* **Minimalist Footprint**: Compact card format (~320px width) designed not to occlude call participants.

### UI Components:
1. **Live Tri-Channel Telemetry**:
   - Three mini progress bars displaying primary detected emotion and confidence for **Face**, **Tone**, and **Words**.
2. **Incongruence Radar / Meter**:
   - A smooth pulsing meter indicating JSD score:
     - Green ($0.0 - 0.35$): Balanced / Clear
     - Yellow ($0.35 - 0.48$): Subtle Nuance
     - Purple / Magenta ($> 0.48$): Social Cue Detected
3. **Subtext Insight Card**:
   - Appears when $JSD \ge \tau$ with smooth fade-in animation.
   - Displays `social_cue_type`, 1-sentence `explanation`, and `suggested_action`.
   - Auto-minimizes after 10 seconds or upon clicking "Dismiss".

---

## 9. Codebase Directory Structure

```
HackGT13/
├── app/
│   ├── __init__.py
│   ├── main.py                     # Entry point & PyQt6 application loop
│   ├── config.py                   # Thresholds, models, window lengths, API keys
│   ├── capture/
│   │   ├── __init__.py
│   │   ├── audio_loopback.py       # soundcard WASAPI loopback recorder
│   │   ├── screen_capture.py       # mss screen grabber & face ROI cropper
│   │   └── vad_detector.py         # Silero VAD speech segmenter
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── taxonomy.py             # Canonical 7 emotions definition & projections
│   │   ├── video_pipeline.py       # Face detection & emotion ONNX inference
│   │   ├── audio_pipeline.py       # Acoustic tone/prosody classifier
│   │   └── semantic_pipeline.py    # faster-whisper STT + RoBERTa text emotion
│   ├── math_engine/
│   │   ├── __init__.py
│   │   └── jsd.py                  # Multi-distribution Jensen-Shannon Divergence
│   ├── agent/
│   │   ├── __init__.py
│   │   └── gemini_reasoner.py      # Gemini 2.5 Flash client with JSON schema
│   └── ui/
│       ├── __init__.py
│       ├── overlay_window.py       # PyQt6 frameless transparent HUD
│       ├── components/             # Telemetry bars, cue cards, animations
│       └── styles.py               # Glassmorphism CSS stylesheets
├── tests/
│   ├── test_jsd.py                 # Unit tests for JSD boundary conditions
│   └── test_mock_cue.py            # Simulated mock run for demo recording
├── assets/                         # Icons, sound indicators, demo clips
├── ARCHITECTURE_PLAN.md            # This document
├── requirements.txt                # Python dependencies
└── README.md                       # Project overview & demo instructions
```

---

## 10. Step-by-Step Implementation Roadmap (36-Hour Hackathon)

### Milestone 1: Math & Core Logic (Hours 0 – 6)
- [x] Standardize 7-class emotion taxonomy (`taxonomy.py`).
- [x] Implement multi-distribution JSD calculation (`jsd.py`) and unit tests.
- [x] Integrate `google-genai` SDK and write structured prompt for Gemini 2.5 Flash (`gemini_reasoner.py`).

### Milestone 2: Audio & Video Ingestion (Hours 6 – 16)
- [x] Set up Windows WASAPI loopback capture using `soundcard`.
- [x] Connect Silero VAD to segment audio on speech pauses.
- [x] Wire `faster-whisper` and RoBERTa text emotion pipeline.
- [x] Wire audio emotion classifier on speech chunks.
- [x] Wire `mss` screen frame grabber + facial emotion classifier.

### Milestone 3: The PyQt6 Overlay UI (Hours 16 – 26)
- [x] Build transparent, frameless, draggable HUD window.
- [x] Add real-time telemetry meters (Face, Tone, Words bars).
- [x] Add dynamic cue card with glow effect when JSD triggers.
- [x] Connect worker threads via `PyQt6.QtCore.pyqtSignal` to guarantee zero UI stuttering.

### Milestone 4: Integration, Testing & Demo Polish (Hours 26 – 34)
- [x] Calibrate $\tau$ threshold using mock sarcasm scenarios (e.g. saying "Oh, that's just brilliant" with a monotone voice).
- [x] Create a built-in "Demo / Simulation Mode" that can replay a prepared sample video for judges with 100% deterministic reliability.
- [x] Polish visual styling, typography, and contrast.

### Milestone 5: Submission & Video (Hours 34 – 36)
- [x] Record 2.5-minute demo video showing side-by-side: Zoom call on left, SocialLens HUD on right catching live sarcasm and subtle cues.
- [x] Write submission documentation highlighting **Meta criteria: human connection, AI essentiality, mathematical grounding (JSD), and empathy**.
