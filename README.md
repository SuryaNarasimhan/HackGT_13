# 👁️ SocialLens
### *Subtitles for Social Subtext: Bringing People Closer Together with Multimodal AI*

[![HackGT 2026](https://img.shields.io/badge/HackGT-2026-blueviolet.svg)](https://hackgt.com)
[![Meta Challenge](https://img.shields.io/badge/Meta%20Track-Bringing%20People%20Closer%20Together-0668E1.svg)](https://about.meta.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 33 Passed](https://img.shields.io/badge/Tests-33%20Passed-10B981.svg)]()

> **Built for the Meta Challenge at HackGT 2026**  
> *“Technology can bring people closer together and deepen real human connections.”*

---

## 🌟 The Problem: The Invisible Barrier in Modern Communication

Video calls have flattened human interaction. In a 2D grid of low-resolution video and compressed audio, the subtle micro-expressions, prosodic inflections, dry humor, and gentle sarcasm that grease the wheels of real-world human connection are often flattened or lost entirely.

For **neurodivergent individuals** (people with Autism Spectrum Condition, ADHD, or severe social anxiety), this communication gap is exhausting. Many spend calls in a state of high hyper-vigilance, constantly asking themselves:
* *"Did they mean that literally, or were they joking?"*
* *"They said 'I'm fine', but their voice sounded tight—did I upset them?"*
* *"Should I laugh, apologize, or say thank you?"*

This anxiety leads to video call fatigue and social withdrawal. Rather than connecting us, video calling can paradoxically make us feel more isolated.

---

## 💡 The Solution: SocialLens

**SocialLens** is a lightweight, frameless, transparent desktop HUD overlay that sits unobtrusively on top of any video call (Zoom, Google Meet, Microsoft Teams). It functions like **closed captions for social subtext**.

Instead of being a generic chatbot or a surveillance tool, SocialLens acts as an **Empathetic Social Co-Pilot**:
1. It listens to the incoming audio and monitors the speaker’s video tile in real time.
2. It breaks communication down into **three independent channels**: **Words (Semantics)**, **Voice Tone (Prosody)**, and **Facial Expression (Vision)**.
3. It measures **cross-modal incongruence** using **Jensen–Shannon Divergence (JSD)**.
4. When a statistically significant contradiction is detected (e.g. glowing words paired with a monotone voice and deadpan face), it triggers **Google Gemini 2.5 Flash** to translate the contradiction into a gentle, non-judgmental subtext tip.
5. If communication is congruent, **it stays silent**, preventing notification fatigue.

---

## 📐 The Science & Architecture: Multi-Distribution JSD

Most hackathon AI projects are simple LLM wrappers that dump text into a prompt. **SocialLens is grounded in computational social psychology.**

According to Albert Mehrabian’s pioneering communication research, human emotional intent is conveyed through three channels: **Verbal Words (7%)**, **Vocal Tone (38%)**, and **Facial Expression (55%)**. Sarcasm, passive frustration, and social masking are fundamentally defined by **incongruence** between these channels.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        LIVE VIDEO CALL (Zoom / Meet)                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
Screen Video Capture (mss)                           WASAPI Speaker Loopback
         │                                                     │
Video Emotion Pipeline                                  Silero VAD (Pause >= 500ms)
(Face Detection & Expression)                                  │
         │                                    ┌────────────────┴───────────────┐
         │                                    ▼                                ▼
         │                           Acoustic Prosody STT (faster-whisper) + RoBERTa
         │                           (F0, Pitch Variance)    (Semantic Sentiment)
         │                                    │                                │
         ▼                                    ▼                                ▼
[ P_Video in R^7 ]                   [ P_Audio in R^7 ]               [ P_Semantic in R^7 ]
  (7-Emotion Simplex)                  (7-Emotion Simplex)              (7-Emotion Simplex)
         │                                    │                                │
         └────────────────────────────────────┼────────────────────────────────┘
                                              ▼
                             [ Multi-Distribution JSD Engine ]
                                  JSD(P_v, P_a, P_s) >= Tau ?
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
               NO (In Sync)                                   YES (Cross-Modal Clash)
           (Quiet live telemetry)                                     ▼
                                                         [ Gemini 2.5 Flash Reasoner ]
                                                           Structured JSON Synthesis
                                                                      │
                                                                      ▼
                                                         [ Transparent Desktop HUD ]
                                                           (Real-Time Subtext Tip)
```

### The Generalized 3-Distribution JSD:
For three discrete emotion probability distributions $P_V, P_A, P_S$ on the 7-simplex:

$$M = \frac{1}{3}\left(P_V + P_A + P_S\right)$$

$$JSD_{\text{raw}}(P_V, P_A, P_S) = H(M) - \frac{1}{3}\Big[H(P_V) + H(P_A) + H(P_S)\Big]$$

Where Shannon Entropy $H(P) = -\sum_{i=1}^{7} p_i \log_2(p_i + \epsilon)$.  
To strictly bound the metric between $[0.0, 1.0]$, we normalize by the theoretical maximum for $N=3$:

$$JSD_{\text{norm}} = \frac{JSD_{\text{raw}}}{\log_2(3)}$$

* **In Sync ($JSD < 0.35$):** All channels agree (e.g. smiling face + enthusiastic tone + "Great job!"). Zero LLM calls. Zero distraction.
* **Friction Trigger ($JSD \ge 0.40$ or Pairwise Clash $\ge 0.65$):** Non-verbal channels conflict with verbal words (e.g. deadpan face + monotone voice + "Yeah, fantastic"). Gemini 2.5 Flash is dispatched to translate the subtext.

---

## 🏆 Alignment with Meta’s Hackathon Criteria

| Judging Criterion | How SocialLens Wins |
| :--- | :--- |
| **Meaningful Human Connection** | Instead of an AI bot replacing humans, SocialLens acts as an assistive bridge. It empowers neurodivergent people to connect with peers, colleagues, and family with confidence and reduced anxiety. |
| **Essential & Well-Integrated AI** | AI is not an afterthought; the multi-modal pipeline (DSP Prosody + Computer Vision + STT + JSD Anomaly Gating + Gemini 2.5 Flash) makes the product impossible without modern multimodal AI. |
| **Originality & Novelty** | Moves beyond speech-to-text captions by introducing **"Subtext Captions"** driven by cross-modal divergence mathematics. |
| **Strength of Working Demo** | Includes a deterministic, reliable demo mode with on-screen triggers and hotkeys (`1`, `2`, `3`) that judges can immediately test. |

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-team/SocialLens.git
cd SocialLens
```

### 2. Set Up Virtual Environment & Dependencies

**Option A: Using `uv` (Recommended — 10x faster installation):**
```bash
uv sync
```
*(Automatically creates `.venv`, provisions Python 3.12, and installs all 71 locked dependencies from `uv.lock` in seconds)*

**Option B: Using standard `pip`:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Configure API Key
Copy the example environment file and add your Google Gemini API key:
```bash
copy .env.example .env
```
Edit `.env`:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```
*(Note: If no API key is provided, SocialLens automatically activates its built-in rule-based heuristic reasoner so demos and tests remain 100% functional offline!)*

### 4. Run the Full Test Suite
```bash
uv run pytest
# or: python -m unittest discover -s tests -p "test_*.py" -v
```
*(33 comprehensive tests verifying math, reasoner, pipelines, and UI launch)*

---

## 🎬 Live Demo & Judging Guide

### Running in Demo Simulation Mode:
```bash
# Using uv (fastest)
uv run sociallens --demo

# Or directly with python
uv run python -m app.main --demo
```


The glassmorphic HUD will float on your desktop in **Demo Mode**. You can press hotkeys or click the buttons on the HUD:

* **Press `1` — Dry Sarcasm / Irony Scenario**:
  * *Speaker says:* `"Yeah, that's just fantastic."`
  * *Signal:* Positive words (Joy 90%) + Flat monotone tone (Neutral 88%) + Deadpan face (Neutral 88%).
  * *Result:* JSD spikes to **0.446** (Amber/Purple).
  * *HUD Card:* **DRY SARCASM / IRONY** badge pops up with tip: *"Acknowledge the shared setback lightly rather than taking the literal praise at face value."*
* **Press `2` — Sincere Praise Scenario**:
  * *Speaker says:* `"Thank you so much, this is great!"`
  * *Signal:* All channels in harmony (Joy > 85%).
  * *Result:* JSD drops to **0.007** (Green "In Sync"). The HUD stays calm—demonstrating zero notification fatigue!
* **Press `3` — Concealed Frustration Scenario**:
  * *Speaker says:* `"No, it's totally fine, don't worry about it."`
  * *Signal:* Polite words + Vocal tension (Anger 80%) + Facial grimace (Disgust 80%).
  * *Result:* JSD spikes to **0.579**.
  * *HUD Card:* **CONCEALED FRUSTRATION** badge pops up with tip: *"Offer reassurance or gently invite them to share if they have concerns."*

### Running in Live Call Mode:
```bash
python -m app.main
```
Positions the HUD over Zoom/Meet, continuously listening to Windows WASAPI speaker loopback and screen frames.

---

## 📹 2.5-Minute Demo Video Script & Storyboard

| Timestamp | Video Visual | Audio / Narration |
| :--- | :--- | :--- |
| **0:00 – 0:30** | Split screen: Zoom call with someone speaking with a flat expression. Cut to neurodivergent person looking puzzled and anxious. | *"Over 1 billion people experience neurodivergence or social anxiety. On 2D video calls, micro-expressions and tone get flattened. When someone says 'Oh, fantastic' with a deadpan face, are they thrilled or frustrated? The fear of misreading these cues causes immense fatigue."* |
| **0:30 – 1:00** | Screen capture zooms into the SocialLens HUD floating over the video call. Show the 3 live confidence meters (Face, Tone, Words) and the JSD gauge. | *"Meet SocialLens—an assistive desktop copilot that acts as subtitles for social subtext. We don't just dump text into an LLM. We monitor three independent channels: what is said, vocal tone, and facial expression. Using the Jensen-Shannon Divergence, we mathematically measure cross-modal incongruence."* |
| **1:00 – 1:40** | **Live Demo 1**: Presenter triggers Scenario 1 (`Key 1`). The telemetry bars jump, JSD spikes to purple, and the Insight Card smoothly fades in. | *"Watch what happens when a speaker delivers deadpan sarcasm. Their words express praise, but their tone and expression are deadpan. The JSD meter immediately detects this conflict and triggers Gemini 2.5 Flash. In under 300 milliseconds, SocialLens translates the contradiction into a gentle response tip."* |
| **1:40 – 2:05** | **Live Demo 2 & 3**: Presenter triggers Scenario 2 (`Key 2`), then Scenario 3 (`Key 3`). | *"Notice that when the speaker offers sincere praise, all channels agree—the HUD stays completely calm with zero distraction. But when passive frustration arises, the card alerts the user with constructive de-escalation advice."* |
| **2:05 – 2:30** | Full desktop shot showing the polished glassmorphic interface seamlessly coexisting with a video call. Final title card with HackGT and Meta logos. | *"SocialLens doesn't replace human relationships—it empowers people to connect with confidence, empathy, and peace of mind. By bringing people closer together through multimodal AI, SocialLens makes digital communication accessible for everyone."* |

---

## 🛠️ Tech Stack Breakdown

* **GUI / Overlay**: PyQt6 (`FramelessWindowHint`, `WindowStaysOnTopHint`, `WA_TranslucentBackground`) with standard library desktop fallback
* **Screen Ingestion**: `mss` 60+ FPS screen grabber with timestamped ring buffer and dynamic ROI selector
* **Audio Capture**: `soundcard` Windows WASAPI loopback downsampling to 16 kHz mono
* **Voice Activity Detection**: Silero VAD (PyTorch / ONNX) with 500ms silence pause segmentation
* **Speech-to-Text**: `faster-whisper` (`base.en` INT8 on CPU)
* **Semantic Emotion**: `j-hartmann/emotion-english-distilroberta-base` with resilient lexicon fallback
* **Acoustic Prosody**: Real-time DSP fundamental frequency autocorrelation ($F_0$), pitch variance, and RMS energy
* **Facial Emotion**: OpenCV face tracking and keyframe temporal aggregation
* **Mathematical Core**: Multi-Distribution Jensen–Shannon Divergence (JSD) bounded in $[0, 1]$
* **Cognitive Reasoner**: Google Gemini (`gemini-2.5-flash`) with structured JSON schema

---

## 👥 The Team
Built with ❤️ at **HackGT 2026** under the **Meta Track**.
