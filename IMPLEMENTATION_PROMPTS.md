# Implementation Prompts: "SocialLens"
**A Step-by-Step Prompt Execution Guide for Modular Implementation**

This file contains sequential, self-contained prompts you can copy-paste to build **SocialLens** step-by-step. Each prompt is scoped to build, test, and verify one layer before moving to the next—preventing dependency conflicts and ensuring zero UI lag.

---

## Roadmap Overview

```
[ Phase 1: Environment & Math Engine ] ──► [ Phase 2: Gemini Reasoner Agent ]
                                                          │
[ Phase 4: Capture & VAD Ingestion ]   ◄── [ Phase 3: Three Emotion Pipelines ]
         │
         ▼
[ Phase 5: PyQt6 Transparent HUD ]     ──► [ Phase 6: End-to-End & Demo Mode ]
```

---

## Phase 1: Environment & Core Math Engine

### 🔹 Prompt 1.1: Project Scaffolding, Dependencies & Emotion Taxonomy
> **Prompt:**
> "Let's start Phase 1 of SocialLens. Please:
> 1. Create a `requirements.txt` with all necessary dependencies (`soundcard`, `numpy`, `scipy`, `faster-whisper`, `transformers`, `torch`, `mss`, `opencv-python`, `PyQt6`, `google-genai`, `python-dotenv`).
> 2. Create the project directory structure as specified in [PROJECT_PLAN.md](file:///c:/Users/hsu_s/Projects/GaTech/HackGT13/PROJECT_PLAN.md).
> 3. Implement `app/config.py` for global settings (sample rates, buffer sizes, JSD threshold $\tau$, model names, API keys).
> 4. Implement `app/pipelines/taxonomy.py` defining the canonical 7-emotion discrete simplex (`joy`, `surprise`, `sadness`, `anger`, `disgust`, `fear`, `neutral`), including helper functions for vector validation and softmax normalization."

---

### 🔹 Prompt 1.2: Multi-Distribution Jensen–Shannon Divergence (JSD) & Unit Tests
> **Prompt:**
> "Now let's build the mathematical core in `app/math_engine/jsd.py`:
> 1. Implement the generalized Jensen–Shannon Divergence for three discrete distributions ($P_V, P_A, P_S$) using base-2 logarithm so the output is strictly bounded in $[0, 1]$.
> 2. Include $\epsilon$-clamping for numerical stability to prevent $\log_2(0)$ or division-by-zero errors.
> 3. Add pairwise JSD calculation functions to identify which specific modalities conflict the most (e.g., Words vs. Tone).
> 4. Create `tests/test_jsd.py` with pytest unit tests verifying:
>    - Identical distributions yield $JSD = 0$.
>    - Completely disjoint orthogonal distributions yield maximum divergence $\approx 1.0$.
>    - Symmetry: order of distributions does not alter the result.
>    - Run the tests to confirm they pass."

---

## Phase 2: Cognitive Reasoning Engine (Gemini 2.5 Flash)

### 🔹 Prompt 2.1: Gemini Reasoning Agent & Structured Schema
> **Prompt:**
> "Let's implement the cognitive reasoner in `app/agent/gemini_reasoner.py`:
> 1. Use the new `google-genai` SDK (`gemini-2.5-flash`).
> 2. Implement Pydantic or structured JSON schema output returning: `social_cue_type`, `confidence`, `explanation`, and `suggested_action`.
> 3. Write an empathetic prompt that instructs the model to translate cross-modal incongruence into assistive, non-accusatory advice for a neurodivergent user.
> 4. Add a graceful fallback/mock mode so the application still works if an API key is missing or offline.
> 5. Create a test script `tests/test_gemini_reasoner.py` that simulates a sarcasm scenario and verifies structured JSON response parsing."

---

## Phase 3: The Three Perception Pipelines

### 🔹 Prompt 3.1: Semantic Pipeline (STT + Text Emotion)
> **Prompt:**
> "Let's implement the Semantic Pipeline in `app/pipelines/semantic_pipeline.py`:
> 1. Use `faster-whisper` (`base.en` on CPU/GPU) to transcribe speech audio buffers into text.
> 2. Use a fast pre-trained Hugging Face emotion classifier (`j-hartmann/emotion-english-distilroberta-base`) to classify the transcript into an emotion distribution.
> 3. Project the model's output into the canonical 7-emotion simplex via `taxonomy.py`.
> 4. Write a standalone test to transcribe a sample audio file or string and output the normalized 7-D probability vector."

---

### 🔹 Prompt 3.2: Audio Acoustic/Prosody Pipeline
> **Prompt:**
> "Let's implement the Audio Prosody Pipeline in `app/pipelines/audio_pipeline.py`:
> 1. Ingest raw 16kHz mono audio waveform segments.
> 2. Extract acoustic tone emotion using a lightweight pre-trained audio emotion model (or feature extractor using prosodic pitch variance, energy contour, and speech rate).
> 3. Map the predicted probabilities to the canonical 7-emotion simplex.
> 4. Include fallback heuristics (e.g., pitch variance analysis: flat monotone vs. expressive pitch shifts).
> 5. Verify the pipeline on a sample audio clip and confirm the output format."

---

### 🔹 Prompt 3.3: Video Perception Pipeline
> **Prompt:**
> "Let's implement the Video Pipeline in `app/pipelines/video_pipeline.py`:
> 1. Receive RGB frames/crops from screen capture.
> 2. Detect face bounding box and extract facial emotion probabilities using a lightweight ONNX/OpenCV or DeepFace model.
> 3. Aggregate predictions across sampled keyframes (e.g., 3-5 frames over the utterance window) and map them to the canonical 7-emotion simplex.
> 4. Handle edge cases gracefully: if no face is detected (camera off/occluded), return a uniform neutral distribution with a flag.
> 5. Verify the pipeline on a sample webcam or screenshot image."

---

## Phase 4: Capture & Voice Activity Detection (VAD)

### 🔹 Prompt 4.1: Audio Loopback & Silero VAD Ingestion
> **Prompt:**
> "Let's implement the audio capture and speech segmentation:
> 1. In `app/capture/audio_loopback.py`, use `soundcard` to capture Windows default speaker loopback (WASAPI) continuously into a thread-safe circular buffer (last 8 seconds of 16kHz mono audio).
> 2. In `app/capture/vad_detector.py`, integrate Silero VAD to detect speech start and speech end.
> 3. Trigger an 'utterance completed' event when speech is followed by $\ge 500\text{ ms}$ of silence.
> 4. Slice the corresponding audio segment and emit it to the processing coordinator."

---

### 🔹 Prompt 4.2: Screen Capture & Face Cropping
> **Prompt:**
> "Let's implement the screen capture system in `app/capture/screen_capture.py`:
> 1. Use `mss` to capture a designated screen region or the active Zoom/Meet window at low CPU cost.
> 2. Maintain a ring buffer of recent frames timestamped to match the audio stream.
> 3. Add an auto-detection or interactive selection utility allowing the user to select the screen tile containing the speaker's video."

---

## Phase 5: Desktop HUD Overlay (PyQt6)

### 🔹 Prompt 5.1: Frameless Transparent Window & Glassmorphism Styling
> **Prompt:**
> "Let's build the desktop overlay foundation in `app/ui/overlay_window.py` and `app/ui/styles.py`:
> 1. Create a `QMainWindow` with `Qt.WindowType.FramelessWindowHint`, `Qt.WindowType.WindowStaysOnTopHint`, and `Qt.WidgetAttribute.WA_TranslucentBackground`.
> 2. Implement mouse drag events so the user can smoothly position the overlay anywhere on the screen.
> 3. Apply modern glassmorphism CSS (dark translucent slate background, rounded corners, subtle neon border, modern typography).
> 4. Add a compact header with status pill ('Active / Listening'), minimize button, and close button."

---

### 🔹 Prompt 5.2: Live Telemetry Meters & Incongruence Gauge
> **Prompt:**
> "Let's add the live telemetry components to the HUD:
> 1. Create a 3-channel meter widget displaying:
>    - **Face (Video)**: Top detected emotion + percentage bar.
>    - **Tone (Audio)**: Top detected emotion + percentage bar.
>    - **Words (Semantics)**: Top detected emotion + percentage bar.
> 2. Add an **Incongruence Gauge** displaying the current JSD divergence score ($0.0 - 1.0$) with smooth color transitions:
>    - Green ($0.0 - 0.35$): Balanced / In-sync
>    - Amber ($0.35 - 0.48$): Subtle variance
>    - Purple / Magenta ($> 0.48$): High divergence (Cue Detected)
> 3. Ensure all UI updates run on PyQt signals to avoid cross-thread UI freezes."

---

### 🔹 Prompt 5.3: Subtext Insight Card (Animated Cue Card)
> **Prompt:**
> "Let's build the Subtext Insight Card widget in `app/ui/components/cue_card.py`:
> 1. Create an expandable notification card that reveals itself with a smooth fade-in animation whenever Gemini emits a social cue analysis.
> 2. Display:
>    - **Cue Badge**: e.g., 'Dry Sarcasm / Irony' (with distinct color tags).
>    - **Explanation**: 1-2 sentence non-judgmental breakdown of the cross-modal mismatch.
>    - **Suggested Action / Response**: Supportive advice on how to respond.
>    - **Dismiss Button** or auto-fade timer (after 10 seconds).
> 3. Provide manual trigger/test buttons to preview card animations during development."

---

## Phase 6: Orchestration, Integration & Hackathon Demo Mode

### 🔹 Prompt 6.1: Pipeline Coordinator & Application Loop
> **Prompt:**
> "Let's bind everything together in `app/coordinator.py` and `app/main.py`:
> 1. Create a `QThread` worker that coordinates:
>    `VAD trigger ──► Parallel pipelines (Video, Audio, Semantics) ──► JSD calculation ──► Gated Gemini Call ──► Emit PyQt Signal`.
> 2. Connect worker signals to update the HUD telemetry bars and display the cue card when $JSD \ge \tau$.
> 3. Implement clean startup and shutdown logic (gracefully terminating background audio/screen capture threads)."

---

### 🔹 Prompt 6.2: Built-in Demo / Simulation Mode
> **Prompt:**
> "For the hackathon submission and live judging, let's create a fail-safe simulation mode in `app/mock_demo.py`:
> 1. Implement a `--demo` CLI flag that loads pre-recorded scenario data (e.g., Scenario 1: Deadpan sarcasm; Scenario 2: Sincere praise; Scenario 3: Concealed frustration).
> 2. Replay these scenarios through the pipeline so judges see the telemetry bars react in real time, the JSD divergence meter spike, and the Gemini cue card pop up reliably without requiring a live Zoom call during presentation.
> 3. Add hotkeys (e.g., keys `1`, `2`, `3`) to trigger specific demo scenarios on demand."

---

### 🔹 Prompt 6.3: Polish, Documentation & Verification
> **Prompt:**
> "Let's finalize the project for submission:
> 1. Test running `python app/main.py --demo` and verify end-to-end responsiveness and visual polish.
> 2. Update `README.md` with:
>    - Project pitch and alignment with Meta's challenge ('Bringing People Closer Together with AI').
>    - Architecture diagram and JSD mathematical explanation.
>    - Quickstart installation instructions.
>    - Demo video script guide for the 2–3 minute video presentation."
