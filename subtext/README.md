# Subtext for macOS

Subtext is a small floating communication aid for personal video conversations. It combines a live transcript, locally measured voice and facial-behavior cues, and timing into short, aligned evidence windows. Gemini interprets those windows cautiously and can suggest a gentle check-in. The goal is mutual understanding between people, not judging either person.

## What happens to conversation data

- The Subtext app requests microphone access for itself and captures the MacBook's built-in microphone and selected call-window audio locally. The local Python service transcribes both with faster-whisper. Local microphone transcription starts muted each session; the microphone button turns it on while call-window audio continues either way.
- Call-window video is sampled at 5 frames per second and analyzed locally with Apple Vision. Small face crops are sent over loopback to the local Python service for CNN+LSTM inference; crops are not saved or sent to Gemini.
- Raw audio is analyzed in memory for timing, pauses, approximate pitch, loudness, and speaking rate. Audio is not sent to Gemini or written to disk.
- When a Gemini API key is configured, Subtext sends the transcript and compact numeric cues for a recent 14-second window to Gemini for interpretation. It does not send raw audio or video.
- Recent transcript and cue data stay in process memory and are cleared when the session is cleared or the service exits.

## Requirements

- macOS 13 or newer
- Xcode Command Line Tools (Swift and Swift Package Manager)
- Python 3.11 or newer
- A Gemini API key for interpretation (transcription and local feature extraction work without one)

The first launch installs the local Python packages and downloads model files as needed. Whisper and face-model weights are cached locally.

## Run

For a persistent local setup, replace the placeholder in `backend/.env.local` with your Gemini API key. The launch script reads that private file automatically, then start Subtext:

    ./scripts/run.sh

You can also set the key in your shell for a one-time launch:

    export GEMINI_API_KEY="your-key"
    ./scripts/run.sh

The local key file is ignored by Git, so the key is not saved in the repository. The default model is `gemini-3.5-flash-lite`; set `SUBTEXT_GEMINI_MODEL` to use another compatible Gemini model.

Choose the call window in the overlay, then choose **Start listening**. macOS will ask for microphone and screen-recording access as needed. The first session loads Whisper. Use the refresh button if the call window does not appear.

While `scripts/run.sh` is open, the terminal prints a `DIAG heartbeat` every 10 seconds with capture packet ages, audio queue/VAD/Whisper progress, and face queue/inference progress. It does not print transcript text or image data. Native capture and socket failures are forwarded to the same terminal.

## Current behavior

- The overlay displays separate transcript lines for “You” (microphone captured by the macOS app) and “Other person” (selected call-window audio).
- Voice activity detection and Whisper confidence checks suppress clips and output that look like silence, noise, or unreliable decoding. The local acoustic schema includes speech intervals, response gaps, pauses, approximate median pitch and pitch range, loudness, speaking rate, and change from a short session baseline.
- Apple Vision extracts face presence, a normalized face box, head orientation, mouth aperture, and eye aperture from sampled call-window frames. It crops up to four faces to 224 × 224 JPEGs for local inference. Bounding-box overlap keeps temporary tracks aligned across nearby frames; they are not identity labels.
- The overlay reads the displayed participant name from each detected meeting tile with on-device text recognition and uses it in the valence estimate. Names stay in the macOS app and are removed before visual frames are sent to the local service.
- The local ResNet50 encoder creates one feature vector per sampled face frame. An LSTM reads the newest ten vectors (about two seconds at 5 fps), then updates a negative-to-positive valence estimate. The model predicts seven expression classes; Subtext maps their probabilities to a signed score for display. The score is a rough facial-expression proxy, not a direct valence regression or a reading of someone's inner feelings.
- The temporal checkpoints come from [EMO-AffectNetModel](https://github.com/ElenaRyumina/EMO-AffectNetModel), whose authors describe the released weights as for scientific use. Py-Feat Detectorv2 is not used in this live path because it produces per-frame predictions without a recurrent sequence model.
- Face crops are sent only to `127.0.0.1:8765` for inference and held in memory while processed. The UI receives numeric scores; the crop bytes are excluded from the conversation schema and Gemini requests. The model weights are cached under `~/Library/Caches/Subtext/AffectNetLSTM`.
- A small live preview shows the selected call window with the detected face landmarks drawn over it. Preview frames stay in app memory and are not sent to the local service or Gemini.
- The Python service aligns transcript, audio, and visual records by session-relative timestamps and IDs, with a one-second timeline index. It sends recent aligned windows to Gemini about every nine seconds when all three modalities have evidence.
- Gemini returns schema-validated observations, tentative hypotheses, and a separate implied-language signal. The overlay raises a “Possible implied meaning” card for a quoted idiom, figurative phrase, indirect message, or sarcasm only when text, aligned audio, and a time-aligned visual change support it and Gemini reports confidence of at least 70/100. The score is Gemini's self-rating, not a calibrated probability. This alert is separate from the existing “Possible moment” card.
- No inference is presented as a fact about what someone feels or intends. Facial or vocal behavior alone is not treated as an emotion label.

## Limits

- The app targets personal, usually one-to-one calls. It does not identify speakers in the captured window; multiple remote voices may be grouped as “Other person.”
- Microphone audio can include speaker bleed when call audio plays through speakers. Headphones may reduce duplicate speech.
- Subtext cannot read mute state from every meeting app. When you mute or unmute in the call, use the MacBook microphone button in Subtext to match it; remote call audio continues to be transcribed while the local microphone is muted.
- Whisper uses English by default. Voice activity and prosody values are approximate and affected by noise, distance, and the selected input device.
- Face landmarks can be unavailable when a face is small, occluded, off-screen, or not rendered in the selected window. Five frames per second still misses brief expressions, and the LSTM needs ten frames before its first estimate.
- The live valence feature downloads the public TorchScript checkpoints on first use. The checkpoint authors describe their weights as for scientific use; review that limitation before using the model beyond this prototype.
- Gemini requires internet access and a configured API key. Without a key, the app still transcribes and extracts local cues but does not interpret them.
- This is a prototype. Its cues are not clinical, diagnostic, or reliable determinations of emotion or intent.

## Project layout

- `macos/SubtextOverlay`: SwiftUI/AppKit overlay, ScreenCaptureKit window capture, and local Apple Vision feature extraction
- `backend`: FastAPI service, local transcription and acoustic features, schemas, alignment, and Gemini API client
- `scripts`: build and launch the app and local service

See [ARCHITECTURE.md](ARCHITECTURE.md) for the data flow and API details.
