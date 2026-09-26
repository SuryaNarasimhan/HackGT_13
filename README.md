# MSAS desktop companion

A Windows laptop application that sits alongside a call with a quiet screen border and floating explanation controls. Google Meet is the first trial. An optional Edge extension captures browser tabs; OS window/screen capture remains available for other applications. macOS and Linux are outside the supported scope.

## Keep video live with Edge minimized

Use **Connect browser tab**. This captures the tab directly and streams it locally to MSAS. The older **Choose window or screen** mode depends on the source window being rendered and does not support the minimized-browser requirement.

One-time setup:

1. Open `edge://extensions` in Edge (Chromium 116 or newer).
2. Enable **Developer mode**, click **Load unpacked**, and select `C:\Users\Lakshmi\Desktop\MSAS\extension` (the `extension` folder in this project).
3. Pin **MSAS Call Connection** from Edge's Extensions menu. If already installed, click **Reload** on its extension card after code updates.

For each call:

1. Fully restart MSAS. Open your call tab in Edge.
2. Optionally enable **Include call audio** in MSAS, then click **Connect browser tab**. Copy its complete pairing code. An unused code expires after two minutes.
3. Switch to the call tab, open **MSAS Call Connection**, paste the code, and click **Connect this tab**.
4. Wait for the live preview, then minimize Edge. Keep Edge and the call tab open. Pin the desired participant and select their video area in MSAS for expression analysis.
5. Stop from MSAS or the extension. A new connection requires a new code. Capture follows the selected tab across navigation until sharing stops.

The extension runs capture in an offscreen document, so closing its popup does not stop the stream. Optional audio comes from the tab only; its sound is restored in the extension and the MSAS preview stays muted to avoid doubling sound. Physical microphone/camera access is not requested. No cloud service, API key, or new npm dependency is needed.

The signaling listener binds only to `127.0.0.1:47831`, requires a random per-session token, and exists only during pairing/capture. It validates Host, supplied Origin, client ID, and bounded messages. Signaling carries connection descriptions; media travels over a same-machine WebRTC peer connection with no STUN/TURN servers. No media recording or persistent pairing storage is created. Stop clears the preview, facial analysis, peer, pairing code and listener. The extension releases its tracks when the connection ends. If the port is busy, close other MSAS instances.

Verification: all 12 unit checks and syntax checks passed. An actual unpacked Edge extension was exercised through its popup with a changing synthetic tab. WebRTC delivered its video to the app renderer, pixels continued changing after setting the source window to minimized in a headless browser test, and app Stop ended extension sharing. Electron IPC was simulated in this test; a native Electron plus live Meet session is still a manual Windows acceptance check. Other call platforms are not yet validated.

## Speaker-baseline AI backend

The merged `app/` package contains the Python multimodal pipeline from the `speaker-baseline` branch. It learns an in-memory voice and face baseline for the current speaker, marks readings as warming up, usual, changed, or unavailable, and only uses informative channels when calculating cross-modal mismatch. This avoids treating a naturally flat voice, a rarely smiling face, missing video, or unclear speech as evidence of hidden meaning.

The Python backend and the Electron companion currently have separate entry points. Merging the branch preserves both working prototypes; it does not yet route the Electron tab stream into the Python coordinator or display Python cue results in the Electron UI. The Electron app remains the default `npm start` experience. To run the Python overlay separately on Windows:

```powershell
uv sync
Copy-Item .env.example .env
uv run sociallens --demo
```

`GEMINI_API_KEY` is optional for the deterministic/offline fallback. Live mode is `uv run sociallens`; it uses Windows audio/screen capture and may download local ML models on first use. Run the backend checks with `uv run pytest`. Do not claim end-to-end Electron/backend integration until the media and cue IPC boundary is implemented and tested.

## Run locally

Install Node.js 22.12 or newer with npm, then run in this repository:

```sh
npm install
npm start
```

This launches a native Electron window, not a localhost website. No web server, account, or AI API key is needed. `npm run check` checks JavaScript syntax. The lockfile pins dependencies; use `npm ci` for repeat installs.

On this checkout, Electron is already downloaded. If npm is not on your terminal's PATH, launch the installed binary directly from PowerShell:

```powershell
Set-Location C:\Users\Lakshmi\Desktop\MSAS
& .\node_modules\electron\dist\electron.exe .
```

## Verification status

JavaScript syntax and dependency audit checks passed. Browser-based renderer checks passed for source selection, scripted cue cycling, synthetic video-stream cleanup, late capture cancellation, and the 960px layout. Main-process checks with simulated Electron APIs passed for sender authorization, microphone denial, overlay lifecycle, explicit audio opt-in, and single-use capture grants. The setup layout was visually inspected.

Native desktop verification remains incomplete: Electron's GPU/renderer subprocesses failed to start in the agent environment, including in a minimal unrelated Electron window. These checks therefore do not establish that native capture, system audio, or the overlay works on the demo laptop. Run the manual Google Meet trial below in a normal desktop session before presenting those features as verified.

## What works in this prototype

- Desktop setup interface with optional, explicitly entered language and cultural context.
- Window/screen selection, local video preview, and opt-in system-output audio capture.
- Audio level meter that distinguishes a present track from a sustained lack of signal. Captured audio is never played back, avoiding feedback.
- A click-through border and a separate interactive panel above other windows. The controls can expand explanations, return to setup, and stop the session.
- Local facial landmarks and tentative expression estimates for a selected participant.
- Stop/close cleanup: media tracks stop, audio analysis closes, overlays disappear, and context fields clear. Demo mode never captures a call.

**Not implemented:** transcription, vocal-tone inference, sarcasm or other social-cue interpretations, translated output, packaged installers, code signing, or automatic tracking of a participant/window across monitors. Context fields are session-only UI inputs reserved for the later inference pipeline; they do not change the scripted demo.

## Google Meet trial on Windows

1. Open a consenting test call in your browser. Keep it visible and use headphones.
2. Launch MSAS and choose the browser window containing the call. Selecting a whole screen exposes everything visible on that screen to the local preview.
3. Enable **Include system audio** if you want to test audio. This captures available system output and may include other applications; selecting a window does not isolate its audio. The microphone is not requested.
4. Click **Start companion**. Verify the local video preview and audio meter while the other participant speaks. A track with no signal is not proof of working call-audio capture.
5. Click **Return to call**. The border and floating controls remain. The border is click-through; the small panel is intentionally interactive.
6. Expand the panel to inspect capture status. Live mode never produces simulated cues.
7. Stop from the floating panel. Verify the overlays disappear and the setup preview and context clear.
8. Return to setup and follow the local expression analysis trial below.

## Capture and display limitations

Windows is the only supported target. Actual Google Meet audio, source selection, and overlay interaction must be tested on the demo laptop. This repository does not claim universal call compatibility. The preview mirrors the selected browser window; keep Meet as its active tab and leave the window restored. This does not move the meeting into MSAS or create another call participant.

For a selected screen, overlays use that screen. For a window source without a display identifier, overlays use the display containing the MSAS setup window. Move setup to the desired display before starting. The border surrounds a display, not the exact call window; it does not follow a moved call window. A disconnected display stops the session. OS-controlled or exclusive full-screen surfaces may cover the overlay; use a normal maximized call window for the first trial.

### Selected window does not become a live preview

The original prototype rejected every `media` permission. Electron 44.4.5 uses that permission for desktop capture too, with an empty `mediaTypes` array, before invoking the display-source handler. This allowed source thumbnails to work while the live stream was denied. The corrected handler allows that desktop-only request for a pending, explicitly selected source in the setup window's main frame. Physical microphone/camera requests remain denied. See Electron's [media permission implementation](https://github.com/electron/electron/blob/v44.4.5/shell/browser/web_contents_permission_helper.cc).

After updating, fully quit and relaunch MSAS; refreshing the renderer alone does not reload the main-process permission handler. Select the Meet browser window and click **Start companion**. Overlays now open only after the preview plays. Capture and playback each time out after 15 seconds instead of leaving an indefinite connecting state. Errors identify the failed stage and preserve the native error name/message. An audio-meter error no longer closes a working video preview.

## Privacy and application boundaries

This version performs no remote inference and sends no captured content to a server. It creates no media recordings and has no transcript database, analytics, or content logging. Source thumbnails and previews exist in memory; runtime preferences/caches managed by Electron or the OS are distinct from a guarantee of forensic erasure.

The renderer uses an in-memory session partition, a restrictive content security policy, context isolation, sandboxing, and a narrow preload API. Navigation and new windows are blocked. Only the main setup window can authorize capture. The passive border cannot invoke privileged actions. System audio is opt-in and microphone access is not granted.

Source enumeration generates transient thumbnails before capture begins, after the user opens the picker. Stop clears the source cache. No AI provider has been selected, and no promise about future remote-provider retention is implied.

## Layout

- `electron/main.cjs`: window lifecycle, source selection, capture authorization, session state, and scripted examples.
- `electron/preload.cjs`: explicit renderer-to-main bridge.
- `src/index.html`, `src/app.js`, `src/styles.css`: setup interface, media lifecycle, and audio meter.
- `src/overlay.html`, `src/overlay.js`, `src/overlay.css`: passive border and interactive floating panel.
- `docs/architecture.md`: product architecture and next implementation stages.

## Before claiming a working AI demo

Connect the actual transcription and interpretation pipeline, preserve abstention, validate provider privacy settings if remote inference is used, and measure cue latency and accuracy. Local expression inference works; interpretation of speech and intended meaning is still pending.

## Local expression analysis

Human 3.3.6 is pinned and installed with the app. No API key or paid service is needed. After starting capture, pin the other participant in Meet and click **Select participant**. Drag around their video in the preview, or use arrows to move the selection, Shift+arrows to resize, and Enter to confirm. Optional landmarks show face geometry. **Stop analysis** pauses inference without stopping capture.

Only the selected crop is processed in a worker, at most 480 pixels on its longest edge, one request at a time with 250 ms between completions. WebGL is preferred with CPU fallback. Three consistent observations are required before a tentative expression label appears. Missing, weak, multiple-face, or stale observations clear the estimate. Scores are not calibrated probabilities. Expressions do not establish someone's feelings or intentions; lighting, pose, occlusion and individual differences can cause mistakes. This is not identity tracking: keep the same participant pinned and reselect if tiles rearrange.

All models are bundled through the dependency. Frames and results stay in memory; stopping terminates the worker. A local asset allowlist and CSP restrict loading, remote HTTP/WebSocket requests are blocked, and persistent model caching is disabled. Identity and demographic models are disabled. Context settings do not influence this classifier. Expression results appear below the preview; the floating border still shows capture status.

Validation: eight automated tests passed for stabilization, abstention, letterbox coordinates, restricted assets and installed model shards. An actual Edge worker test loaded all three models, detected a face with 468 landmarks, then returned no faces on blank input. This is an integration check, not a population accuracy measurement. Native Google Meet capture plus inference still needs a manual Windows trial. Run `npm run check` and `npm test`.

Manual trial: select a pinned participant, enable landmarks, test a clear face then an empty tile, try two faces in the region, pause and reselect, and stop capture during model loading. Estimates must clear when analysis stops. The simplified UI has Call source and Context; older simulated-demo instructions above describe retained prototype code, not a visible demo button.

Attribution: [Human](https://github.com/vladmandic/human) is MIT licensed. Its [model inventory](https://github.com/vladmandic/human/wiki/Models) identifies the detector/mesh as MediaPipe-derived ([Apache 2.0](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)) and expression model as derived from [face_classification](https://github.com/oarriaga/face_classification) ([MIT](https://github.com/oarriaga/face_classification/blob/master/LICENSE)). Preserve applicable notices when distributing an installer.
