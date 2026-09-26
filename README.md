# MSAS desktop companion

A Windows laptop application that sits alongside a call with a quiet screen border and floating explanation controls. Google Meet is the first manual trial; capture uses operating-system sources, not Meet selectors, extensions, or APIs. macOS and Linux are outside the supported scope.

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
- Four manually stepped, clearly labeled simulated examples: sarcasm, idiom, slang, and an ambiguous statement where interpretation is withheld.
- Stop/close cleanup: media tracks stop, audio analysis closes, overlays disappear, and context fields clear. Demo mode never captures a call.

**Not implemented:** transcription, vocal-tone inference, facial analysis, live AI interpretations, translated output, packaged installers, code signing, or automatic tracking of a participant/window across monitors. Context fields are session-only UI inputs reserved for the later inference pipeline; they do not change the scripted demo.

## Google Meet trial on Windows

1. Open a consenting test call in your browser. Keep it visible and use headphones.
2. Launch MSAS and choose the browser window containing the call. Selecting a whole screen exposes everything visible on that screen to the local preview.
3. Enable **Include system audio** if you want to test audio. This captures available system output and may include other applications; selecting a window does not isolate its audio. The microphone is not requested.
4. Click **Start companion**. Verify the local video preview and audio meter while the other participant speaks. A track with no signal is not proof of working call-audio capture.
5. Click **Return to call**. The border and floating controls remain. The border is click-through; the small panel is intentionally interactive.
6. Expand the panel to inspect capture status. Live mode never produces simulated cues.
7. Stop from the floating panel. Verify the overlays disappear and the setup preview and context clear.
8. Separately, use **Try a simulated cue** to inspect the intended explanation interaction. **Next example** cycles the four examples. No microphone or screen is captured in this mode.

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

Connect the actual transcription and interpretation pipeline, preserve abstention, validate provider privacy settings if remote inference is used, and measure cue latency and accuracy. The current scripted walkthrough demonstrates the interface only.
