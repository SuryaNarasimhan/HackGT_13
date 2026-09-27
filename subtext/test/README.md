# Meeting Overlay Test

A separate macOS overlay prototype for transcribing a two-person meeting. Start with your microphone alone, or select a meeting app window (bring the meeting tab forward first) to include the other person's audio. Your microphone is labeled **You** and the selected window's audio is labeled **Other person**. If window capture fails, microphone transcription continues.

Transcription runs through the repository's local Python service with faster-whisper. The prototype captures audio only; it does not sample the screen or send conversation content to an online interpretation service.

## Run

From the repository root:

```sh
./test/run.sh
```

The first launch installs the local service packages and downloads the Whisper English model. macOS will request microphone and screen-capture permissions. Only one copy of the local service can use port 8765 at a time, so quit the existing Subtext app before launching this prototype.
