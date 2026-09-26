#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$ROOT/macos/SubtextOverlay"
VENV="$ROOT/backend/.venv"
PYTHON_BIN=""

# Load this machine's local Gemini key without storing it in the repository.
GEMINI_ENV_FILE="$ROOT/backend/.env.local"
if [ -z "${GEMINI_API_KEY:-}" ] && [ -f "$GEMINI_ENV_FILE" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$GEMINI_ENV_FILE"
    set +a
fi

if [ -z "${DEVELOPER_DIR:-}" ] && [ -x "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/swift" ]; then
    export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
fi

for candidate in python3.12 python3.13 python3.14 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v "$candidate")"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "Python 3.11 or newer is required."
    exit 1
fi

"$PYTHON_BIN" -c 'import socket, sys
sock = socket.socket()
sock.settimeout(0.5)
try:
    if sock.connect_ex(("127.0.0.1", 8765)) == 0:
        sys.exit("Subtext service is already running on port 8765. Quit the earlier run before starting another.")
finally:
    sock.close()'

if [ ! -x "$VENV/bin/python" ]; then
    "$PYTHON_BIN" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --disable-pip-version-check -r "$ROOT/backend/requirements.txt"
"$ROOT/scripts/package-app.sh"

cd "$ROOT/backend"
"$VENV/bin/python" -m uvicorn app:app --host 127.0.0.1 --port 8765 --log-level info &
BACKEND_PID=$!
trap 'kill "$BACKEND_PID" 2>/dev/null || true' EXIT INT TERM

"$VENV/bin/python" -c 'import time, urllib.request; url="http://127.0.0.1:8765/health"; deadline=time.time()+30
while time.time()<deadline:
    try:
        urllib.request.urlopen(url, timeout=1).read()
        break
    except Exception:
        time.sleep(0.25)
else:
    raise SystemExit("The local transcription service did not start.")'

BUILD_DIR="$(cd "$PROJECT" && swift build -c release --show-bin-path)"
open -W "$BUILD_DIR/SubtextOverlay.app"
