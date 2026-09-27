#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEST_DIR="$ROOT/test"
VENV="$ROOT/backend/.venv"
PYTHON_BIN=""
XCODE_DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"

if [ -x "$XCODE_DEVELOPER_DIR/Toolchains/XcodeDefault.xctoolchain/usr/bin/swift" ]; then
    SWIFT_BIN="$XCODE_DEVELOPER_DIR/Toolchains/XcodeDefault.xctoolchain/usr/bin/swift"
    DEVELOPER_DIR_FOR_BUILD="$XCODE_DEVELOPER_DIR"
else
    SWIFT_BIN="$(command -v swift || true)"
    DEVELOPER_DIR_FOR_BUILD="$(xcode-select -p)"
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
        sys.exit("The local transcription service is already running on port 8765. Quit the other app before starting this test overlay.")
finally:
    sock.close()'

if [ ! -x "$VENV/bin/python" ]; then
    "$PYTHON_BIN" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --disable-pip-version-check -r "$ROOT/backend/requirements.txt"

cd "$TEST_DIR"
DEVELOPER_DIR="$DEVELOPER_DIR_FOR_BUILD" "$SWIFT_BIN" build -c release

APP_DIR="$TEST_DIR/.build/release/MeetingOverlayTest.app"
mkdir -p "$APP_DIR/Contents/MacOS"
cp "$TEST_DIR/.build/release/MeetingOverlayTest" "$APP_DIR/Contents/MacOS/MeetingOverlayTest"
cp "$TEST_DIR/Resources/Info.plist" "$APP_DIR/Contents/Info.plist"

cd "$ROOT/backend"
"$VENV/bin/python" -m uvicorn app:app --host 127.0.0.1 --port 8765 --log-level warning &
BACKEND_PID=$!
trap 'kill "$BACKEND_PID" 2>/dev/null || true' EXIT INT TERM

"$VENV/bin/python" -c 'import time, urllib.request
url="http://127.0.0.1:8765/health"
deadline=time.time()+30
while time.time()<deadline:
    try:
        urllib.request.urlopen(url, timeout=1).read()
        break
    except Exception:
        time.sleep(0.25)
else:
    raise SystemExit("The local transcription service did not start.")'

open -W "$APP_DIR"
