#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$ROOT/macos/SubtextOverlay"

# Prefer the installed full Xcode toolchain when no developer directory was
# explicitly selected; some Command Line Tools installations ship a mismatched SDK.
if [ -z "${DEVELOPER_DIR:-}" ] && [ -x "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/swift" ]; then
    export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
fi

cd "$PROJECT"
swift build -c release

BUILD_DIR="$(swift build -c release --show-bin-path)"
BINARY="$BUILD_DIR/SubtextOverlay"
APP="$BUILD_DIR/SubtextOverlay.app"
mkdir -p "$APP/Contents/MacOS"
cp "$BINARY" "$APP/Contents/MacOS/SubtextOverlay"
cp "$PROJECT/Resources/Info.plist" "$APP/Contents/Info.plist"

SIGNING_IDENTITY="${SUBTEXT_SIGNING_IDENTITY:-}"
if [ -z "$SIGNING_IDENTITY" ]; then
    SIGNING_IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null \
        | awk '/"Apple Development:/ { print $2; exit }')"
fi

SIGNING_REQUIREMENTS=()
if [ -z "$SIGNING_IDENTITY" ] || [ "$SIGNING_IDENTITY" = "-" ]; then
    SIGNING_IDENTITY="-"
    SIGNING_REQUIREMENTS=(--requirements '=designated => identifier "org.subtext.overlay"')
    echo "No stable signing identity found; using Subtext's stable bundle identifier for this local build."
fi

codesign --force --deep --sign "$SIGNING_IDENTITY" \
    "${SIGNING_REQUIREMENTS[@]}" \
    --entitlements "$PROJECT/Resources/SubtextOverlay.entitlements" \
    "$APP"
echo "Built $APP"
