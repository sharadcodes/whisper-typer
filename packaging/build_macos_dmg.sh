#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${APP_NAME:-Whisper Typer}"
DIST_DIR="${DIST_DIR:-dist}"
BUILD_DIR="${BUILD_DIR:-build}"
OUT_DIR="${OUT_DIR:-out}"

mkdir -p "$OUT_DIR"

rm -rf "$DIST_DIR" "$BUILD_DIR"

# Build an .app bundle. We use --windowed to avoid a terminal window.
pyinstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name "$APP_NAME" \
  --distpath "$DIST_DIR" \
  --workpath "$BUILD_DIR" \
  --collect-all customtkinter \
  --collect-all pystray \
  --collect-all pynput \
  --collect-all sounddevice \
  --collect-all faster_whisper \
  packaging/pyinstaller/macos_entry.py

APP_PATH="$DIST_DIR/$APP_NAME.app"
if [[ ! -d "$APP_PATH" ]]; then
  echo "Expected app not found at: $APP_PATH" >&2
  exit 2
fi

# Package DMG.
export APP_PATH
export APP_NAME
export DMG_NAME="$OUT_DIR/Whisper-Typer-macOS.dmg"

python -m dmgbuild -s packaging/dmgbuild/settings.py "$APP_NAME" "$DMG_NAME"

echo "DMG created at: $DMG_NAME"

