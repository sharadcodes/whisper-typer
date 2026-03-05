#!/usr/bin/env sh
# Whisper Typer — Start Faster-Whisper server locally (no Docker)
# Run setup-transcribe-local.sh first if you haven't already.

set -e
cd "$(dirname "$0")"

echo "Whisper Typer — Starting transcribe server (local)"

# Ensure venv exists
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
    echo "No virtual environment found. Run setup-transcribe-local.sh first."
    exit 1
fi

# Load .env if present
if [ -f ".env" ]; then
    set -a; . ./.env; set +a
fi

# Run uvicorn from root/app (same module layout as Docker)
cd root/app
exec uv run uvicorn transcribe_api:app --host 127.0.0.1 --port 8000
