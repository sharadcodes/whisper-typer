#!/usr/bin/env sh
# Whisper Typer — Setup local Whisper server (no Docker)
# Creates a venv, installs server deps, and pre-downloads the model.

set -e
cd "$(dirname "$0")"

echo "Whisper Typer — Setting up local server"

# Load .env if present
if [ -f ".env" ]; then
    # shellcheck disable=SC1091
    set -a; . ./.env; set +a
fi

MODEL="${WHISPER_MODEL:-tiny}"

# Create .env from .env.example if missing
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "  .env created from .env.example — fill in HF_TOKEN if desired."
fi

# Create venv if not present
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

VENV_PYTHON=".venv/bin/python"
[ -d "venv" ] && VENV_PYTHON="venv/bin/python"

# Install server dependencies (ctranslate2 is large — increase timeout)
export UV_HTTP_TIMEOUT=300
echo "Installing server dependencies..."
uv pip install fastapi uvicorn python-multipart faster-whisper numpy
[ $? -ne 0 ] && { echo "Dependency install failed. Try again or increase UV_HTTP_TIMEOUT."; exit 1; }

# Pre-download model
echo "Downloading Whisper model '$MODEL'..."
"$VENV_PYTHON" - <<EOF
import os
from faster_whisper import WhisperModel
model = "$MODEL"
models_dir = os.path.join(os.getcwd(), "models")
os.makedirs(models_dir, exist_ok=True)
print(f"Saving to {models_dir}")
WhisperModel(model, device="cpu", compute_type="int8", download_root=models_dir)
print(f"Model [{model}] ready.")
EOF

echo ""
echo "Done. Start the server with: uv run server.py --local"
