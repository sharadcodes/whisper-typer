#!/usr/bin/env sh
# Whisper Typer — Start Faster-Whisper server (Docker, macOS / Linux)
# Detects platform and GPU availability, runs the appropriate docker compose.

set -e

# Run from script directory (project root)
cd "$(dirname "$0")"

echo "Whisper Typer — Starting Faster-Whisper server (Docker)"

docker compose up -d --build

if [ $? -ne 0 ]; then
  echo "Failed to start container. Check Docker is running."
  exit 1
fi

echo ""
echo "Container started. Logs: docker compose logs -f faster-whisper"

