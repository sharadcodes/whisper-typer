# Whisper Typer

Push-to-talk voice transcription using Faster-Whisper. System tray app + hotkey; supports Windows, macOS, and Linux.

## Run

### 1. Start the Faster-Whisper server (Docker)

Build and run the container (runs on CPU by default):

```powershell
docker compose up -d --build
```

**If you have an NVIDIA GPU** (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Wait until the container is ready. Check logs:

```powershell
docker compose logs -f faster-whisper
```

### 2. Run the app

From the project root:

```powershell
uv run server.py
```

On first run you’ll pick a model (1–5) at the CLI; then the tray starts.

### 3. Use the system tray or keyboard shortcut

A tray icon appears (Windows: taskbar; macOS: menubar; Linux: system tray).

**Keyboard shortcut: Alt+PageUp** — press to start recording, press again to stop and transcribe.

Or **left-click the tray icon** (or right-click → **Record / Stop**) to toggle recording.

- **Blue** — ready
- **Red** — recording
- **Amber** — transcribing

When done, the transcription is **automatically typed into the focused input field**. A toast notification also shows the result.

## Requirements

- **OS:** Windows, macOS, or Linux
- Python 3.x, `uv`
- Dependencies: `uv pip install sounddevice numpy pystray pillow pynput` (or use `uv run server.py`)
- Docker (for the faster-whisper backend)

## Config

- **Server:** In `server.py`, `SERVER_IP` / `SERVER_PORT` (default `127.0.0.1:8000`) — change if the server runs on another host.
- **Model:** Chosen at app startup (CLI prompt 1–5).
- **GPU:** Use `docker-compose.gpu.yml` to run the `faster-whisper:gpu` image with 1 NVIDIA GPU. No code changes needed.

## API

The Docker container runs a **FastAPI** server on port **8000**:

- **GET /** — service info
- **POST /transcribe?model=tiny** — upload raw PCM audio (16 kHz, 16-bit, mono); returns `{"text": "..."}`
