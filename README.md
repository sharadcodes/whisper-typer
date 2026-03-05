# Whisper Typer

Push-to-talk voice transcription using Faster-Whisper. Supports Windows, macOS, and Linux.

## Quick start

1. **Run the client**:

   ```powershell
   uv run client_ui.py
   ```

   `uv` creates the virtual environment and installs dependencies automatically.

2. **Start the server (via UI)**:
   - Go to the **Server** tab.
   - Click **Start Local** (installs server dependencies automatically into the venv if needed).
   - *Alternative:* Click **Start Docker** to run it in a container.

3. **Use it**:
   - Go to the **Transcribe** tab (or minimize the app and use the system tray).
   - Press **Win+G** to start recording.
   - Press **Win+G** again to stop, transcribe, and **automatically type** the text into whatever window you were focused on.

---

## Architecture

This project consists of two parts running together:
1. **Client** (`client_ui.py` / `client.py`): The UI, hotkey listener, and audio recorder.
2. **Server** (`root/app/transcribe_api.py`): A FastAPI server running `faster-whisper`.

You can run both locally using `uv`, or run the server in Docker while running the client locally.

---

## Run (detailed)

### 1. Run the client UI

```powershell
uv run client_ui.py
```

`client_ui.py` is a complete dashboard that lets you:
- **Transcribe**: Pick a model (tiny, base, small, medium, large-v3) and record.
- **Manage Server**: 
  - **Start Local**: Automatically installs dependencies (`fastapi`, `faster-whisper`, etc.) and runs the server.
  - **Start Docker**: Runs `docker compose up -d`.
  - **Model Manager**: Pre-download or delete models to save space.
- **View Logs**: See real-time server output.

### 2. Manual Server Setup (Optional)

If you prefer not to use the UI's server management:

**Manual Local Server:**
```powershell
uv pip install fastapi uvicorn python-multipart faster-whisper numpy
uv run uvicorn root.app.transcribe_api:app --host 127.0.0.1 --port 8000
```

**Manual Docker:**
```powershell
docker compose up -d
```

**NVIDIA GPU on Linux** (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):
```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

---

## Hotkeys & Auto-typing

The client runs a global hotkey listener:

- **Win+G** — Toggle recording.
- When recording is stopped, the client waits for the transcription and then **simulates keyboard typing** to insert the text into the currently focused window.

### Status icon colors

The system tray status icon uses the following colors:

- **Running**: 🟢 green
- **Stopped**: 🔴 red
- **Recording**: 🟠 amber
- **Starting**: 🔵 blue

---

## Requirements

- **OS:** Windows, macOS, or Linux
- **Python:** 3.10+
- **Package manager:** `uv` (recommended)
- **Docker:** Optional, for isolated container deployment

---

## Config

- **.env file**: Create a `.env` file from `.env.example` to set the default `WHISPER_MODEL` or provide an `HF_TOKEN` for faster downloads.
- **Server address:** Configured in `client_ui.py` (`SERVER_IP` / `SERVER_PORT` — default `127.0.0.1:8000`).

---

## API

The server exposes a **FastAPI** endpoint on port **8000**:

- **GET /** — health check and service info
- **POST /transcribe?model=small** — upload raw PCM audio (16 kHz, 16-bit, mono); returns `{"text": "..."}`
