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

## UI Tabs

### Transcribe

- Pick a model (tiny, base, small, medium, large-v3) and record.
- Press the record button or use **Win+G** to toggle recording.
- Transcription result is displayed and automatically typed into the focused window.

### History

- Every successful transcription is saved with a timestamp and the model used.
- Entries persist across app restarts in a local `history.json` file.
- Copy any past transcription to the clipboard with one click.
- Clear all history with the **Clear All** button.

### Server

- **Status card** shows the server's live status (Online / Offline), the server URL, and the active model.
- **Start Local**: Automatically installs dependencies (`fastapi`, `faster-whisper`, etc.) and runs the server.
- **Start Docker**: Runs `docker compose up -d`.
- **Model Manager**: Pre-download or delete models to save space.
- **Logs**: Real-time server output with a clear button.

---

## Manual Server Setup (Optional)

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

- **Win+G** (Windows) or **Cmd+G** (macOS) — Toggle recording.
- When recording is stopped, the client waits for the transcription and then **simulates keyboard typing** to insert the text into the currently focused window.

> **macOS Users:** 
> 1. You must grant **Accessibility** permissions to your terminal (e.g., iTerm or Terminal.app) for the auto-typing to work.
> 2. Grant **Microphone** permissions when prompted.

### System tray icon colors

| State | Color | Meaning |
|-------|-------|---------|
| Idle (server online) | 🟢 Green | Server is running, ready to transcribe |
| Server offline | ⚫ Black | Server is not reachable |
| Recording | 🔴 Red | Audio is being captured |
| Processing | 🟣 Purple | Transcribing audio |

---

## Default Model

The default Whisper model is **small** (good balance of speed and accuracy on CPU). Override it by setting `WHISPER_MODEL` in your `.env` file.

---

## Requirements

- **OS:** Windows, macOS, or Linux
- **Python:** 3.10+
- **Package manager:** [uv](https://github.com/astral-sh/uv) (recommended)
- **Docker:** Optional, for isolated container deployment

---

## Installation

1. **Install `uv`** (if you haven't already):
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   *For macOS/Linux:*
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repo**:
   ```bash
   git clone https://github.com/sharadcodes/whisper-typer.git
   cd whisper-typer
   ```

3. **Run the app**:
   ```bash
   uv run client_ui.py
   ```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
