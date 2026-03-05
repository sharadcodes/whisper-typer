# Whisper Typer

Push-to-talk voice transcription using Faster-Whisper. System tray app + hotkey; supports Windows, macOS, and Linux.

## Quick start

1. **Start the transcribe server** (Docker):

   ```powershell
   # Windows
   .\start-transcribe-docker.ps1
   ```

   ```sh
   # macOS / Linux
   chmod +x start-transcribe-docker.sh && ./start-transcribe-docker.sh
   ```

   On Windows: uses WSL when available (for GPU). On Linux: detects GPU. Use `-NoWsl` to force CPU on Windows.

2. **Run the client**:

   ```powershell
   uv run client.py
   ```

   `uv` creates venv and installs dependencies automatically.

3. **Use it** — tray icon appears. Press **Alt+PageUp** to record, press again to stop and transcribe. Text is typed into the focused field.

---

## Scripts

| Script / File | Purpose |
|-------|---------|
| `start-transcribe-docker.ps1` / `start-transcribe-docker.sh` | Start transcribe server in Docker. |
| `start-transcribe-local.ps1` / `start-transcribe-local.sh` | Start transcribe server locally (no Docker). Run `setup-transcribe-local` first. |
| `setup-transcribe-local.ps1` / `setup-transcribe-local.sh` | One-time setup: venv, deps, model download. For running without Docker. |
| `client.py` | Tray + hotkey client — records on Alt+PageUp and auto-types the result. |
| `client_ui.py` | Window UI client — record button, shows transcription, copy/clear. |

---

## Run (detailed)

### 1. Start the Faster-Whisper server (Docker)

**Helper script** (recommended):

```powershell
# Windows
.\start-transcribe-docker.ps1
```

```sh
# macOS / Linux
./start-transcribe-docker.sh
```

**Local (no Docker):**

```powershell
# Windows — run setup-transcribe-local.ps1 first
.\start-transcribe-local.ps1
```

```sh
# macOS / Linux — run setup-transcribe-local.sh first
chmod +x start-transcribe-local.sh && ./start-transcribe-local.sh
```

**Manual Docker:**

```powershell
docker compose up -d --build
```

**NVIDIA GPU on Linux** (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

**NVIDIA GPU on Windows (WSL2):** `start-transcribe-docker.ps1` auto-uses WSL when available. One-time setup: install [NVIDIA Container Toolkit in WSL2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian). Use `.\start-transcribe-docker.ps1 -NoWsl` to force CPU mode.

Logs: `docker compose logs -f faster-whisper`

### 2. Run the client

**Tray + hotkey client** (Alt+PageUp to record, auto-types result):

```powershell
uv run client.py
```

**Window UI client** (record button, shows text, copy/clear — no hotkeys):

```powershell
uv run client_ui.py
```

Or with standard Python (after `uv venv` and `uv pip install .`):

```powershell
.venv\Scripts\Activate.ps1   # Windows
python client.py       # or: python client_ui.py
```

```sh
source .venv/bin/activate     # macOS / Linux
python client.py       # or: python client_ui.py
```

On first run of `client.py` you pick a model (1–5) at the CLI; then the tray starts. `client_ui.py` has a model selector in the window.

### 3. Use the system tray or keyboard shortcut

- **Blue** — ready
- **Red** — recording
- **Amber** — transcribing

**Alt+PageUp** — toggle recording. Or left-click the tray icon (or right-click → Record / Stop).

When done, the transcription is **automatically typed** into the focused input field. A toast notification shows the result.

---

## Requirements

- **OS:** Windows, macOS, or Linux
- **Python:** 3.x
- **Package manager:** `uv` (recommended) or `pip`
- **Dependencies:** sounddevice, numpy, pystray, pillow, pynput
- **Docker:** for the faster-whisper backend (Windows, macOS including Apple Silicon M1–M4, Linux)

---

## Config

- **Server:** In `client.py`, `SERVER_IP` / `SERVER_PORT` (default `127.0.0.1:8000`)
- **Model:** Chosen at app startup (CLI prompt 1–5)
- **GPU:** Linux: `docker-compose.gpu.yml` + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Windows: `start-transcribe-docker.ps1` uses WSL when available; install toolkit in WSL2.

---

## API

The Docker container runs a **FastAPI** server on port **8000**:

- **GET /** — service info
- **POST /transcribe?model=tiny** — upload raw PCM audio (16 kHz, 16-bit, mono); returns `{"text": "..."}`
