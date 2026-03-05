import os
import sys
from pathlib import Path

# ── Package & Data Paths ──────────────────────────────────────────────────────
# Internal package directory
PACKAGE_DIR = Path(__file__).resolve().parent.parent
# Server directory inside the package
SERVER_DIR = PACKAGE_DIR / "server"

# User Data Directory (Standard for PyPI packages)
# e.g., C:\Users\Name\.whisper-typer or /home/name/.whisper-typer
DATA_DIR = Path.home() / ".whisper-typer"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_FILE = str(DATA_DIR / "history.json")
# Allow override via environment variable
MODELS_DIR = os.environ.get("WHISPER_MODELS_DIR", str(DATA_DIR / "models"))

# ── Server Config ─────────────────────────────────────────────────────────────
SERVER_IP = "127.0.0.1"
SERVER_PORT = 8000
HEALTH_POLL_SEC = 3

# ── Audio Parameters ──────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 300
SILENCE_WINDOW_SECONDS = 1.5   # pause after this long to flush a chunk
SPEECH_THRESHOLD = 0.01        # RMS/max amplitude threshold to detect speech
MIN_SPEECH_SECONDS = 0.2       # ignore tiny mouth-noise/ clicks
STREAM_BLOCK_SIZE = 512        # input callback block size in frames

# ── App Logic ─────────────────────────────────────────────────────────────────
MODELS = ["tiny", "base", "small", "medium", "large-v3"]
TRANSCRIBE_MODE_LIVE = "Live typing"
TRANSCRIBE_MODE_BATCH = "Full Capture"
TRANSCRIBE_MODES = [TRANSCRIBE_MODE_LIVE, TRANSCRIBE_MODE_BATCH]

def get_venv_python() -> str | None:
    """Path to current Python interpreter."""
    # When installed, we usually just want the current sys.executable
    # which is the one managing the environment.
    if getattr(sys, 'frozen', False):
        # If running as an EXE
        return sys.executable
    
    # Check if we are in a virtual environment
    if sys.prefix != sys.base_prefix:
        return sys.executable
        
    return sys.executable
