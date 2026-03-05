import os
import sys

# ── Project Paths ─────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_FILE = os.path.join(ROOT_DIR, "history.json")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

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
TRANSCRIBE_MODE_BATCH = "Recording only"
TRANSCRIBE_MODES = [TRANSCRIBE_MODE_LIVE, TRANSCRIBE_MODE_BATCH]

LOCAL_SERVER_PACKAGES = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("python-multipart", "multipart"),
    ("faster-whisper", "faster_whisper"),
    ("numpy", "numpy"),
]

def get_venv_python() -> str | None:
    """Path to project venv Python, or None if no venv found."""
    if sys.platform == "win32":
        for name in (".venv", "venv"):
            exe = os.path.join(ROOT_DIR, name, "Scripts", "python.exe")
            if os.path.isfile(exe):
                return exe
    else:
        for name in (".venv", "venv"):
            exe = os.path.join(ROOT_DIR, name, "bin", "python")
            if os.path.isfile(exe):
                return exe
    return None
