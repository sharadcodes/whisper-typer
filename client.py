"""Whisper Typer — system tray push-to-talk transcription.
Supports: Windows, macOS, Linux (CPU-only or GPU).
"""
import sys
import json
import logging
import threading
import time
import urllib.request
import urllib.error
import urllib.parse

import numpy as np
import sounddevice as sd
import pystray
from PIL import Image, ImageDraw
from pynput.keyboard import Controller as KeyboardController, GlobalHotKeys

# ── Platform detection ────────────────────────────────────────────────────────
PLATFORM = sys.platform  # "win32", "darwin" (macOS), "linux"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("whisper-typer")

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = {
    "1": ("tiny",   "Tiny    — fastest, ~32x real-time  (CPU fine)"),
    "2": ("base",   "Base    — fast, decent accuracy     (CPU fine)"),
    "3": ("small",  "Small   — good balance              (CPU fine)"),
    "4": ("medium", "Medium  — high accuracy, slower     (GPU helps)"),
    "5": ("large-v3", "Large   — best accuracy, slowest    (GPU recommended)"),
}

def pick_model() -> str:
    print("\n┌─ Whisper Typer — Select Model ──────────────────────────────┐")
    for key, (name, desc) in MODELS.items():
        print(f"│  {key}) {desc:<56}│")
    print("│                                                              │")
    print("│  All models run on CPU-only — GPU speeds things up but      │")
    print("│  is not required.                                            │")
    print("└──────────────────────────────────────────────────────────────┘")
    while True:
        choice = input("Enter choice [1-5] (default: 1 tiny): ").strip() or "1"
        if choice in MODELS:
            name, desc = MODELS[choice]
            print(f"✓ Selected: {desc.strip()}\n")
            return name
        print(f"  Invalid choice '{choice}', try again.")

# ── Config ────────────────────────────────────────────────────────────────────
SERVER_IP          = "127.0.0.1"
SERVER_PORT        = 8000  # FastAPI port (Wyoming is 10300)
SAMPLE_RATE        = 16000
MAX_RECORD_SECONDS = 300   # soft cap; stop manually with hotkey

SELECTED_MODEL: str = "tiny"  # overwritten by pick_model() at startup

# ── Audio helpers ──────────────────────────────────────────────────────────────

def trim_trailing_silence(audio: np.ndarray, threshold: float = 0.005) -> np.ndarray:
    flat = audio.flatten()
    mask = np.abs(flat) > threshold
    if not np.any(mask):
        log.debug("Audio is all silence, returning minimal clip")
        return flat[: int(0.1 * SAMPLE_RATE)].reshape(-1, 1).astype(np.float32)
    last = int(np.where(mask)[0][-1])
    log.debug("Trimmed audio to %.2fs", (last + 1) / SAMPLE_RATE)
    return flat[: last + 1].reshape(-1, 1).astype(np.float32)


def _send_via_fastapi(audio: np.ndarray) -> str | None:
    url = f"http://{SERVER_IP}:{SERVER_PORT}/transcribe?model={urllib.parse.quote(SELECTED_MODEL)}"
    log.debug("Connecting to FastAPI at %s", url)
    try:
        pcm = (audio * 32767).astype(np.int16).tobytes()
        log.debug("Sending %d bytes of PCM audio via POST", len(pcm))
        req = urllib.request.Request(
            url,
            data=pcm,
            method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            text = json.loads(resp.read().decode()).get("text", "")
            log.info("Transcript received: %r", text)
            return text
    except Exception as e:
        log.error("FastAPI error: %s", e)
        return f"[Error: {e}]"


def transcribe(audio: np.ndarray) -> str:
    audio = trim_trailing_silence(audio)
    log.debug("Sending audio to FastAPI…")
    text = _send_via_fastapi(audio)
    result = text or "(no transcription)"
    log.info("Final transcription: %r", result)
    return result


# ── Auto-type (cross-platform) ────────────────────────────────────────────────

def type_text(text: str, delay: float = 0.002) -> None:
    keyboard = KeyboardController()
    log.debug("Typing %d chars into active window", len(text))
    time.sleep(0.15)  # let focus settle after hotkey release
    for char in text:
        keyboard.type(char)
        if delay > 0:
            time.sleep(delay)


# ── Tray icon images ───────────────────────────────────────────────────────────

def _make_icon(color: str) -> Image.Image:
    size = 64
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse([4, 4, size - 4, size - 4], fill=color)
    return img


ICON_IDLE      = _make_icon("#4a90d9")   # blue  — ready
ICON_RECORDING = _make_icon("#e74c3c")   # red   — recording
ICON_WORKING   = _make_icon("#8e44ad")   # purple — transcribing


# ── State ─────────────────────────────────────────────────────────────────────

class State:
    is_recording: bool = False
    input_stream: sd.InputStream | None = None
    recording_chunks: list[np.ndarray] = []
    is_speaking: bool = False
    silence_frames: int = 0
    speech_frames: int = 0
    transcribe_guard: threading.Lock = threading.Lock()
    icon: pystray.Icon | None = None

    # Audio processing constants
    # 1.0 seconds of silence triggers transcription. Adjust if you want a shorter/longer pause window.
    silence_seconds: float = 1.5
    speech_threshold: float = 0.01
    min_speech_seconds: float = 0.2
    stream_block_size: int = 512
    audio_lock: threading.Lock = threading.Lock()


_state = State()


# ── Tray actions ───────────────────────────────────────────────────────────────

def _start_recording(icon: pystray.Icon) -> None:
    log.info("▶ Recording started (stop with hotkey)")
    _state.is_recording    = True
    _state.recording_chunks = []
    _state.is_speaking     = False
    _state.silence_frames  = 0
    _state.speech_frames   = 0

    def audio_callback(indata, frames, _time_info, status):
        if status:
            log.debug("Audio callback status: %s", status)
        with _state.audio_lock:
            if not _state.is_recording:
                return

            block = np.array(indata, copy=True)
            if block.size == 0:
                return
            max_amp = float(np.max(np.abs(block)))
            max_speech_frames = int(_state.silence_seconds * SAMPLE_RATE)
            min_speech_frames = int(_state.min_speech_seconds * SAMPLE_RATE)

            if max_amp >= _state.speech_threshold:
                _state.recording_chunks.append(block)
                _state.speech_frames += frames
                _state.silence_frames = 0
                _state.is_speaking = True
                return

            if not _state.is_speaking:
                # Ignore leading/trailing silence while not in an active segment.
                return

            # Keep a small amount of trailing silence so cutoffs are not clipped.
            _state.recording_chunks.append(block)
            _state.silence_frames += frames
            if _state.speech_frames >= min_speech_frames and _state.silence_frames >= max_speech_frames:
                # finalize one speech segment and queue it for transcription
                segment = np.concatenate(_state.recording_chunks, axis=0).reshape(-1, 1).astype(np.float32)
                _state.recording_chunks = []
                _state.is_speaking = False
                _state.silence_frames = 0
                _state.speech_frames = 0

                def worker(audio_segment: np.ndarray):
                    with _state.transcribe_guard:
                        text = transcribe(audio_segment)
                        if text:
                            type_text(text + " ")

                threading.Thread(target=worker, args=(segment,), daemon=True).start()

    _state.input_stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=_state.stream_block_size,
        callback=audio_callback,
    )
    _state.input_stream.start()
    icon.icon  = ICON_RECORDING
    icon.title = "Whisper Typer — recording…"


def _stop_and_transcribe(icon: pystray.Icon) -> None:
    log.info("■ Recording stopped, transcribing…")
    with _state.audio_lock:
        _state.is_recording = False
        if _state.input_stream:
            _state.input_stream.stop()
            _state.input_stream.close()
        _state.input_stream = None
        chunks = _state.recording_chunks
        _state.recording_chunks = []
        _state.is_speaking = False
        _state.silence_frames = 0
        _state.speech_frames = 0

    recording = np.concatenate(chunks, axis=0).reshape(-1, 1).astype(np.float32) if chunks else None
    sd.stop()
    _state.is_recording    = False
    icon.icon  = ICON_WORKING
    icon.title = "Whisper Typer — transcribing…"

    def worker():
        if recording is not None:
            if not np.any(np.abs(recording) >= _state.speech_threshold):
                log.debug("No speech in final segment; skipping transcription.")
            else:
                with _state.transcribe_guard:
                    text = transcribe(recording)
                    if text:
                        type_text(text)
        icon.icon  = ICON_IDLE
        icon.title = "Whisper Typer — Alt+PageUp to record"
        log.info("✓ Done")

    threading.Thread(target=worker, daemon=True).start()


def on_record_toggle(icon: pystray.Icon, item=None) -> None:
    log.debug("Toggle called — is_recording=%s", _state.is_recording)
    if _state.is_recording:
        _stop_and_transcribe(icon)
    else:
        _start_recording(icon)


def start_hotkey_listener():
    def on_activate():
        log.debug("Hotkey Alt+PageUp fired")
        if _state.icon:
            on_record_toggle(_state.icon)

    hotkeys = GlobalHotKeys({"<alt>+<page_up>": on_activate})
    hotkeys.daemon = True
    hotkeys.start()
    log.info("Hotkey listener started — Alt+PageUp to toggle recording")


def on_quit(icon: pystray.Icon, item=None) -> None:
    log.info("Quitting…")
    sd.stop()
    icon.stop()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    global SELECTED_MODEL
    log.info("Platform: %s", PLATFORM)
    SELECTED_MODEL = pick_model()
    log.info("Model: %s  |  FastAPI: %s:%d", SELECTED_MODEL, SERVER_IP, SERVER_PORT)

    start_hotkey_listener()

    menu = pystray.Menu(
        pystray.MenuItem("Record / Stop (Alt+PageUp)", on_record_toggle, default=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", on_quit),
    )

    icon = pystray.Icon(
        name="whisper-typer",
        icon=ICON_IDLE,
        title="Whisper Typer — Alt+PageUp to record",
        menu=menu,
    )

    _state.icon = icon
    log.info("Tray icon running — press Alt+PageUp or click the tray icon.")
    icon.run()


if __name__ == "__main__":
    main()
