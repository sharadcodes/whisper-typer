"""Whisper Typer — UI client.
Three tabs:
  • Transcribe — record audio, send to server, display result.
  • History    — timestamped log of every transcription.
  • Server     — start/stop server (local or Docker), live log stream.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import logging
import time
from queue import Queue
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime

import numpy as np
import sounddevice as sd
from pynput.keyboard import Controller as KeyboardController, GlobalHotKeys
import customtkinter as ctk
from PIL import Image, ImageDraw

# ── Config ────────────────────────────────────────────────────────────────────
SERVER_IP          = "127.0.0.1"
SERVER_PORT        = 8000
SAMPLE_RATE        = 16000
MAX_RECORD_SECONDS = 300
HEALTH_POLL_SEC    = 3
SILENCE_WINDOW_SECONDS = 1.5   # pause after this long to flush a chunk
SPEECH_THRESHOLD   = 0.01     # RMS/max amplitude threshold to detect speech
MIN_SPEECH_SECONDS = 0.2      # ignore tiny mouth-noise/ clicks
STREAM_BLOCK_SIZE  = 512      # input callback block size in frames

ROOT_DIR      = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE  = os.path.join(ROOT_DIR, "history.json")
MODELS   = ["tiny", "base", "small", "medium", "large-v3"]
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


def _get_venv_python() -> str | None:
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


def _python_has_module(python: str, module: str) -> bool:
    check = subprocess.run(
        [python, "-c", f"import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('{module}') else 1)"],
        capture_output=True,
        text=True,
    )
    return check.returncode == 0


# ── Audio helpers ─────────────────────────────────────────────────────────────

def trim_trailing_silence(audio: np.ndarray, threshold: float = 0.005) -> np.ndarray:
    flat = audio.flatten()
    mask = np.abs(flat) > threshold
    if not np.any(mask):
        return flat[: int(0.1 * SAMPLE_RATE)].reshape(-1, 1).astype(np.float32)
    last = int(np.where(mask)[0][-1])
    return flat[: last + 1].reshape(-1, 1).astype(np.float32)


def _make_status_icon(status: str) -> Image.Image:
    """Create a small status icon for the system tray."""
    size = 32
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    color_map = {
        "running": "#27ae60",       # green  — server online
        "stopped": "#000000",       # black  — server offline
        "recording": "#e74c3c",     # red    — recording
        "processing": "#8e44ad",    # purple — transcribing
    }
    color = color_map.get(status, "gray50")
    
    # Convert hex to RGB
    color = color.lstrip("#")
    r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
    
    draw.ellipse([4, 4, size - 4, size - 4], fill=(r, g, b, 255))
    return img


def send_to_server(audio: np.ndarray, model: str) -> str:
    url = f"http://{SERVER_IP}:{SERVER_PORT}/transcribe?model={urllib.parse.quote(model)}"
    try:
        pcm = (audio * 32767).astype(np.int16).tobytes()
        req = urllib.request.Request(
            url,
            data=pcm,
            method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode()).get("text", "")
    except urllib.error.URLError as e:
        return f"[Server unreachable — start it in the Server tab. {e}]"
    except Exception as e:
        return f"[Error: {e}]"


# ── App ───────────────────────────────────────────────────────────────────────

class WhisperUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Whisper Typer")
        self.geometry("960x680")
        self.minsize(420, 540)

        # recording / processing state
        self._recording           = False
        self._processing          = False
        self._processing_count    = 0
        self._processing_lock     = threading.Lock()
        self._transcribe_lock     = threading.Lock()
        self._transcribe_queue: Queue = Queue()
        self._recording_stream: sd.InputStream | None = None
        self._recording_chunks: list[np.ndarray] = []
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0
        self._audio_lock = threading.Lock()
        self._last_transcription = ""
        self._transcribe_mode_var = ctk.StringVar(value=TRANSCRIBE_MODE_LIVE)

        # transcript history: list of {"time": str, "model": str, "text": str}
        self._history: list[dict] = self._load_history()

        # server process handles
        self._local_proc: subprocess.Popen | None = None   # local uvicorn
        self._docker_log_proc: subprocess.Popen | None = None  # docker logs -f

        # last known server health
        self._server_running = False
        self._auto_server_start_attempted = False

        self._build_ui()
        self._setup_tray()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._set_srv_state("stopped")
        threading.Thread(target=self._health_worker, daemon=True).start()
        threading.Thread(target=self._transcription_queue_worker, daemon=True).start()
        threading.Thread(target=self._auto_start_server_if_not_running, daemon=True).start()
        
        # Start hotkey listener for Win+G (record)
        threading.Thread(target=self._hotkey_listener, daemon=True).start()
        
        # Update tray icon periodically
        self._update_tray_icon()

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.grid(row=0, column=0, padx=20, pady=(18, 0), sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            hdr, text="Whisper Typer",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        self._hdr_status = ctk.CTkLabel(
            hdr, text="● Server Offline",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#000000",
        )
        self._hdr_status.grid(row=0, column=1, sticky="e")

        # ── Tabs ──────────────────────────────────────────────────────────────
        self._tabs = ctk.CTkTabview(self, corner_radius=10)
        self._tabs.grid(row=1, column=0, padx=16, pady=(10, 16), sticky="nsew")

        self._tab_t = self._tabs.add("  Transcribe  ")
        self._tab_h = self._tabs.add("  History  ")
        self._tab_s = self._tabs.add("  Server  ")

        self._build_transcribe_tab()
        self._build_history_tab()
        self._build_server_tab()

    # ── Transcribe tab ────────────────────────────────────────────────────────

    def _build_transcribe_tab(self):
        tab = self._tab_t
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(4, weight=1)

        # Model row
        model_row = ctk.CTkFrame(tab, fg_color="transparent")
        model_row.grid(row=0, column=0, pady=(8, 14), sticky="ew")
        ctk.CTkLabel(model_row, text="Model", font=ctk.CTkFont(size=13)).pack(side="left", padx=(0, 10))
        self._model_var = ctk.StringVar(value="small")
        ctk.CTkOptionMenu(model_row, values=MODELS, variable=self._model_var, width=160).pack(side="left")

        # Recording and utility buttons
        control_row = ctk.CTkFrame(tab, fg_color="transparent")
        control_row.grid(row=1, column=0, pady=(0, 16), sticky="ew")
        control_row.grid_columnconfigure((0, 1), weight=1)

        self._btn_record = ctk.CTkButton(
            control_row,
            text="⏺  Start Recording",
            font=ctk.CTkFont(size=15, weight="bold"),
            height=52,
            corner_radius=12,
            command=self._toggle_recording,
        )
        self._btn_record.grid(row=0, column=0, sticky="ew")
        self._btn_rec_fg    = self._btn_record.cget("fg_color")
        self._btn_rec_hover = self._btn_record.cget("hover_color")

        self._btn_clear = ctk.CTkButton(
            control_row,
            text="🧹  Clear Text",
            font=ctk.CTkFont(size=15, weight="bold"),
            height=52,
            corner_radius=12,
            fg_color="#636e72",
            hover_color="#4f585d",
            command=self._clear_transcript_text,
        )
        self._btn_clear.grid(row=0, column=1, padx=(10, 0), sticky="ew")

        # Input mode row
        mode_row = ctk.CTkFrame(tab, fg_color="transparent")
        mode_row.grid(row=2, column=0, pady=(0, 12), sticky="ew")
        mode_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            mode_row, text="Input mode", font=ctk.CTkFont(size=12),
        ).grid(row=0, column=0, sticky="w")
        self._mode_selector = ctk.CTkSegmentedButton(
            mode_row,
            values=TRANSCRIBE_MODES,
            command=self._on_transcribe_mode_change,
            width=300,
            corner_radius=10,
        )
        self._mode_selector.set(TRANSCRIBE_MODE_LIVE)
        self._mode_selector.grid(row=0, column=1, padx=(12, 0), sticky="w")

        # Transcription label
        ctk.CTkLabel(
            tab, text="Transcription",
            font=ctk.CTkFont(size=12),
            text_color="gray60",
        ).grid(row=3, column=0, sticky="w")

        # Transcription box
        self._textbox = ctk.CTkTextbox(
            tab,
            font=ctk.CTkFont(size=14),
            corner_radius=8,
            wrap="word",
        )
        self._textbox.grid(row=4, column=0, pady=(4, 8), sticky="nsew")
        self._textbox.configure(state="disabled")

        # Status label
        self._tx_status = ctk.CTkLabel(
            tab, text="Ready. Press Win+G to start recording.",
            font=ctk.CTkFont(size=11),
            text_color="gray55",
        )
        self._tx_status.grid(row=5, column=0, pady=(8, 0), sticky="w")

    def _clear_transcript_text(self):
        self._textbox.configure(state="normal")
        self._textbox.delete("1.0", "end")
        self._textbox.configure(state="disabled")
        self._last_transcription = ""
        self._set_tx_status("Transcript cleared.", "gray55")

    def _is_live_mode(self) -> bool:
        return self._transcribe_mode_var.get() == TRANSCRIBE_MODE_LIVE

    def _on_transcribe_mode_change(self, _value: str | None = None):
        if _value is not None:
            self._transcribe_mode_var.set(_value)
            if self._recording:
                self._mode_selector.set(self._transcribe_mode_var.get())
                return

        if self._recording:
            return
        if self._is_live_mode():
            self._set_tx_status("Ready. Press Win+G to start recording.", "gray55")
        else:
            self._set_tx_status("Recording only mode: stop to send.", "gray55")

    # ── History persistence ─────────────────────────────────────────────────

    @staticmethod
    def _load_history() -> list[dict]:
        if os.path.isfile(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return []

    @staticmethod
    def _save_history_to_disk(history: list[dict]):
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    # ── History tab ────────────────────────────────────────────────────────────

    def _build_history_tab(self):
        tab = self._tab_h
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Header row: count badge + clear button
        hdr = ctk.CTkFrame(tab, fg_color="transparent")
        hdr.grid(row=0, column=0, pady=(8, 6), sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)

        self._hist_count_label = ctk.CTkLabel(
            hdr, text="0 transcriptions",
            font=ctk.CTkFont(size=12),
            text_color="gray55",
        )
        self._hist_count_label.grid(row=0, column=0, sticky="w")

        ctk.CTkButton(
            hdr, text="Clear All", width=80, height=26,
            font=ctk.CTkFont(size=11),
            fg_color="#c0392b", hover_color="#a93226",
            command=self._clear_history,
        ).grid(row=0, column=1, sticky="e")

        # Scrollable area for history entries
        self._hist_scroll = ctk.CTkScrollableFrame(
            tab, corner_radius=8, fg_color="transparent",
        )
        self._hist_scroll.grid(row=1, column=0, sticky="nsew")
        self._hist_scroll.grid_columnconfigure(0, weight=1)

        # Empty-state label (shown when no history)
        self._hist_empty = ctk.CTkLabel(
            self._hist_scroll,
            text="No transcriptions yet.\nRecord something to see it here.",
            font=ctk.CTkFont(size=13),
            text_color="gray45",
            justify="center",
        )

        # Replay persisted history into the UI
        if self._history:
            for entry in self._history:
                self._render_history_card(entry)
            n = len(self._history)
            self._hist_count_label.configure(
                text=f"{n} transcription{'s' if n != 1 else ''}"
            )
        else:
            self._hist_empty.grid(row=0, column=0, pady=60)

    def _render_history_card(self, entry: dict, idx: int | None = None):
        """Create a card widget for a single history entry."""
        if idx is None:
            idx = len(self._hist_scroll.winfo_children())

        card = ctk.CTkFrame(self._hist_scroll, corner_radius=8)
        card.grid(row=idx, column=0, pady=(0, 8), sticky="ew")
        card.grid_columnconfigure(0, weight=1)

        meta = ctk.CTkFrame(card, fg_color="transparent")
        meta.grid(row=0, column=0, padx=12, pady=(10, 0), sticky="ew")
        meta.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            meta, text=entry["time"],
            font=ctk.CTkFont(size=11),
            text_color="gray55",
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            meta, text=entry["model"],
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="gray50",
            corner_radius=3,
            fg_color="gray25",
            padx=6,
        ).grid(row=0, column=1, sticky="e")

        ctk.CTkLabel(
            card, text=entry["text"],
            font=ctk.CTkFont(size=13),
            wraplength=380,
            justify="left",
            anchor="w",
        ).grid(row=1, column=0, padx=12, pady=(6, 4), sticky="ew")

        def _copy(t=entry["text"]):
            self.clipboard_clear()
            self.clipboard_append(t)
            self._set_tx_status("Copied to clipboard.", "gray55")

        ctk.CTkButton(
            card, text="Copy", width=60, height=24,
            font=ctk.CTkFont(size=11),
            fg_color="gray30", hover_color="gray25",
            command=_copy,
        ).grid(row=2, column=0, padx=12, pady=(2, 10), sticky="w")

    def _add_history_entry(self, text: str, model: str):
        """Append a transcription to history, update UI, and persist."""
        ts = datetime.now().strftime("%I:%M %p").lstrip("0")
        date_str = datetime.now().strftime("%b %d")
        entry = {"time": f"{date_str}, {ts}", "model": model, "text": text}
        self._history.append(entry)

        self._hist_empty.grid_forget()
        self._render_history_card(entry, idx=len(self._history) - 1)

        n = len(self._history)
        self._hist_count_label.configure(
            text=f"{n} transcription{'s' if n != 1 else ''}"
        )
        self._hist_scroll._parent_canvas.yview_moveto(1.0)

        self._save_history_to_disk(self._history)

    def _clear_history(self):
        """Remove all history entries from UI and disk."""
        self._history.clear()
        for widget in self._hist_scroll.winfo_children():
            widget.destroy()

        self._hist_empty = ctk.CTkLabel(
            self._hist_scroll,
            text="No transcriptions yet.\nRecord something to see it here.",
            font=ctk.CTkFont(size=13),
            text_color="gray45",
            justify="center",
        )
        self._hist_empty.grid(row=0, column=0, pady=60)
        self._hist_count_label.configure(text="0 transcriptions")

        try:
            os.remove(HISTORY_FILE)
        except OSError:
            pass

    # ── Server tab ────────────────────────────────────────────────────────────

    def _build_server_tab(self):
        tab = self._tab_s
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(3, weight=1)

        # ── Status card ──────────────────────────────────────────────────
        card = ctk.CTkFrame(tab, corner_radius=10)
        card.grid(row=0, column=0, pady=(8, 8), sticky="ew")
        card.grid_columnconfigure(1, weight=1)

        # Colored status bar (left edge indicator)
        self._srv_status_bar = ctk.CTkFrame(card, width=6, corner_radius=3, fg_color="gray50")
        self._srv_status_bar.grid(row=0, column=0, rowspan=3, padx=(8, 0), pady=10, sticky="ns")

        # Status title row
        status_top = ctk.CTkFrame(card, fg_color="transparent")
        status_top.grid(row=0, column=1, padx=(12, 16), pady=(12, 0), sticky="ew")
        status_top.grid_columnconfigure(0, weight=1)

        self._srv_status_label = ctk.CTkLabel(
            status_top, text="Checking…",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="gray50",
        )
        self._srv_status_label.grid(row=0, column=0, sticky="w")

        self._srv_status_badge = ctk.CTkLabel(
            status_top, text="  OFFLINE  ",
            font=ctk.CTkFont(size=10, weight="bold"),
            corner_radius=4,
            fg_color="gray35",
            text_color="gray70",
        )
        self._srv_status_badge.grid(row=0, column=1, sticky="e")

        # Info row (URL + model)
        info_row = ctk.CTkFrame(card, fg_color="transparent")
        info_row.grid(row=1, column=1, padx=(12, 16), pady=(4, 0), sticky="ew")

        self._srv_url_label = ctk.CTkLabel(
            info_row,
            text=f"http://{SERVER_IP}:{SERVER_PORT}",
            font=ctk.CTkFont(family="Courier New", size=11),
            text_color="gray55",
        )
        self._srv_url_label.pack(side="left")

        self._srv_model_label = ctk.CTkLabel(
            info_row,
            text="model: small",
            font=ctk.CTkFont(size=11),
            text_color="gray55",
        )
        self._srv_model_label.pack(side="right")

        # Control buttons
        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.grid(row=2, column=1, padx=(8, 12), pady=(8, 12), sticky="ew")

        self._btn_start_local = ctk.CTkButton(
            btn_row, text="▶  Start Local", width=130, height=32,
            font=ctk.CTkFont(size=12),
            command=self._start_local,
        )
        self._btn_start_local.pack(side="left", padx=(0, 6))

        self._btn_start_docker = ctk.CTkButton(
            btn_row, text="🐳  Start Docker", width=140, height=32,
            font=ctk.CTkFont(size=12),
            fg_color="#1a6b3a", hover_color="#145530",
            command=self._start_docker,
        )
        if shutil.which("docker"):
            self._btn_start_docker.pack(side="left", padx=(0, 6))

        self._btn_stop_srv = ctk.CTkButton(
            btn_row, text="Stop", width=70, height=32,
            font=ctk.CTkFont(size=12),
            fg_color="gray35", hover_color="gray28",
            command=self._stop_server,
        )
        self._btn_stop_srv.pack(side="left", padx=(0, 6))

        self._btn_kill_srv = ctk.CTkButton(
            btn_row, text="Force Kill", width=90, height=32,
            font=ctk.CTkFont(size=12),
            fg_color="#c0392b", hover_color="#a93226",
            command=self._kill_server,
        )
        self._btn_kill_srv.pack(side="left")

        # Model Downloader card
        dl_card = ctk.CTkFrame(tab, corner_radius=10)
        dl_card.grid(row=1, column=0, pady=(0, 12), sticky="ew")
        dl_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(dl_card, text="Model Manager", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, columnspan=3, padx=16, pady=(10, 6), sticky="w")
        
        self._model_dl_var = ctk.StringVar(value="small")
        self._model_dl_menu = ctk.CTkOptionMenu(dl_card, values=MODELS, variable=self._model_dl_var, width=120)
        self._model_dl_menu.grid(row=1, column=0, padx=(16, 8), pady=(0, 14), sticky="w")

        self._btn_download = ctk.CTkButton(
            dl_card, text="Download",
            command=self._download_model,
        )
        self._btn_download.grid(row=1, column=1, padx=(8, 8), pady=(0, 14), sticky="ew")

        self._btn_delete = ctk.CTkButton(
            dl_card, text="Delete",
            fg_color="#c0392b", hover_color="#a93226",
            command=self._delete_model,
        )
        self._btn_delete.grid(row=1, column=2, padx=(0, 16), pady=(0, 14), sticky="ew")

        # Log header
        log_hdr = ctk.CTkFrame(tab, fg_color="transparent")
        log_hdr.grid(row=2, column=0, sticky="ew")
        ctk.CTkLabel(log_hdr, text="Logs", font=ctk.CTkFont(size=12), text_color="gray60").pack(side="left")
        ctk.CTkButton(
            log_hdr, text="Clear", width=64, height=24,
            font=ctk.CTkFont(size=11),
            fg_color="gray30", hover_color="gray25",
            command=self._clear_logs,
        ).pack(side="right")

        # Log box
        self._log_box = ctk.CTkTextbox(
            tab,
            font=ctk.CTkFont(family="Courier New", size=11),
            corner_radius=8,
            wrap="none",
        )
        self._log_box.grid(row=3, column=0, pady=(4, 0), sticky="nsew")
        self._log_box.configure(state="disabled")

    # ── Local dependency bootstrap ───────────────────────────────────────────────

    def _ensure_local_server_deps(self, python: str) -> bool:
        missing: list[str] = []
        for pkg_name, module_name in LOCAL_SERVER_PACKAGES:
            if not _python_has_module(python, module_name):
                missing.append(pkg_name)

        if not missing:
            return True

        self._log(f"Missing packages in local venv: {', '.join(missing)}\n")
        self._log("Installing missing packages now…\n")

        uv_exe = shutil.which("uv")
        if uv_exe:
            self._log("Using uv to install local dependencies.\n")
            install = subprocess.run(
                [uv_exe, "pip", "install", "--python", python, *missing],
                capture_output=True,
                text=True,
            )
            if install.returncode != 0:
                output = (install.stdout + install.stderr).strip() or "(no output)"
                self._log(f"Failed to install dependencies with uv:\n{output}\n")
                self._log("Please run 'uv venv' to create a virtual environment.\n")
                return False
            self._log("Local dependency install complete.\n")
            return True

        self._log("uv command not found in PATH. Trying python -m ensurepip + pip.\n")
        pip_bootstrap = subprocess.run([python, "-m", "ensurepip", "--upgrade"], capture_output=True, text=True)
        if pip_bootstrap.returncode != 0:
            output = (pip_bootstrap.stdout + pip_bootstrap.stderr).strip() or "(no output)"
            self._log(f"Failed to initialize pip:\n{output}\n")
            self._log("Install uv and add it to PATH.\n")
            return False

        install = subprocess.run([python, "-m", "pip", "install", *missing], capture_output=True, text=True)
        if install.returncode != 0:
            output = (install.stdout + install.stderr).strip() or "(no output)"
            self._log(f"Failed to install dependencies:\n{output}\n")
            self._log("Run setup-transcribe-local.ps1/.sh for a full install.\n")
            return False

        self._log("Local dependency install complete.\n")
        return True

    # ══════════════════════════════════════════════════════════════════════════
    # SERVER MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════════

    def _start_local(self):
        if self._local_proc and self._local_proc.poll() is None:
            self._log("Server is already running locally.\n")
            return

        python = _get_venv_python()
        if not python:
            self._log("No venv found. Please run 'uv run client_ui.py' to automatically create one and install deps.\n")
            return

        self._btn_start_local.configure(state="disabled")

        def run_srv():
            # Load environment variables from .env if it exists
            env_vars = os.environ.copy()
            env_path = os.path.join(ROOT_DIR, ".env")
            if os.path.isfile(env_path):
                self.after(0, self._log, "Loading environment from .env\n")
                with open(env_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            env_vars[k.strip()] = v.strip()

            if not self._ensure_local_server_deps(python):
                self.after(0, lambda: self._set_srv_state("stopped"))
                self.after(0, lambda: self._btn_start_local.configure(state="normal"))
                return

            if not os.path.isfile(python):
                self.after(0, self._log, f"Resolved python path does not exist anymore: {python}\n")
                self.after(0, lambda: self._set_srv_state("stopped"))
                self.after(0, lambda: self._btn_start_local.configure(state="normal"))
                return

            server_dir = os.path.join(ROOT_DIR, "root", "app")
            self.after(0, self._log, f"Using venv: {python}\n")
            self.after(0, self._log, "Starting local server…\n")

            try:
                self._local_proc = subprocess.Popen(
                    [
                        python, "-m", "uvicorn", "transcribe_api:app",
                        "--host", SERVER_IP, "--port", str(SERVER_PORT),
                    ],
                    cwd=server_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env_vars, # Pass loaded .env vars
                )
                self.after(0, lambda: self._btn_start_local.configure(state="normal"))
                self._stream_logs(self._local_proc)
            except Exception as e:
                self.after(0, self._log, f"Failed to start: {e}\n")
                self.after(0, lambda: self._set_srv_state("stopped"))
                self.after(0, lambda: self._btn_start_local.configure(state="normal"))
                return

        threading.Thread(target=run_srv, daemon=True).start()

    def _start_docker(self):
        self._log("docker compose up -d --build\n")

        def run():
            try:
                proc = subprocess.Popen(
                    ["docker", "compose", "up", "-d", "--build"],
                    cwd=ROOT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in proc.stdout:
                    self.after(0, self._log, line)
                proc.wait()
                if proc.returncode == 0:
                    self.after(0, self._log, "Container started. Tailing logs…\n")
                    self.after(0, self._tail_docker_logs)
                else:
                    self.after(0, self._log, f"docker compose exited with code {proc.returncode}\n")
                    self.after(0, self._log, "Troubleshooting: Ensure Docker Desktop is running and the daemon is active.\n")
            except FileNotFoundError:
                self.after(0, self._log, "docker not found — is Docker Desktop installed and in PATH?\n")
            except Exception as e:
                self.after(0, self._log, f"Docker error: {e}\n")

        threading.Thread(target=run, daemon=True).start()

    def _tail_docker_logs(self):
        try:
            self._docker_log_proc = subprocess.Popen(
                ["docker", "compose", "logs", "-f", "--tail", "50", "faster-whisper"],
                cwd=ROOT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            threading.Thread(
                target=self._stream_logs, args=(self._docker_log_proc,), daemon=True
            ).start()
        except Exception as e:
            self._log(f"Failed to tail docker logs: {e}\n")

    def _stop_server(self):
        """Gracefully stop the server (terminate)."""
        stopped_something = False

        if self._local_proc and self._local_proc.poll() is None:
            self._local_proc.terminate()
            self._local_proc = None
            self._log("Local server stopped (graceful).\n")
            stopped_something = True

        if self._docker_log_proc and self._docker_log_proc.poll() is None:
            self._docker_log_proc.terminate()
            self._docker_log_proc = None

        # Only stop the Docker container if we didn't have a local proc
        if not stopped_something:
            self._log("docker compose stop (graceful)…\n")
            def run_stop():
                subprocess.run(
                    ["docker", "compose", "stop"],
                    cwd=ROOT_DIR,
                    capture_output=True,
                )
                self.after(0, self._log, "Docker container stopped.\n")
            threading.Thread(target=run_stop, daemon=True).start()

        self._set_srv_state("stopped")

    def _kill_server(self):
        """Forcefully kill the server (kill -9 / terminate + kill)."""
        killed_something = False

        if self._local_proc and self._local_proc.poll() is None:
            self._log("Force killing local server…\n")
            try:
                import signal
                if hasattr(signal, 'SIGKILL'):
                    os.kill(self._local_proc.pid, signal.SIGKILL)
                else:
                    # Windows: use taskkill
                    subprocess.run(
                        ["taskkill", "/PID", str(self._local_proc.pid), "/F"],
                        capture_output=True,
                    )
                self._local_proc = None
                self._log("Local server force killed.\n")
                killed_something = True
            except Exception as e:
                self._log(f"Failed to kill local server: {e}\n")

        if self._docker_log_proc and self._docker_log_proc.poll() is None:
            try:
                self._docker_log_proc.kill()
            except:
                pass
            self._docker_log_proc = None

        # Force stop Docker container
        if not killed_something or True:  # Always offer docker force kill
            self._log("docker compose down --remove-orphans (force)…\n")
            def run_kill():
                try:
                    subprocess.run(
                        ["docker", "compose", "down", "--remove-orphans", "-v"],
                        cwd=ROOT_DIR,
                        capture_output=True,
                        timeout=10,
                    )
                    self.after(0, self._log, "Docker containers force killed and removed.\n")
                except subprocess.TimeoutExpired:
                    self.after(0, self._log, "Docker force kill timed out. Try again or restart Docker.\n")
                except FileNotFoundError:
                    pass
                except Exception as e:
                    self.after(0, self._log, f"Docker force kill error: {e}\n")
            threading.Thread(target=run_kill, daemon=True).start()

        self._set_srv_state("stopped")

    def _stream_logs(self, proc: subprocess.Popen):
        for line in proc.stdout:
            self.after(0, self._log, line)

    # ══════════════════════════════════════════════════════════════════════════
    # HEALTH CHECK
    # ══════════════════════════════════════════════════════════════════════════

    def _is_server_reachable(self, timeout_sec: float = 2.0) -> bool:
        try:
            with urllib.request.urlopen(f"http://{SERVER_IP}:{SERVER_PORT}/", timeout=timeout_sec) as r:
                if r.status != 200:
                    return False
                body = json.loads(r.read().decode())
                if isinstance(body, dict):
                    return body.get("service") == "whisper-transcribe" or bool(body)
                return bool(body)
        except Exception:
            return False

    def _auto_start_server_if_not_running(self):
        if self._auto_server_start_attempted:
            return

        self._auto_server_start_attempted = True
        time.sleep(0.8)

        if self._is_server_reachable():
            self.after(0, lambda: self._set_srv_state("running"))
            return

        if not _get_venv_python():
            self.after(0, self._log, "Auto-start skipped: no local venv found.\n")
            return

        self.after(0, self._log, "Starting local server automatically on launch…\n")
        self.after(0, self._start_local)

    def _health_worker(self):
        while True:
            try:
                with urllib.request.urlopen(
                    f"http://{SERVER_IP}:{SERVER_PORT}/", timeout=2
                ) as r:
                    running = r.status == 200
                    body = json.loads(r.read().decode()) if running else {}
            except Exception:
                running = False
                body = {}

            self.after(0, self._on_health_result, running, body)
            time.sleep(HEALTH_POLL_SEC)

    def _on_health_result(self, running: bool, body: dict | None = None):
        changed = running != self._server_running
        self._server_running = running
        if changed:
            self._set_srv_state("running" if running else "stopped")
        if running and hasattr(self, "_srv_url_label"):
            self._srv_url_label.configure(text_color="gray70" if running else "gray45")

    # ══════════════════════════════════════════════════════════════════════════
    # FOCUS TRACKING
    # ══════════════════════════════════════════════════════════════════════════

    def _hotkey_listener(self):
        """Listen for global hotkeys: Win+G (record/type)."""
        def on_activate():
            self.after(0, self._hotkey_toggle_record_and_type)

        try:
            hotkeys = GlobalHotKeys({"<cmd>+g": on_activate})
            hotkeys.daemon = True
            hotkeys.start()
            hotkeys.join()
        except Exception as e:
            self._set_tx_status(f"Hotkey setup failed: {e}", "#e74c3c")

    def _hotkey_toggle_record_and_type(self):
        """Win+G: Toggle recording."""
        now = time.time()
        if hasattr(self, '_last_hotkey_time') and now - self._last_hotkey_time < 0.5:
            return  # Debounce to prevent multiple rapid triggers
        self._last_hotkey_time = now

        if self._recording:
            # Stop recording and transcribe
            self._stop_recording()
        else:
            # Start recording
            self._start_recording()

    def _auto_type_text(self, text: str):
        """Automatically type text using keyboard simulation."""
        if not text:
            return
        
        try:
            time.sleep(0.5)  # Wait for focus to return to the target window
            
            kb = KeyboardController()
            # Type each character with a small delay for reliability
            for char in text:
                kb.type(char)
                time.sleep(0.005)  # Slightly longer delay for keyboard input
            
            # Update status after typing completes
            self.after(0, lambda: self._set_tx_status("Done typing.", "gray55"))
        except Exception as e:
            self.after(0, lambda: self._set_tx_status(f"Error typing: {e}", "#e74c3c"))

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL DOWNLOAD
    # ══════════════════════════════════════════════════════════════════════════

    def _download_model(self):
        model = self._model_dl_var.get()
        python = _get_venv_python()
        if not python:
            self._log("No venv found. Cannot download model.\n")
            return
        
        # Check if model already exists
        models_dir = os.path.join(ROOT_DIR, "models")
        model_path = os.path.join(models_dir, model)
        if os.path.isdir(model_path):
            self._log(f"Model '{model}' is already downloaded at {model_path}\n")
            return

        self._btn_download.configure(state="disabled", text="Downloading...")
        
        def run_dl():
            # Ensure dependencies are present before downloading (inside thread)
            if not self._ensure_local_server_deps(python):
                self.after(0, lambda: self._btn_download.configure(state="normal", text="Download Model"))
                return

            self.after(0, lambda: self._log(f"Downloading model '{model}'...\n"))
            try:
                # Use faster-whisper to download
                # Make sure the models dir exists
                models_dir = os.path.join(ROOT_DIR, "models")
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir)

                cmd = [
                    python, "-c",
                    f"from faster_whisper import WhisperModel; print('Loading...'); WhisperModel('{model}', device='cpu', compute_type='int8', download_root='models'); print('Done.')"
                ]
                proc = subprocess.Popen(
                    cmd,
                    cwd=ROOT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                if proc.stdout:
                    for line in proc.stdout:
                        self.after(0, self._log, line)
                proc.wait()
                if proc.returncode == 0:
                    self.after(0, self._log, f"Model '{model}' is ready.\n")
                else:
                    self.after(0, self._log, f"Failed to download model '{model}'. See logs above.\n")
            except Exception as e:
                self.after(0, self._log, f"Error during model download: {e}\n")
            finally:
                self.after(0, lambda: self._btn_download.configure(state="normal", text="Download Model"))

        threading.Thread(target=run_dl, daemon=True).start()

    def _delete_model(self):
        """Delete the selected model from ./models/"""
        model = self._model_dl_var.get()
        models_dir = os.path.join(ROOT_DIR, "models")
        model_path = os.path.join(models_dir, model)
        
        if not os.path.isdir(model_path):
            self._log(f"Model '{model}' not found at {model_path}\n")
            return
        
        self._btn_delete.configure(state="disabled", text="Deleting...")
        
        def run_del():
            try:
                import shutil
                shutil.rmtree(model_path)
                self.after(0, self._log, f"Model '{model}' deleted successfully.\n")
            except Exception as e:
                self.after(0, self._log, f"Failed to delete model '{model}': {e}\n")
            finally:
                self.after(0, lambda: self._btn_delete.configure(state="normal", text="Delete"))
        
        threading.Thread(target=run_del, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    # RECORDING
    # ══════════════════════════════════════════════════════════════════════════

    def _toggle_recording(self):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _set_processing_state(self, active: bool):
        with self._processing_lock:
            if active:
                self._processing_count += 1
            else:
                self._processing_count = max(0, self._processing_count - 1)
            self._processing = self._processing_count > 0

    def _start_recording(self):
        self._recording = True
        self._recording_stream = None
        self._recording_chunks = []
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0

        def callback(indata, _frames, _time_info, status):
            if status:
                log_msg = status.message if hasattr(status, "message") else str(status)
                log = logging.getLogger("whisper-typer")
                log.debug("Audio callback status: %s", log_msg)

            if indata.size == 0:
                return

            with self._audio_lock:
                if not self._recording:
                    return

                block = np.array(indata, copy=True)
                max_amp = float(np.max(np.abs(block)))
                silence_frames_to_flush = int(SILENCE_WINDOW_SECONDS * SAMPLE_RATE)
                min_speech_frames = int(MIN_SPEECH_SECONDS * SAMPLE_RATE)

                if self._is_live_mode():
                    if max_amp >= SPEECH_THRESHOLD:
                        self._recording_chunks.append(block)
                        self._speech_frames += block.shape[0]
                        self._silence_frames = 0
                        self._is_speaking = True
                        return

                    if not self._is_speaking:
                        # Ignore pure silence before the first spoken segment.
                        return

                    # Keep a little trailing audio to avoid clipping the end of words.
                    self._recording_chunks.append(block)
                    self._silence_frames += block.shape[0]
                    if self._speech_frames >= min_speech_frames and self._silence_frames >= silence_frames_to_flush:
                        segment = np.concatenate(self._recording_chunks, axis=0)
                        self._recording_chunks = []
                        self._is_speaking = False
                        self._silence_frames = 0
                        self._speech_frames = 0
                        self._enqueue_transcription(segment, "append", False, auto_type=True)
                else:
                    # Recording mode: collect chunks until you manually stop.
                    self._recording_chunks.append(block)
                    if max_amp >= SPEECH_THRESHOLD:
                        self._speech_frames += block.shape[0]
                        self._is_speaking = True
                        self._silence_frames = 0

        self._recording_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=STREAM_BLOCK_SIZE,
            callback=callback,
        )
        self._recording_stream.start()
        self._btn_record.configure(
            text="⏹  Stop Recording",
            fg_color="#c0392b",
            hover_color="#a93226",
        )
        self._hdr_status.configure(text="🎤 Recording", text_color="#e74c3c")
        if self._is_live_mode():
            self._set_tx_status("Recording… Speak, then pause to auto-send.", "#e74c3c")
        else:
            self._set_tx_status("Recording… Speak, then stop to send.", "#e74c3c")

    def _stop_recording(self):
        if self._recording_stream:
            self._recording_stream.stop()
            self._recording_stream.close()
            self._recording_stream = None

        with self._audio_lock:
            self._recording = False
            recording_chunks = self._recording_chunks
            self._recording_chunks = []
            in_speech = self._is_speaking
            self._is_speaking = False
            self._silence_frames = 0
            self._speech_frames = 0
            sd.stop()
            min_speech_frames = int(MIN_SPEECH_SECONDS * SAMPLE_RATE)
            speech_frames = sum(chunk.shape[0] for chunk in recording_chunks)

        if self._is_live_mode():
            recording = np.concatenate(recording_chunks, axis=0) if speech_frames >= min_speech_frames else None
        else:
            recording = np.concatenate(recording_chunks, axis=0) if recording_chunks else None
        self._recording = False

        self._processing = True
        self._btn_record.configure(state="disabled", text="Transcribing…")
        self._hdr_status.configure(text="⏳ Transcribing", text_color="#8e44ad")
        self._set_tx_status("Transcribing…", "#8e44ad")

        if in_speech and recording is not None:
            self._enqueue_transcription(recording, "replace", True, auto_type=True)
        else:
            self._processing = False
            self._set_processing_state(False)
            self._btn_record.configure(
                state="normal",
                text="⏺  Start Recording",
                fg_color=self._btn_rec_fg,
                hover_color=self._btn_rec_hover,
            )
            self._hdr_status.configure(text="✓ Ready", text_color="#27ae60")
            self._set_tx_status("Ready. Press Win+G to start recording.", "gray55")

    def _transcribe_worker(
        self,
        recording: np.ndarray,
        mode: str = "replace",
        stop_after: bool = False,
        auto_type: bool = True,
    ):
        if recording is None or recording.size == 0:
            if stop_after:
                self.after(0, self._finish_after_transcription, auto_type)
            return

        text = ""
        try:
            audio = trim_trailing_silence(recording)
            model = self._model_var.get()
            text = send_to_server(audio, model)
        except Exception as e:
            text = f"[Error: {e}]"

        if stop_after:
            self.after(0, self._show_result, text, auto_type)
            return

        if mode == "append":
            self.after(0, self._append_result, text, auto_type)
        else:
            self.after(0, self._show_result, text, auto_type)

    def _append_result(self, text: str, auto_type: bool = True):
        text = text.strip()
        if not text:
            return
        if text and not text.startswith("["):
            self._add_history_entry(text, self._model_var.get())
        self._textbox.configure(state="normal")
        current = self._textbox.get("1.0", "end").strip()
        if current and not current.endswith((" ", "\n")):
            self._textbox.insert("end", " ")
        self._textbox.insert("end", text)
        self._textbox.configure(state="disabled")
        if auto_type:
            threading.Thread(target=self._auto_type_text, args=(text,), daemon=True).start()
            if not self._recording:
                self._set_tx_status("Done typing.", "gray55")
        else:
            self._set_tx_status("Transcription ready in output box.", "gray55")

    def _finish_after_transcription(self, auto_type: bool = True):
        self._btn_record.configure(
            state="normal",
            text="⏺  Start Recording",
            fg_color=self._btn_rec_fg,
            hover_color=self._btn_rec_hover,
        )
        self._hdr_status.configure(text="✓ Transcribed", text_color="#27ae60")
        if auto_type:
            self._set_tx_status("Done typing.", "gray55")
        else:
            self._set_tx_status("Ready. Press Win+G to start recording.", "gray55")
        state = getattr(self, "_current_srv_state", "stopped")
        self.after(0, lambda: self._set_srv_state(state))

    def _transcription_queue_worker(self):
        while True:
            recording, mode, stop_after, auto_type = self._transcribe_queue.get()
            try:
                self._transcribe_worker(recording, mode, stop_after, auto_type)
            finally:
                self._set_processing_state(False)
                self._transcribe_queue.task_done()

    def _enqueue_transcription(
        self,
        recording: np.ndarray,
        mode: str = "replace",
        stop_after: bool = False,
        auto_type: bool = True,
    ):
        if recording is None or recording.size == 0:
            if stop_after:
                self.after(0, self._finish_after_transcription, auto_type)
            return

        self._set_processing_state(True)
        self._transcribe_queue.put((recording, mode, stop_after, auto_type))

    def _show_result(self, text: str, auto_type: bool = True):
        text = text.strip()
        self._processing = False
        self._last_transcription = text
        self._textbox.configure(state="normal")
        self._textbox.delete("1.0", "end")
        self._textbox.insert("1.0", text)
        self._textbox.configure(state="disabled")

        if text and not text.startswith("["):
            self._add_history_entry(text, self._model_var.get())

        self._btn_record.configure(
            state="normal",
            text="⏺  Start Recording",
            fg_color=self._btn_rec_fg,
            hover_color=self._btn_rec_hover,
        )
        self._hdr_status.configure(text="✓ Transcribed", text_color="#27ae60")

        def _type_and_restore():
            if auto_type:
                self._auto_type_text(text)
            # Restore header to show server status
            state = getattr(self, "_current_srv_state", "stopped")
            self.after(0, lambda: self._set_srv_state(state))

        if auto_type:
            self._set_tx_status("Typing…", "#f39c12")
        else:
            self._set_tx_status("Transcription ready in output box.", "#f39c12")
        threading.Thread(target=_type_and_restore, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _set_srv_state(self, state: str):
        """state: 'running' | 'stopped'
        Updates: Server tab card, header badge, and tray icon/tooltip.
        """
        self._current_srv_state = state

        cfg = {
            "running": {
                "title": "Server Running",
                "color": "#27ae60",
                "badge": "  ONLINE  ",
                "badge_fg": "#1b7a3d",
                "badge_text": "#d4f5e0",
                "hdr": "● Server Online",
            },
            "stopped": {
                "title": "Server Stopped",
                "color": "#000000",
                "badge": "  OFFLINE  ",
                "badge_fg": "#000000",
                "badge_text": "#f5f5f5",
                "hdr": "● Server Offline",
            },
        }
        c = cfg.get(state, cfg["stopped"])

        # Server tab card
        if hasattr(self, "_srv_status_label"):
            self._srv_status_label.configure(text=c["title"], text_color=c["color"])
        if hasattr(self, "_srv_status_bar"):
            self._srv_status_bar.configure(fg_color=c["color"])
        if hasattr(self, "_srv_status_badge"):
            self._srv_status_badge.configure(
                text=c["badge"], fg_color=c["badge_fg"], text_color=c["badge_text"],
            )
        if hasattr(self, "_srv_model_label"):
            model = self._model_var.get() if hasattr(self, "_model_var") else "small"
            self._srv_model_label.configure(text=f"model: {model}")

        # Header badge (only if not mid-recording/transcribing)
        if hasattr(self, "_hdr_status") and not self._recording:
            self._hdr_status.configure(text=c["hdr"], text_color=c["color"])

    def _log(self, text: str):
        self._log_box.configure(state="normal")
        self._log_box.insert("end", text)
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _clear_logs(self):
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")

    def _set_tx_status(self, message: str, color: str = "gray55"):
        self._tx_status.configure(text=message, text_color=color)

    def _setup_tray(self):
        """Set up system tray icon on Windows."""
        if sys.platform != "win32":
            return
        
        try:
            from pystray import Icon, Menu, MenuItem
            
            self._tray_icon = None
            self._tray_status = "stopped"
            self._tray_recording = False
            
            def tray_toggle_record(icon, item):
                self._toggle_recording()
            
            def tray_show(icon, item):
                self.deiconify()
                self.lift()
                self.focus()
            
            def tray_quit(icon, item):
                self._on_close()
            
            menu = Menu(
                MenuItem("Record (Win+G)", tray_toggle_record),
                MenuItem("Show Window", tray_show),
                MenuItem("Quit", tray_quit),
            )
            
            self._tray_icon = Icon(
                "whisper-typer",
                _make_status_icon("stopped"),
                menu=menu,
            )
            
            threading.Thread(target=self._tray_icon.run, daemon=True).start()
        except ImportError:
            pass  # pystray not installed
        except Exception as e:
            self._set_tx_status(f"Tray error: {e}", "#e74c3c")

    def _update_tray_icon(self):
        """Update tray icon to reflect current state.
        Priority: recording (red) > processing (purple) > server online (green) > offline (black).
        """
        if not hasattr(self, '_tray_icon') or not self._tray_icon:
            self.after(1000, self._update_tray_icon)
            return

        if self._recording:
            status = "recording"
            title = "Whisper Typer — Recording…"
        elif self._processing:
            status = "processing"
            title = "Whisper Typer — Transcribing…"
        elif self._server_running:
            status = "running"
            title = "Whisper Typer — Server Online"
        else:
            status = "stopped"
            title = "Whisper Typer — Server Offline"

        prev = (self._tray_status, self._tray_recording)
        curr = (status, self._recording)
        if curr != prev:
            try:
                self._tray_status = status
                self._tray_recording = self._recording
                self._tray_icon.icon = _make_status_icon(status)
                self._tray_icon.title = title
            except Exception:
                pass

        self.after(500, self._update_tray_icon)

    def _on_close(self):
        if self._local_proc and self._local_proc.poll() is None:
            self._local_proc.terminate()
        if self._docker_log_proc and self._docker_log_proc.poll() is None:
            self._docker_log_proc.terminate()
        if hasattr(self, '_tray_icon') and self._tray_icon:
            try:
                self._tray_icon.stop()
            except:
                pass
        self.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = WhisperUI()
    app.mainloop()


if __name__ == "__main__":
    main()

