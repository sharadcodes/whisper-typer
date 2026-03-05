"""Whisper Typer — UI client.
Two tabs:
  • Transcribe — record audio, send to server, display result.
  • Server     — start/stop server (local or Docker), live log stream.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
import urllib.parse

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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS   = ["tiny", "base", "small", "medium", "large-v3"]
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
        "running": "#27ae60",    # green
        "stopped": "#e74c3c",    # red
        "recording": "#f39c12",  # amber
        "starting": "#3498db",   # blue
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
        self.geometry("480x680")
        self.minsize(420, 540)

        # recording state
        self._recording       = False
        self._recording_data: np.ndarray | None = None
        self._last_transcription = ""  # Store the last result for auto-typing

        # server process handles
        self._local_proc: subprocess.Popen | None = None   # local uvicorn
        self._docker_log_proc: subprocess.Popen | None = None  # docker logs -f

        # last known server health
        self._server_running = False

        self._build_ui()
        self._setup_tray()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._set_srv_state("stopped")
        threading.Thread(target=self._health_worker, daemon=True).start()
        
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
            hdr, text="Ready",
            font=ctk.CTkFont(size=12),
            text_color="gray55",
        )
        self._hdr_status.grid(row=0, column=1, sticky="e")

        # ── Tabs ──────────────────────────────────────────────────────────────
        self._tabs = ctk.CTkTabview(self, corner_radius=10)
        self._tabs.grid(row=1, column=0, padx=16, pady=(10, 16), sticky="nsew")

        self._tab_t = self._tabs.add("  Transcribe  ")
        self._tab_s = self._tabs.add("  Server  ")

        self._build_transcribe_tab()
        self._build_server_tab()

    # ── Transcribe tab ────────────────────────────────────────────────────────

    def _build_transcribe_tab(self):
        tab = self._tab_t
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(3, weight=1)

        # Model row
        model_row = ctk.CTkFrame(tab, fg_color="transparent")
        model_row.grid(row=0, column=0, pady=(8, 14), sticky="ew")
        ctk.CTkLabel(model_row, text="Model", font=ctk.CTkFont(size=13)).pack(side="left", padx=(0, 10))
        self._model_var = ctk.StringVar(value="small")
        ctk.CTkOptionMenu(model_row, values=MODELS, variable=self._model_var, width=160).pack(side="left")

        # Record button
        self._btn_record = ctk.CTkButton(
            tab,
            text="⏺  Start Recording",
            font=ctk.CTkFont(size=15, weight="bold"),
            height=52,
            corner_radius=12,
            command=self._toggle_recording,
        )
        self._btn_record.grid(row=1, column=0, pady=(0, 16), sticky="ew")
        self._btn_rec_fg    = self._btn_record.cget("fg_color")
        self._btn_rec_hover = self._btn_record.cget("hover_color")

        # Transcription label
        ctk.CTkLabel(
            tab, text="Transcription",
            font=ctk.CTkFont(size=12),
            text_color="gray60",
        ).grid(row=2, column=0, sticky="w")

        # Transcription box
        self._textbox = ctk.CTkTextbox(
            tab,
            font=ctk.CTkFont(size=14),
            corner_radius=8,
            wrap="word",
        )
        self._textbox.grid(row=3, column=0, pady=(4, 8), sticky="nsew")
        self._textbox.configure(state="disabled")

        # Status label
        self._tx_status = ctk.CTkLabel(
            tab, text="Ready. Press Win+G to start recording.",
            font=ctk.CTkFont(size=11),
            text_color="gray55",
        )
        self._tx_status.grid(row=4, column=0, pady=(8, 0), sticky="w")

    # ── Server tab ────────────────────────────────────────────────────────────

    def _build_server_tab(self):
        tab = self._tab_s
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(3, weight=1) # Adjusted row count

        # Status + control card
        card = ctk.CTkFrame(tab, corner_radius=10)
        card.grid(row=0, column=0, pady=(8, 8), sticky="ew")
        card.grid_columnconfigure(0, weight=1)

        self._srv_status_label = ctk.CTkLabel(
            card, text="● Checking…",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="gray50",
        )
        self._srv_status_label.grid(row=0, column=0, padx=16, pady=(14, 6), sticky="w")

        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.grid(row=1, column=0, padx=12, pady=(0, 14), sticky="ew")

        self._btn_start_local = ctk.CTkButton(
            btn_row, text="Start Local", width=120,
            command=self._start_local,
        )
        self._btn_start_local.pack(side="left", padx=(0, 8))

        self._btn_start_docker = ctk.CTkButton(
            btn_row, text="Start Docker", width=120,
            fg_color="#1a6b3a", hover_color="#145530",
            command=self._start_docker,
        )
        # Only show Docker button if docker command is available
        if shutil.which("docker"):
            self._btn_start_docker.pack(side="left", padx=(0, 8))

        self._btn_stop_srv = ctk.CTkButton(
            btn_row, text="Stop", width=80,
            fg_color="gray35", hover_color="gray28",
            command=self._stop_server,
        )
        self._btn_stop_srv.pack(side="left", padx=(0, 8))

        self._btn_kill_srv = ctk.CTkButton(
            btn_row, text="Force Kill", width=100,
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

        self._set_srv_state("starting")
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
        self._set_srv_state("starting")

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

    def _health_worker(self):
        while True:
            try:
                with urllib.request.urlopen(
                    f"http://{SERVER_IP}:{SERVER_PORT}/", timeout=2
                ) as r:
                    running = r.status == 200
            except Exception:
                running = False

            self.after(0, self._on_health_result, running)
            time.sleep(HEALTH_POLL_SEC)

    def _on_health_result(self, running: bool):
        if running != self._server_running:
            self._server_running = running
            self._set_srv_state("running" if running else "stopped")

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

    def _start_recording(self):
        self._recording = True
        self._recording_data = sd.rec(
            int(MAX_RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        self._btn_record.configure(
            text="⏹  Stop Recording",
            fg_color="#c0392b",
            hover_color="#a93226",
        )
        self._hdr_status.configure(text="🎤 Recording", text_color="#e74c3c")
        self._set_tx_status("Recording… (Press Win+G to stop)", "#e74c3c")

    def _stop_recording(self):
        sd.stop()
        recording = self._recording_data
        self._recording = False
        self._recording_data = None

        self._btn_record.configure(state="disabled", text="Transcribing…")
        self._hdr_status.configure(text="⏳ Transcribing", text_color="#f39c12")
        self._set_tx_status("Transcribing…", "#f39c12")

        threading.Thread(
            target=self._transcribe_worker, args=(recording,), daemon=True
        ).start()

    def _transcribe_worker(self, recording: np.ndarray):
        audio = trim_trailing_silence(recording)
        model = self._model_var.get()
        text = send_to_server(audio, model)
        self.after(0, self._show_result, text)

    def _show_result(self, text: str):
        text = text.strip()
        self._last_transcription = text
        self._textbox.configure(state="normal")
        self._textbox.delete("1.0", "end")
        self._textbox.insert("1.0", text)
        self._textbox.configure(state="disabled")

        self._btn_record.configure(
            state="normal",
            text="⏺  Start Recording",
            fg_color=self._btn_rec_fg,
            hover_color=self._btn_rec_hover,
        )
        self._hdr_status.configure(text="✓ Done", text_color="#27ae60")
        self._set_tx_status("Typing…", "#f39c12")
        
        # Auto-type the result in a background thread
        threading.Thread(target=self._auto_type_text, args=(text,), daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _set_srv_state(self, state: str):
        """state: 'running' | 'stopped' | 'starting' (server status for Server tab only)"""
        cfg = {
            "running":  ("● Running",   "#27ae60"),
            "stopped":  ("● Stopped",   "#e74c3c"),
            "starting": ("◌ Starting…", "#f39c12"),
        }
        text, color = cfg.get(state, ("● Unknown", "gray50"))
        # Only update Server tab status, not header
        if hasattr(self, '_srv_status_label'):
            self._srv_status_label.configure(text=text, text_color=color)

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
        """Update tray icon to reflect current transcription status."""
        if not hasattr(self, '_tray_icon') or not self._tray_icon:
            self.after(1000, self._update_tray_icon)
            return
        
        # Determine status based on transcription state only
        if self._recording:
            status = "recording"
            title = "🎤 Recording"
        elif self._last_transcription:
            status = "running"
            title = f"✓ {self._last_transcription[:20]}..."
        else:
            status = "stopped"
            title = "Ready to record"
        
        # Update if changed
        if status != self._tray_status or self._recording != self._tray_recording:
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
