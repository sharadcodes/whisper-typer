import os
import json
import shutil
import subprocess
import threading
from datetime import datetime
from typing import Callable, Any

from .config import (
    ROOT_DIR, HISTORY_FILE, SERVER_IP, SERVER_PORT,
    get_venv_python
)
from .api import is_server_reachable

class WhisperManager:
    """Manages background operations: Server processes, Model downloads, History."""

    def __init__(self, log_callback: Callable[[str], None] = None, status_callback: Callable[[str, Any], None] = None):
        self.log_callback = log_callback or (lambda x: None)
        self.status_callback = status_callback or (lambda x, y: None)
        
        self.local_proc: subprocess.Popen | None = None
        self.docker_log_proc: subprocess.Popen | None = None
        self.server_running = False
        
        self.history = self._load_history()

    # ── History ───────────────────────────────────────────────────────────────

    def _load_history(self) -> list[dict]:
        if os.path.isfile(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, list) else []
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def save_history(self):
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def add_history_entry(self, text: str, model: str) -> dict:
        ts = datetime.now().strftime("%I:%M %p").lstrip("0")
        date_str = datetime.now().strftime("%b %d")
        entry = {"time": f"{date_str}, {ts}", "model": model, "text": text}
        self.history.append(entry)
        self.save_history()
        return entry

    def clear_history(self):
        self.history.clear()
        try:
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
        except OSError:
            pass

    # ── Server Processes ──────────────────────────────────────────────────────

    def check_server_health(self) -> bool:
        self.server_running = is_server_reachable()
        return self.server_running

    def start_local_server(self):
        if self.local_proc and self.local_proc.poll() is None:
            self.log_callback("Server is already running locally.\n")
            return

        python = get_venv_python()
        if not python:
            self.log_callback("No venv found. Please run 'uv run' to set up.\n")
            return

        def run_srv():
            env_vars = os.environ.copy()
            env_path = os.path.join(ROOT_DIR, ".env")
            if os.path.isfile(env_path):
                with open(env_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            env_vars[k.strip()] = v.strip()

            server_dir = os.path.join(ROOT_DIR, "root", "app")
            self.log_callback(f"Starting local server with {python}…\n")

            try:
                self.local_proc = subprocess.Popen(
                    [
                        python, "-m", "uvicorn", "transcribe_api:app",
                        "--host", SERVER_IP, "--port", str(SERVER_PORT),
                    ],
                    cwd=server_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env_vars,
                )
                self._stream_logs(self.local_proc)
            except Exception as e:
                self.log_callback(f"Failed to start local server: {e}\n")

        threading.Thread(target=run_srv, daemon=True).start()

    def stop_server(self):
        if self.local_proc and self.local_proc.poll() is None:
            self.local_proc.terminate()
            self.local_proc = None
            self.log_callback("Local server stopped.\n")

        if self.docker_log_proc and self.docker_log_proc.poll() is None:
            self.docker_log_proc.terminate()
            self.docker_log_proc = None

    def kill_server(self):
        if self.local_proc and self.local_proc.poll() is None:
            try:
                import signal
                if hasattr(signal, 'SIGKILL'):
                    os.kill(self.local_proc.pid, signal.SIGKILL)
                else:
                    subprocess.run(["taskkill", "/PID", str(self.local_proc.pid), "/F"], capture_output=True)
                self.local_proc = None
                self.log_callback("Local server force killed.\n")
            except Exception as e:
                self.log_callback(f"Kill error: {e}\n")

    def _stream_logs(self, proc: subprocess.Popen):
        if proc.stdout:
            for line in proc.stdout:
                self.log_callback(line)

    # ── Model Management ──────────────────────────────────────────────────────

    def download_model(self, model_name: str, on_complete: Callable[[bool], None]):
        python = get_venv_python()
        if not python:
            self.log_callback("No venv found.\n")
            return

        def run():
            self.log_callback(f"Downloading model '{model_name}'...\n")
            try:
                cmd = [
                    python, "-c",
                    f"from faster_whisper import WhisperModel; WhisperModel('{model_name}', device='cpu', compute_type='int8', download_root='models')"
                ]
                proc = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True)
                if proc.returncode == 0:
                    self.log_callback(f"Model '{model_name}' ready.\n")
                    on_complete(True)
                else:
                    self.log_callback(f"Download failed: {proc.stderr}\n")
                    on_complete(False)
            except Exception as e:
                self.log_callback(f"Error: {e}\n")
                on_complete(False)

        threading.Thread(target=run, daemon=True).start()

    def delete_model(self, model_name: str):
        model_path = os.path.join(ROOT_DIR, "models", model_name)
        if os.path.isdir(model_path):
            try:
                shutil.rmtree(model_path)
                self.log_callback(f"Deleted model '{model_name}'.\n")
                return True
            except Exception as e:
                self.log_callback(f"Delete failed: {e}\n")
        return False
