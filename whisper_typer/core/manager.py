import os
import sys
import json
import shutil
import subprocess
import threading
from datetime import datetime
from typing import Callable, Any

from .config import (
    HISTORY_FILE, SERVER_IP, SERVER_PORT,
    SERVER_DIR, MODELS_DIR, get_venv_python
)
from .api import is_server_reachable
from .utils import is_port_in_use

class WhisperManager:
    """Manages background operations with industry-standard process lifecycle handling."""

    def __init__(self, log_callback: Callable[[str], None] = None, status_callback: Callable[[str, Any], None] = None):
        self.log_callback = log_callback or (lambda x: None)
        self.status_callback = status_callback or (lambda x, y: None)
        
        self.local_proc: subprocess.Popen | None = None
        self.docker_log_proc: subprocess.Popen | None = None
        self.server_running = False
        self.server_starting = False
        
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
        ts = datetime.now().strftime("%I:%M %p")
        if ts.startswith("0"):
            ts = ts[1:]
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
        """Industry practice: Check if process is still alive + HTTP health check."""
        if self.local_proc:
            exit_code = self.local_proc.poll()
            if exit_code is not None:
                self.local_proc = None
                self.server_running = False
                self.server_starting = False
                self.log_callback(f"Local server process exited unexpectedly with code {exit_code}.\n")
                return False

        self.server_running = is_server_reachable()
        if self.server_running:
            self.server_starting = False
        return self.server_running

    def start_local_server(self):
        if (self.local_proc and self.local_proc.poll() is None) or self.server_starting:
            return

        if is_port_in_use(SERVER_PORT, SERVER_IP):
            if is_server_reachable(timeout_sec=0.5):
                self.log_callback(f"Server already running on port {SERVER_PORT} (external process).\n")
                self.server_running = True
                return
            else:
                self.log_callback(f"Error: Port {SERVER_PORT} is occupied by another application.\n")
                return

        python = get_venv_python()
        if not python:
            self.log_callback("Python interpreter not found.\n")
            return

        self.server_starting = True

        def run_srv():
            server_dir = str(SERVER_DIR)
            
            # Load environment variables from .env if it exists in project root
            # Project root is one level up from PACKAGE_DIR (whisper_typer/)
            project_root = SERVER_DIR.parent.parent
            env_vars = os.environ.copy()
            env_path = project_root / ".env"
            
            if env_path.exists():
                try:
                    with open(env_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                k, v = line.split("=", 1)
                                if k.startswith("export "):
                                    k = k[7:]
                                # Handle inline comments
                                if " #" in v:
                                    v = v.split(" #")[0]
                                # Strip whitespace and surrounding quotes (single or double)
                                v = v.strip().strip('"').strip("'")
                                env_vars[k.strip()] = v
                except Exception as e:
                    self.log_callback(f"Warning: Failed to read .env file: {e}\n")

            self.log_callback(f"Starting local server with {python}…\n")

            try:
                kwargs = {
                    "cwd": server_dir,
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.STDOUT,
                    "text": True,
                    "bufsize": 1,
                    "env": env_vars,
                }
                if sys.platform == "win32":
                    kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

                # Point to the app inside the package
                self.local_proc = subprocess.Popen(
                    [
                        python, "-m", "uvicorn", "transcribe_api:app",
                        "--host", SERVER_IP, "--port", str(SERVER_PORT),
                    ],
                    **kwargs
                )
                self._stream_logs(self.local_proc)
            except Exception as e:
                self.server_starting = False
                self.log_callback(f"Failed to start local server: {e}\n")

        threading.Thread(target=run_srv, daemon=True).start()

    def stop_server(self):
        self.server_starting = False
        proc = self.local_proc
        if proc and proc.poll() is None:
            self.log_callback("Terminating local server...\n")

            def cleanup(p):
                try:
                    if sys.platform == "win32":
                        subprocess.run(
                            ["taskkill", "/PID", str(p.pid), "/F", "/T"],
                            capture_output=True, check=False
                        )
                    else:
                        p.terminate()
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        p.kill()
                finally:
                    if self.local_proc is p:
                        self.local_proc = None

            threading.Thread(target=cleanup, args=(proc,), daemon=True).start()

    def close(self):
        """Synchronous shutdown used during app exit (2-second grace period then kill)."""
        self.server_starting = False
        proc = self.local_proc
        if proc and proc.poll() is None:
            self.log_callback("Shutting down server...\n")
            if sys.platform == "win32":
                # taskkill /T kills the whole process tree, including uvicorn worker
                # child processes that proc.terminate() would leave as orphans.
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/F", "/T"],
                    capture_output=True, check=False
                )
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
            self.local_proc = None

    def kill_server(self):
        self.server_starting = False
        if self.local_proc and self.local_proc.poll() is None:
            try:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/PID", str(self.local_proc.pid), "/F", "/T"], 
                                 capture_output=True, check=False)
                else:
                    import signal
                    os.kill(self.local_proc.pid, signal.SIGKILL)
                
                self.local_proc = None
                self.log_callback("Local server process tree killed.\n")
            except Exception as e:
                self.log_callback(f"Kill error: {e}\n")

    def _stream_logs(self, proc: subprocess.Popen):
        if not proc.stdout:
            return
        
        def read_logs():
            for line in proc.stdout:
                if not self.server_starting and not self.server_running:
                    break
                self.log_callback(line)
        threading.Thread(target=read_logs, daemon=True).start()

    # ── Model Management ──────────────────────────────────────────────────────

    def download_model(self, model_name: str, on_complete: Callable[[bool], None]):
        python = get_venv_python()
        if not python:
            self.log_callback("Python interpreter not found.\n")
            return

        def run():
            self.log_callback(f"Downloading model '{model_name}' to {MODELS_DIR}...\n")
            try:
                # Use raw string for download_root to handle Windows backslashes
                script = f"""
import os
from faster_whisper import WhisperModel
model_name = os.environ['MODEL_NAME']
models_dir = os.environ['MODELS_DIR']
WhisperModel(model_name, device='cpu', compute_type='int8', download_root=models_dir)
"""
                env = os.environ.copy()
                env["MODEL_NAME"] = model_name
                env["MODELS_DIR"] = str(MODELS_DIR)

                cmd = [python, "-c", script]
                proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
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
        # faster_whisper downloads from Systran on HuggingFace; the local cache
        # directory is named with the HF convention, NOT the bare model name.
        hf_cache_dir = f"models--Systran--faster-whisper-{model_name}"
        candidates = [
            os.path.join(MODELS_DIR, hf_cache_dir),
            os.path.join(MODELS_DIR, model_name),  # fallback for manual placements
        ]
        for model_path in candidates:
            if os.path.isdir(model_path):
                try:
                    shutil.rmtree(model_path)
                    self.log_callback(f"Deleted model '{model_name}'.\n")
                    return True
                except Exception as e:
                    self.log_callback(f"Delete failed: {e}\n")
                    return False
        self.log_callback(f"Model '{model_name}' not found in {MODELS_DIR}.\n")
        return False
