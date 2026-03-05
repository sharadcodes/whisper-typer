"""Whisper Typer — UI client.
Records audio, sends to the transcribe server, and displays the result.
No hotkeys, no auto-typing into other windows.
"""
import json
import threading
import urllib.request
import urllib.error
import urllib.parse

import numpy as np
import sounddevice as sd
import customtkinter as ctk

# ── Config ────────────────────────────────────────────────────────────────────
SERVER_IP          = "127.0.0.1"
SERVER_PORT        = 8000
SAMPLE_RATE        = 16000
MAX_RECORD_SECONDS = 300

MODELS = ["tiny", "base", "small", "medium", "large-v3"]

# ── Audio helpers ─────────────────────────────────────────────────────────────

def trim_trailing_silence(audio: np.ndarray, threshold: float = 0.005) -> np.ndarray:
    flat = audio.flatten()
    mask = np.abs(flat) > threshold
    if not np.any(mask):
        return flat[: int(0.1 * SAMPLE_RATE)].reshape(-1, 1).astype(np.float32)
    last = int(np.where(mask)[0][-1])
    return flat[: last + 1].reshape(-1, 1).astype(np.float32)


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
        return f"[Server unreachable — is it running? {e}]"
    except Exception as e:
        return f"[Error: {e}]"


# ── UI ────────────────────────────────────────────────────────────────────────

class WhisperUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Whisper Typer")
        self.geometry("440x540")
        self.resizable(False, False)

        self._recording = False
        self._recording_data: np.ndarray | None = None

        self._build_ui()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)

        # Title
        ctk.CTkLabel(
            self,
            text="Whisper Typer",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, pady=(22, 6), padx=24, sticky="w")

        # Model row
        model_frame = ctk.CTkFrame(self, fg_color="transparent")
        model_frame.grid(row=1, column=0, padx=24, pady=(0, 14), sticky="ew")
        ctk.CTkLabel(model_frame, text="Model", font=ctk.CTkFont(size=13)).pack(side="left", padx=(0, 10))
        self._model_var = ctk.StringVar(value="tiny")
        ctk.CTkOptionMenu(
            model_frame,
            values=MODELS,
            variable=self._model_var,
            width=160,
        ).pack(side="left")

        # Record button
        self._btn_record = ctk.CTkButton(
            self,
            text="⏺  Start Recording",
            font=ctk.CTkFont(size=15, weight="bold"),
            height=52,
            corner_radius=12,
            command=self._toggle_recording,
        )
        self._btn_record.grid(row=2, column=0, padx=24, pady=(0, 18), sticky="ew")
        # Store default colours so we can restore them later
        self._btn_default_fg    = self._btn_record.cget("fg_color")
        self._btn_default_hover = self._btn_record.cget("hover_color")

        # Transcription section
        ctk.CTkLabel(
            self,
            text="Transcription",
            font=ctk.CTkFont(size=13),
            text_color="gray60",
        ).grid(row=3, column=0, padx=26, sticky="w")

        self._textbox = ctk.CTkTextbox(
            self,
            height=230,
            font=ctk.CTkFont(size=14),
            corner_radius=10,
            wrap="word",
        )
        self._textbox.grid(row=4, column=0, padx=24, pady=(4, 8), sticky="ew")
        self._textbox.configure(state="disabled")

        # Action buttons row
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.grid(row=5, column=0, padx=24, sticky="ew")

        self._status_label = ctk.CTkLabel(
            btn_row,
            text="Ready",
            font=ctk.CTkFont(size=12),
            text_color="gray55",
        )
        self._status_label.pack(side="left")

        ctk.CTkButton(
            btn_row,
            text="Clear",
            width=76,
            fg_color="gray30",
            hover_color="gray25",
            command=self._clear_text,
        ).pack(side="right", padx=(8, 0))

        ctk.CTkButton(
            btn_row,
            text="Copy",
            width=76,
            command=self._copy_text,
        ).pack(side="right")

    # ── Actions ───────────────────────────────────────────────────────────────

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
        self._set_status("Recording…", "#e74c3c")

    def _stop_recording(self):
        sd.stop()
        recording = self._recording_data
        self._recording = False
        self._recording_data = None

        self._btn_record.configure(state="disabled", text="Transcribing…")
        self._set_status("Transcribing…", "#f39c12")

        threading.Thread(
            target=self._transcribe_worker,
            args=(recording,),
            daemon=True,
        ).start()

    def _transcribe_worker(self, recording: np.ndarray):
        audio = trim_trailing_silence(recording)
        model = self._model_var.get()
        text = send_to_server(audio, model)
        self.after(0, self._show_result, text)

    def _show_result(self, text: str):
        self._textbox.configure(state="normal")
        self._textbox.delete("1.0", "end")
        self._textbox.insert("1.0", text.strip())
        self._textbox.configure(state="disabled")

        self._btn_record.configure(
            state="normal",
            text="⏺  Start Recording",
            fg_color=self._btn_default_fg,
            hover_color=self._btn_default_hover,
        )
        self._set_status("Done.", "gray55")

    def _copy_text(self):
        text = self._textbox.get("1.0", "end").strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self._set_status("Copied to clipboard.", "gray55")

    def _clear_text(self):
        self._textbox.configure(state="normal")
        self._textbox.delete("1.0", "end")
        self._textbox.configure(state="disabled")
        self._set_status("Cleared.", "gray55")

    def _set_status(self, message: str, color: str = "gray55"):
        self._status_label.configure(text=message, text_color=color)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = WhisperUI()
    app.mainloop()


if __name__ == "__main__":
    main()
