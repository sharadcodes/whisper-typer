"""Whisper Typer — GUI client."""
import threading
import time
from queue import Queue

import numpy as np
import sounddevice as sd
from pynput.keyboard import Controller as KeyboardController, GlobalHotKeys
import customtkinter as ctk

from src.config import (
    SERVER_IP, SERVER_PORT, SAMPLE_RATE, 
    SILENCE_WINDOW_SECONDS, SPEECH_THRESHOLD, MIN_SPEECH_SECONDS,
    STREAM_BLOCK_SIZE, MODELS, TRANSCRIBE_MODE_LIVE, 
    TRANSCRIBE_MODES, HEALTH_POLL_SEC
)
from src.utils import trim_trailing_silence, make_status_icon
from src.api import send_to_server
from src.manager import WhisperManager

class WhisperUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Whisper Typer")
        self.geometry("960x680")
        self.minsize(420, 540)

        # ── State Management ──────────────────────────────────────────────────
        self.manager = WhisperManager(
            log_callback=self._log,
            status_callback=None # Placeholder if needed later
        )

        self._recording = False
        self._processing = False
        self._processing_count = 0
        self._processing_lock = threading.Lock()
        
        self._audio_lock = threading.Lock()
        self._transcribe_queue: Queue = Queue()
        self._recording_stream: sd.InputStream | None = None
        self._recording_chunks: list[np.ndarray] = []
        
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0
        self._last_transcription = ""
        self._transcribe_mode_var = ctk.StringVar(value=TRANSCRIBE_MODE_LIVE)

        self._build_ui()
        self._setup_tray()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        self._set_srv_state("stopped")
        
        # ── Workers ───────────────────────────────────────────────────────────
        threading.Thread(target=self._health_worker, daemon=True).start()
        threading.Thread(target=self._transcription_queue_worker, daemon=True).start()
        threading.Thread(target=self._auto_start_srv_worker, daemon=True).start()
        threading.Thread(target=self._hotkey_listener, daemon=True).start()
        self._update_tray_icon()

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD UI
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.grid(row=0, column=0, padx=20, pady=(18, 0), sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(hdr, text="Whisper Typer", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, sticky="w")
        self._hdr_status = ctk.CTkLabel(hdr, text="● Server Offline", font=ctk.CTkFont(size=12, weight="bold"), text_color="#000000")
        self._hdr_status.grid(row=0, column=1, sticky="e")

        # Tabs
        self._tabs = ctk.CTkTabview(self, corner_radius=10)
        self._tabs.grid(row=1, column=0, padx=16, pady=(10, 16), sticky="nsew")
        self._tab_t = self._tabs.add("  Transcribe  ")
        self._tab_h = self._tabs.add("  History  ")
        self._tab_s = self._tabs.add("  Server  ")

        self._build_transcribe_tab()
        self._build_history_tab()
        self._build_server_tab()

    def _build_transcribe_tab(self):
        tab = self._tab_t
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(4, weight=1)

        # Model row
        row0 = ctk.CTkFrame(tab, fg_color="transparent")
        row0.grid(row=0, column=0, pady=(8, 14), sticky="ew")
        ctk.CTkLabel(row0, text="Model", font=ctk.CTkFont(size=13)).pack(side="left", padx=(0, 10))
        self._model_var = ctk.StringVar(value="small")
        ctk.CTkOptionMenu(row0, values=MODELS, variable=self._model_var, width=160).pack(side="left")

        # Controls
        row1 = ctk.CTkFrame(tab, fg_color="transparent")
        row1.grid(row=1, column=0, pady=(0, 16), sticky="ew")
        row1.grid_columnconfigure((0, 1), weight=1)

        self._btn_record = ctk.CTkButton(row1, text="⏺  Start Recording", font=ctk.CTkFont(size=15, weight="bold"), height=52, corner_radius=12, command=self._toggle_recording)
        self._btn_record.grid(row=0, column=0, sticky="ew")
        self._btn_rec_fg = self._btn_record.cget("fg_color")
        self._btn_rec_hover = self._btn_record.cget("hover_color")

        ctk.CTkButton(row1, text="🧹  Clear Text", font=ctk.CTkFont(size=15, weight="bold"), height=52, corner_radius=12, fg_color="#636e72", hover_color="#4f585d", command=self._clear_transcript_text).grid(row=0, column=1, padx=(10, 0), sticky="ew")

        # Mode
        row2 = ctk.CTkFrame(tab, fg_color="transparent")
        row2.grid(row=2, column=0, pady=(0, 12), sticky="ew")
        ctk.CTkLabel(row2, text="Input mode", font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w")
        self._mode_selector = ctk.CTkSegmentedButton(row2, values=TRANSCRIBE_MODES, command=self._on_mode_change, width=300, corner_radius=10)
        self._mode_selector.set(TRANSCRIBE_MODE_LIVE)
        self._mode_selector.grid(row=0, column=1, padx=(12, 0), sticky="w")

        # Output
        ctk.CTkLabel(tab, text="Transcription", font=ctk.CTkFont(size=12), text_color="gray60").grid(row=3, column=0, sticky="w")
        self._textbox = ctk.CTkTextbox(tab, font=ctk.CTkFont(size=14), corner_radius=8, wrap="word")
        self._textbox.grid(row=4, column=0, pady=(4, 8), sticky="nsew")
        self._textbox.configure(state="disabled")

        self._tx_status = ctk.CTkLabel(tab, text="Ready. Press Win+G to start recording.", font=ctk.CTkFont(size=11), text_color="gray55")
        self._tx_status.grid(row=5, column=0, pady=(8, 0), sticky="w")

    def _build_history_tab(self):
        tab = self._tab_h
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        hdr = ctk.CTkFrame(tab, fg_color="transparent")
        hdr.grid(row=0, column=0, pady=(8, 6), sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)

        self._hist_count_label = ctk.CTkLabel(hdr, text="0 transcriptions", font=ctk.CTkFont(size=12), text_color="gray55")
        self._hist_count_label.grid(row=0, column=0, sticky="w")

        ctk.CTkButton(hdr, text="Clear All", width=80, height=26, font=ctk.CTkFont(size=11), fg_color="#c0392b", hover_color="#a93226", command=self._clear_history).grid(row=0, column=1, sticky="e")

        self._hist_scroll = ctk.CTkScrollableFrame(tab, corner_radius=8, fg_color="transparent")
        self._hist_scroll.grid(row=1, column=0, sticky="nsew")
        self._hist_scroll.grid_columnconfigure(0, weight=1)

        self._hist_empty = ctk.CTkLabel(self._hist_scroll, text="No transcriptions yet.", font=ctk.CTkFont(size=13), text_color="gray45")
        
        # Load existing history
        for entry in self.manager.history:
            self._render_history_card(entry)
        self._update_history_count()

    def _build_server_tab(self):
        tab = self._tab_s
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(3, weight=1)

        # Status Card
        card = ctk.CTkFrame(tab, corner_radius=10)
        card.grid(row=0, column=0, pady=(8, 8), sticky="ew")
        card.grid_columnconfigure(1, weight=1)

        self._srv_status_bar = ctk.CTkFrame(card, width=6, corner_radius=3, fg_color="gray50")
        self._srv_status_bar.grid(row=0, column=0, rowspan=3, padx=(8, 0), pady=10, sticky="ns")

        stop_top = ctk.CTkFrame(card, fg_color="transparent")
        stop_top.grid(row=0, column=1, padx=(12, 16), pady=(12, 0), sticky="ew")
        stop_top.grid_columnconfigure(0, weight=1)

        self._srv_status_label = ctk.CTkLabel(stop_top, text="Checking…", font=ctk.CTkFont(size=15, weight="bold"), text_color="gray50")
        self._srv_status_label.grid(row=0, column=0, sticky="w")

        self._srv_status_badge = ctk.CTkLabel(stop_top, text="  OFFLINE  ", font=ctk.CTkFont(size=10, weight="bold"), corner_radius=4, fg_color="gray35", text_color="gray70")
        self._srv_status_badge.grid(row=0, column=1, sticky="e")

        info_row = ctk.CTkFrame(card, fg_color="transparent")
        info_row.grid(row=1, column=1, padx=(12, 16), pady=(4, 0), sticky="ew")
        ctk.CTkLabel(info_row, text=f"http://{SERVER_IP}:{SERVER_PORT}", font=ctk.CTkFont(family="Courier", size=11), text_color="gray55").pack(side="left")
        self._srv_model_label = ctk.CTkLabel(info_row, text="model: small", font=ctk.CTkFont(size=11), text_color="gray55")
        self._srv_model_label.pack(side="right")

        btns = ctk.CTkFrame(card, fg_color="transparent")
        btns.grid(row=2, column=1, padx=(8, 12), pady=(8, 12), sticky="ew")
        self._btn_start_local = ctk.CTkButton(btns, text="▶  Start Local", width=130, height=32, command=self.manager.start_local_server)
        self._btn_start_local.pack(side="left", padx=(0, 6))
        ctk.CTkButton(btns, text="Stop", width=70, height=32, fg_color="gray35", command=self.manager.stop_server).pack(side="left", padx=(0, 6))
        ctk.CTkButton(btns, text="Force Kill", width=90, height=32, fg_color="#c0392b", command=self.manager.kill_server).pack(side="left")

        # Manager Card
        m_card = ctk.CTkFrame(tab, corner_radius=10)
        m_card.grid(row=1, column=0, pady=(0, 12), sticky="ew")
        m_card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(m_card, text="Model Manager", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, columnspan=3, padx=16, pady=(10, 6), sticky="w")
        self._model_dl_var = ctk.StringVar(value="small")
        self._model_dl_menu = ctk.CTkOptionMenu(m_card, values=MODELS, variable=self._model_dl_var, width=120)
        self._model_dl_menu.grid(row=1, column=0, padx=(16, 8), pady=(0, 14), sticky="w")
        self._btn_download = ctk.CTkButton(m_card, text="Download", command=self._download_model_ui)
        self._btn_download.grid(row=1, column=1, padx=(8, 8), pady=(0, 14), sticky="ew")
        self._btn_delete = ctk.CTkButton(m_card, text="Delete", fg_color="#c0392b", command=self._delete_model_ui)
        self._btn_delete.grid(row=1, column=2, padx=(0, 16), pady=(0, 14), sticky="ew")

        # Logs
        l_hdr = ctk.CTkFrame(tab, fg_color="transparent")
        l_hdr.grid(row=2, column=0, sticky="ew")
        ctk.CTkLabel(l_hdr, text="Logs", font=ctk.CTkFont(size=12), text_color="gray60").pack(side="left")
        ctk.CTkButton(l_hdr, text="Clear", width=64, height=24, command=self._clear_logs).pack(side="right")
        self._log_box = ctk.CTkTextbox(tab, font=ctk.CTkFont(family="Courier", size=11), corner_radius=8, wrap="none")
        self._log_box.grid(row=3, column=0, pady=(4, 0), sticky="nsew")
        self._log_box.configure(state="disabled")

    # ── UI LOGIC ──────────────────────────────────────────────────────────────

    def _toggle_recording(self):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        self._recording = True
        self._recording_chunks = []
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0

        def cb(indata, _frames, _time, status):
            if not self._recording or indata.size == 0:
                return
            with self._audio_lock:
                block = np.array(indata, copy=True)
                amp = float(np.max(np.abs(block)))
                silence_limit = int(SILENCE_WINDOW_SECONDS * SAMPLE_RATE)
                min_speech = int(MIN_SPEECH_SECONDS * SAMPLE_RATE)

                if self._transcribe_mode_var.get() == TRANSCRIBE_MODE_LIVE:
                    if amp >= SPEECH_THRESHOLD:
                        self._recording_chunks.append(block)
                        self._speech_frames += block.shape[0]
                        self._silence_frames = 0
                        self._is_speaking = True
                    elif self._is_speaking:
                        self._recording_chunks.append(block)
                        self._silence_frames += block.shape[0]
                        if self._speech_frames >= min_speech and self._silence_frames >= silence_limit:
                            segment = np.concatenate(self._recording_chunks, axis=0)
                            self._recording_chunks = []
                            self._is_speaking = False
                            self._silence_frames = 0
                            self._speech_frames = 0
                            self._enqueue_transcription(segment, "append", False)
                else:
                    self._recording_chunks.append(block)
                    if amp >= SPEECH_THRESHOLD:
                        self._speech_frames += block.shape[0]
                        self._is_speaking = True

        self._recording_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, 
            channels=1, 
            dtype="float32", 
            blocksize=STREAM_BLOCK_SIZE, 
            callback=cb
        )
        self._recording_stream.start()
        self._btn_record.configure(text="⏹  Stop Recording", fg_color="#c0392b")
        self._hdr_status.configure(text="🎤 Recording", text_color="#e74c3c")
        msg = "Recording… Speak, then pause." if self._transcribe_mode_var.get() == TRANSCRIBE_MODE_LIVE else "Recording… Speak, then stop."
        self._set_tx_status(msg, "#e74c3c")

    def _stop_recording(self):
        if self._recording_stream:
            self._recording_stream.stop()
            self._recording_stream.close()
            self._recording_stream = None

        with self._audio_lock:
            self._recording = False
            chunks = self._recording_chunks
            self._recording_chunks = []
            self._is_speaking = False
            sd.stop()
            
        if chunks:
            recording = np.concatenate(chunks, axis=0)
            mode = "append" if self._transcribe_mode_var.get() == TRANSCRIBE_MODE_LIVE else "replace"
            self._enqueue_transcription(recording, mode, True)
        else:
            self._finish_ui_reset()

    def _enqueue_transcription(self, audio, mode, stop_after):
        self._set_processing_state(True)
        self._transcribe_queue.put((audio, mode, stop_after))

    def _transcription_queue_worker(self):
        while True:
            audio, mode, stop_after = self._transcribe_queue.get()
            try:
                processed = trim_trailing_silence(audio)
                text = send_to_server(processed, self._model_var.get())
                self.after(0, self._handle_result, text, mode, stop_after)
            finally:
                self._set_processing_state(False)
                self._transcribe_queue.task_done()

    def _handle_result(self, text, mode, stop_after):
        text = text.strip()
        if not text:
            if stop_after:
                self._finish_ui_reset()
            return

        if not text.startswith("["):
            entry = self.manager.add_history_entry(text, self._model_var.get())
            self._render_history_card(entry)
            self._update_history_count()

        self._textbox.configure(state="normal")
        if mode == "replace":
            self._textbox.delete("1.0", "end")
        
        current = self._textbox.get("1.0", "end").strip()
        if current and mode == "append":
            self._textbox.insert("end", " ")
        self._textbox.insert("end", text)
        self._textbox.configure(state="disabled")
        
        threading.Thread(target=self._auto_type_text, args=(text,), daemon=True).start()
        if stop_after:
            self._finish_ui_reset()

    def _finish_ui_reset(self):
        self._btn_record.configure(state="normal", text="⏺  Start Recording", fg_color=self._btn_rec_fg)
        self._hdr_status.configure(text="✓ Ready", text_color="#27ae60")
        self._set_tx_status("Ready. Press Win+G to start.", "gray55")

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _log(self, msg):
        self.after(0, self._append_log_ui, msg)

    def _append_log_ui(self, msg):
        self._log_box.configure(state="normal")
        self._log_box.insert("end", msg)
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _set_tx_status(self, msg, color="gray55"):
        self._tx_status.configure(text=msg, text_color=color)

    def _set_processing_state(self, active):
        with self._processing_lock:
            self._processing_count += 1 if active else -1
            self._processing_count = max(0, self._processing_count)
            self._processing = self._processing_count > 0

    def _update_history_count(self):
        n = len(self.manager.history)
        self._hist_count_label.configure(text=f"{n} transcription{'s' if n != 1 else ''}")

    def _render_history_card(self, entry):
        self._hist_empty.grid_forget()
        card = ctk.CTkFrame(self._hist_scroll, corner_radius=8)
        card.grid(row=len(self.manager.history), column=0, pady=(0, 8), sticky="ew")
        ctk.CTkLabel(card, text=f"{entry['time']} | {entry['model']}", font=ctk.CTkFont(size=10), text_color="gray55").pack(anchor="w", padx=12, pady=(8, 0))
        ctk.CTkLabel(card, text=entry["text"], font=ctk.CTkFont(size=13), wraplength=400, justify="left").pack(anchor="w", padx=12, pady=(4, 8))

    def _health_worker(self):
        while True:
            is_up = self.manager.check_server_health()
            self.after(0, self._set_srv_state, "running" if is_up else "stopped")
            time.sleep(HEALTH_POLL_SEC)

    def _set_srv_state(self, state):
        cfg = {
            "running": {"title": "Server Running", "color": "#27ae60", "badge": "  ONLINE  ", "hdr": "● Server Online"},
            "stopped": {"title": "Server Stopped", "color": "#000000", "badge": "  OFFLINE  ", "hdr": "● Server Offline"},
        }
        c = cfg[state]
        self._srv_status_label.configure(text=c["title"], text_color=c["color"])
        self._srv_status_bar.configure(fg_color=c["color"])
        self._srv_status_badge.configure(text=c["badge"])
        if not self._recording:
            self._hdr_status.configure(text=c["hdr"], text_color=c["color"])

    def _auto_start_srv_worker(self):
        time.sleep(1)
        if not self.manager.server_running:
            self.manager.start_local_server()

    def _hotkey_listener(self):
        def on_activate():
            self.after(0, self._toggle_recording)
        try:
            h = GlobalHotKeys({"<cmd>+g": on_activate})
            h.start()
            h.join()
        except Exception:
            pass

    def _auto_type_text(self, text):
        if not text:
            return
        try:
            time.sleep(0.4)
            kb = KeyboardController()
            for c in text:
                kb.type(c)
                time.sleep(0.005)
        except Exception:
            pass

    def _update_tray_icon(self):
        if hasattr(self, "_tray_icon") and self._tray_icon:
            s = "recording" if self._recording else ("processing" if self._processing else ("running" if self.manager.server_running else "stopped"))
            self._tray_icon.icon = make_status_icon(s)
        self.after(500, self._update_tray_icon)

    def _setup_tray(self):
        try:
            from pystray import Icon, Menu, MenuItem
            m = Menu(MenuItem("Toggle Recording", lambda: self._toggle_recording()), MenuItem("Quit", self._on_close))
            self._tray_icon = Icon("whisper-typer", make_status_icon("stopped"), menu=m)
            threading.Thread(target=self._tray_icon.run, daemon=True).start()
        except Exception:
            pass

    def _download_model_ui(self):
        m = self._model_dl_var.get()
        self._btn_download.configure(state="disabled", text="Downloading...")
        self.manager.download_model(m, lambda success: self.after(0, lambda: self._btn_download.configure(state="normal", text="Download")))

    def _delete_model_ui(self):
        self.manager.delete_model(self._model_dl_var.get())

    def _clear_history(self):
        self.manager.clear_history()
        for w in self._hist_scroll.winfo_children():
            w.destroy()
        self._update_history_count()

    def _clear_logs(self):
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")

    def _clear_transcript_text(self):
        self._textbox.configure(state="normal")
        self._textbox.delete("1.0", "end")
        self._textbox.configure(state="disabled")

    def _on_mode_change(self, val):
        self._transcribe_mode_var.set(val)

    def _on_close(self):
        self.manager.stop_server()
        if hasattr(self, "_tray_icon"):
            self._tray_icon.stop()
        self.destroy()

if __name__ == "__main__":
    WhisperUI().mainloop()
