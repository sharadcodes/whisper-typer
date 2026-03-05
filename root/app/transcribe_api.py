"""FastAPI server inside the container: POST /transcribe with raw PCM (16kHz, 16-bit, mono)."""
import os
import io
import wave
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000

app = FastAPI(title="Whisper Transcribe API", version="1.0")

# Cache models in memory
_models = {}

def get_model(model_name: str) -> WhisperModel:
    if model_name not in _models:
        print(f"Loading Whisper model '{model_name}'...")
        # CPU only by default. Change device="cuda" if you have GPU support.
        # "auto" compute_type will fall back to int8 on CPU.
        _models[model_name] = WhisperModel(
            model_size_or_path=model_name,
            device="auto",
            compute_type="int8",
            download_root="/config",
        )
    return _models[model_name]

def trim_trailing_silence(audio: np.ndarray, threshold: float = 0.005) -> np.ndarray:
    flat = audio.flatten()
    mask = np.abs(flat) > threshold
    if not np.any(mask):
        return audio[: int(0.1 * SAMPLE_RATE)]
    last = np.where(mask)[0][-1]
    return flat[: last + 1]

@app.get("/")
async def root():
    return {"service": "whisper-transcribe", "docs": "/docs", "transcribe": "POST /transcribe"}

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="Raw PCM 16kHz 16-bit mono"),
    model: str = "tiny"
):
    """Accept raw PCM audio: 16 kHz, 16-bit, mono. Returns {"text": "..."}."""
    try:
        body = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {e}")
    if len(body) < 100:
        raise HTTPException(status_code=400, detail="Audio too short (need at least 100 bytes)")

    try:
        # Convert 16-bit PCM to float32 expected by Whisper
        audio_int16 = np.frombuffer(body, dtype=np.int16)
        audio_float = (audio_int16.astype(np.float32) / 32767.0)
        audio_float = trim_trailing_silence(audio_float)

        # Transcribe directly
        whisper_model = get_model(model)
        segments, info = whisper_model.transcribe(audio_float, beam_size=1)
        
        text = "".join(segment.text for segment in segments)
        return JSONResponse(content={"text": text.strip()})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Transcription failed: {e}")
