"""FastAPI server — POST /transcribe with raw PCM (16kHz, 16-bit, mono).
Runs inside Docker (download_root=/config) or locally (download_root=<project>/models).
"""
import os
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "tiny")
_MODELS_DIR_DEFAULT = (
    "/config"
    if os.path.isdir("/config")
    else str(Path(__file__).resolve().parents[2] / "models")
)
MODELS_DIR = os.environ.get("WHISPER_MODELS_DIR", _MODELS_DIR_DEFAULT)

# Cache models in memory
_models: dict[str, WhisperModel] = {}


def get_model(model_name: str) -> WhisperModel:
    if model_name not in _models:
        print(f"Loading Whisper model '{model_name}'...")
        _models[model_name] = WhisperModel(
            model_size_or_path=model_name,
            device="auto",
            compute_type="int8",
            download_root=MODELS_DIR,
        )
    return _models[model_name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model files are already on disk (downloaded at image build time).
    # Load into memory now so the first request isn't slow.
    get_model(DEFAULT_MODEL)
    print(f"Model '{DEFAULT_MODEL}' loaded and ready.")
    yield


def trim_trailing_silence(audio: np.ndarray, threshold: float = 0.005) -> np.ndarray:
    flat = audio.flatten()
    mask = np.abs(flat) > threshold
    if not np.any(mask):
        return audio[: int(0.1 * SAMPLE_RATE)]
    last = np.where(mask)[0][-1]
    return flat[: last + 1]


app = FastAPI(title="Whisper Transcribe API", version="1.0", lifespan=lifespan)


@app.get("/")
async def root():
    return {"service": "whisper-transcribe", "docs": "/docs", "transcribe": "POST /transcribe"}


@app.post("/transcribe")
async def transcribe(request: Request, model: str = DEFAULT_MODEL):
    """Accept raw PCM audio in request body (16 kHz, 16-bit, mono). Returns {"text": "..."}."""
    try:
        body = await request.body()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read body: {e}")
    if len(body) < 100:
        raise HTTPException(status_code=400, detail="Audio too short (need at least 100 bytes)")

    try:
        audio_int16 = np.frombuffer(body, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32767.0
        audio_float = trim_trailing_silence(audio_float)

        whisper_model = get_model(model)
        segments, info = whisper_model.transcribe(audio_float, beam_size=1)

        text = "".join(segment.text for segment in segments)
        return JSONResponse(content={"text": text.strip()})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Transcription failed: {e}")
