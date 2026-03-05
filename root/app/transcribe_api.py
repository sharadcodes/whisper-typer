"""FastAPI server — POST /transcribe with raw PCM (16kHz, 16-bit, mono).
Runs inside Docker (download_root=/config) or locally (download_root=<project>/models).
"""
import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("whisper-api")

SAMPLE_RATE = 16000
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "small")
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
        logger.info("Loading Whisper model '%s'...", model_name)
        try:
            _models[model_name] = WhisperModel(
                model_size_or_path=model_name,
                device="auto",
                compute_type="int8",
                download_root=MODELS_DIR,
            )
        except Exception as e:
            logger.error("Failed to load model '%s': %s", model_name, e)
            raise RuntimeError(f"Could not load Whisper model '{model_name}': {e}")
    return _models[model_name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load default model
    try:
        get_model(DEFAULT_MODEL)
        logger.info("Model '%s' loaded and ready.", DEFAULT_MODEL)
    except Exception as e:
        logger.error("Lifespan startup failed: %s", e)
    yield


def trim_trailing_silence(audio: np.ndarray, threshold: float = 0.005) -> np.ndarray:
    if audio.size == 0:
        return audio
    flat = audio.flatten()
    mask = np.abs(flat) > threshold
    if not np.any(mask):
        return np.empty(0, dtype=audio.dtype)
    last = np.where(mask)[0][-1]
    return flat[: last + 1]


app = FastAPI(title="Whisper Transcribe API", version="1.1", lifespan=lifespan)


@app.get("/")
async def root():
    return {
        "service": "whisper-transcribe",
        "status": "online",
        "default_model": DEFAULT_MODEL,
        "loaded_models": list(_models.keys())
    }


@app.post("/transcribe")
async def transcribe(
    request: Request,
    model: str = Query(default=DEFAULT_MODEL, description="Whisper model name")
):
    """Accept raw PCM audio in request body (16 kHz, 16-bit, mono). Returns {"text": "..."}."""
    # 1. Read body
    try:
        body = await request.body()
    except Exception as e:
        logger.error("Body read error: %s", e)
        raise HTTPException(status_code=400, detail=f"Failed to read request body: {e}")

    if not body:
        raise HTTPException(status_code=400, detail="Empty request body")
    
    if len(body) < 100:
        raise HTTPException(status_code=400, detail="Audio too short (need at least 100 bytes of PCM)")

    # 2. Process audio
    try:
        audio_int16 = np.frombuffer(body, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32767.0
        audio_float = trim_trailing_silence(audio_float)
        
        if audio_float.size == 0:
            return JSONResponse(content={"text": "", "language": "en"})
            
    except Exception as e:
        logger.error("Audio processing error: %s", e)
        raise HTTPException(status_code=422, detail=f"Invalid PCM data: {e}")

    # 3. Transcribe
    try:
        whisper_model = get_model(model)
        segments, info = whisper_model.transcribe(audio_float, beam_size=1)
        text = "".join(segment.text for segment in segments)
        
        result = text.strip()
        logger.info("Transcribed %d bytes -> %d chars", len(body), len(result))
        return JSONResponse(content={"text": result, "language": info.language})

    except RuntimeError as e:
        logger.error("Model error: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected transcription failure")
        raise HTTPException(status_code=500, detail=f"Internal transcription error: {e}")
