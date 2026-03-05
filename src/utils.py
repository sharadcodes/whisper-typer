import numpy as np
from PIL import Image, ImageDraw
from .config import SAMPLE_RATE

def trim_trailing_silence(audio: np.ndarray, threshold: float = 0.005) -> np.ndarray:
    """Trim silence from the end of an audio segment."""
    flat = audio.flatten()
    mask = np.abs(flat) > threshold
    if not np.any(mask):
        # Fallback for pure silence
        return flat[: int(0.1 * SAMPLE_RATE)].reshape(-1, 1).astype(np.float32)
    last = int(np.where(mask)[0][-1])
    return flat[: last + 1].reshape(-1, 1).astype(np.float32)

def make_status_icon(status: str) -> Image.Image:
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
