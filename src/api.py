import json
import urllib.request
import urllib.error
import urllib.parse
import numpy as np
from .config import SERVER_IP, SERVER_PORT

def send_to_server(audio: np.ndarray, model: str) -> str:
    """Send PCM audio to the transcription server."""
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
            data = json.loads(resp.read().decode())
            return data.get("text", "")
    except urllib.error.HTTPError as e:
        try:
            error_detail = json.loads(e.read().decode()).get("detail", str(e))
        except Exception:
            error_detail = str(e)
        return f"[Server Error {e.code}: {error_detail}]"
    except urllib.error.URLError as e:
        return f"[Connection Error: {e.reason}. Is the server running?]"
    except TimeoutError:
        return "[Request timed out. The model might be taking too long or the server is hanging.]"
    except Exception as e:
        return f"[Unexpected Client Error: {e}]"

def is_server_reachable(timeout_sec: float = 2.0) -> bool:
    """Check if the transcription server is online and responding."""
    try:
        url = f"http://{SERVER_IP}:{SERVER_PORT}/"
        with urllib.request.urlopen(url, timeout=timeout_sec) as r:
            if r.status != 200:
                return False
            body = json.loads(r.read().decode())
            if isinstance(body, dict):
                return body.get("service") == "whisper-transcribe" or bool(body)
            return bool(body)
    except Exception:
        return False
