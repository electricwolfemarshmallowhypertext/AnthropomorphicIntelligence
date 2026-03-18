import io
import base64
from PIL import Image

def b64jpeg_to_pil(data_b64: str) -> Image.Image:
    """Decode base64 JPEG data (without prefix) into an RGB PIL.Image."""
    raw = base64.b64decode(data_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")
