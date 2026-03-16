import io
import base64
from PIL import Image

def b64jpeg_to_pil(data_b64: str) -> Image.Image:
    """将 base64（不含前缀）JPEG 解码为 RGB PIL.Image。"""
    raw = base64.b64decode(data_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")
