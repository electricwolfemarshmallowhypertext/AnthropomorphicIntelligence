import os
from functools import lru_cache
from typing import Optional

class Settings:
    # Basic configuration
    USE_DUMMY: bool = os.getenv("USE_DUMMY", "0") != "0"
    ALLOWED_ORIGINS: list[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Model configuration
    CKPT_PATH: Optional[str] = os.getenv("CKPT_PATH")
    DEVICE_ID: int = int(os.getenv("DEVICE_ID", "0"))
    
    # Inference configuration
    USE_AUDIO_IN_VIDEO: bool = os.getenv("USE_AUDIO_IN_VIDEO", "0") != "0"
    MAX_KV_TOKENS: int = int(os.getenv("MAX_KV_TOKENS", "16384"))
    ASSISTANT_NUM: int = int(os.getenv("ASSISTANT_NUM", "1"))
    ENABLE_TTS: bool = os.getenv("ENABLE_TTS", "1") != "0"
    SAVE_DIR: str = os.getenv("SAVE_DIR", "./infer_output")
    THRESHOLD: float = float(os.getenv("THRESHOLD", "0.7"))
    
    # Generation configuration
    DO_SAMPLE: bool = os.getenv("DO_SAMPLE", "1") != "0"
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "12"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P: float = float(os.getenv("TOP_P", "0.9"))
    REPETITION_PENALTY: float = float(os.getenv("REPETITION_PENALTY", "1.25"))

@lru_cache(maxsize=1)
def get_settings() -> Settings:
     settings = Settings()
     if not settings.CKPT_PATH:
         raise RuntimeError(
             "CKPT_PATH environment variable is not set. "
             "Please provide a valid checkpoint path via the CKPT_PATH environment variable."
         )
     return settings