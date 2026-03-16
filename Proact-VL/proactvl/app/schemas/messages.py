from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class Frame(BaseModel):
    ts: int
    w: int
    h: int
    mime: Literal["image/jpeg"] = "image/jpeg"
    data: str = Field(..., description="Base64 data without the data: prefix")

class Hello(BaseModel):
    type: Literal["hello"]
    frames_per_sec: Optional[int] = None
    want_comment_hz: Optional[int] = None

class InferRequest(BaseModel):
    type: Literal["infer"]
    request_id: int
    ts: int
    source: Literal["webcam", "screen", "file", "unknown"] = "unknown"
    frames: List[Frame]
    query: Optional[str] = None

# ===== Added: control messages =====
class SetParamsRequest(BaseModel):
    type: Literal["set_params"]
    request_id: int
    threshold: float = Field(..., ge=0.0, le=1.0)
    assistant_id: Optional[int] = None  # If None, applies to all assistants

class ClearCacheRequest(BaseModel):
    type: Literal["clear_cache"]
    request_id: int
    assistant_id: Optional[int] = None  # If None, applies to all assistants

class SetSystemPromptRequest(BaseModel):
    type: Literal["set_system_prompt"]
    request_id: int
    system_prompt: str = Field("", max_length=5000)
    assistant_id: Optional[int] = None  # If None, applies to all assistants

class SetAssistantCountRequest(BaseModel):
    type: Literal["set_assistant_count"]
    request_id: int
    count: int = Field(..., ge=1, le=10)
# ===== Responses =====
class CommentResponse(BaseModel):
    type: Literal["comment"] = "comment"
    request_id: int
    text: str
    audio_mime: Optional[str] = None         # e.g. "audio/wav"
    audio_base64: Optional[str] = None       # Base64-encoded WAV bytes
    speaker: Optional[int] = None            # Recommended unified name: speaker

class ErrorResponse(BaseModel):
    type: Literal["error"] = "error"
    request_id: Optional[int] = None
    text: str

class StatusResponse(BaseModel):
    type: Literal["status"] = "status"
    request_id: Optional[int] = None
    text: str
