from typing import List, Optional
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F

import threading  # ✅ Added: concurrent lock

from proactvl.app.core.config import get_settings
from proactvl.infer.multi_assistant_inference import MultiAssistantStreamInference, AssistantResponse
from proactvl.model.modeling_proact import ProAct_OmniModel, ProActConfig
import base64
import numpy as np
import io
import wave

_settings = get_settings()


GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

class Model:
    def __init__(self) -> None:
        # Device / precision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16

        # Concurrent lock and runtime parameters for set_params/set_system_prompt/clear_cache
        self._lock = threading.RLock()
        # self.threshold: float = _settings.THRESHOLD
        threshold = _settings.THRESHOLD
        self.system_prompt: str = ""  # Current system prompt for frontend/logging

        # ===== Load model initialization parameters from config =====
        model_config = None
        ckpt_path = _settings.CKPT_PATH
        device_id = _settings.DEVICE_ID
        infer_config = {
            'use_audio_in_video': _settings.USE_AUDIO_IN_VIDEO,
            'max_kv_tokens': _settings.MAX_KV_TOKENS,
            'assistant_num': _settings.ASSISTANT_NUM,
            'enable_tts': _settings.ENABLE_TTS,
            'save_dir': _settings.SAVE_DIR,
        }
        generate_config = {
            'do_sample': _settings.DO_SAMPLE,
            'max_new_tokens': _settings.MAX_NEW_TOKENS,
            'temperature': _settings.TEMPERATURE,
            'top_p': _settings.TOP_P,
            'repetition_penalty': _settings.REPETITION_PENALTY,
        }
        talker_config = None
        self.model = MultiAssistantStreamInference(
            model_config, 
            ckpt_path, 
            infer_config, 
            generate_config, 
            talker_config, 
            f'cuda:{device_id}'
        )
        for assistant in self.model.assistants:
            assistant.clear_session()
            assistant.prime_system_prompt()
            assistant.set_threshold(threshold)
        # self.model.assistants[0].prime_system_prompt()
        # # ✅ Ensure gating threshold matches the local setting
        # self.model.assistants[0].state_threshold = self.threshold

        # Preprocessing (infer_one_chunk may already handle this, but keep the custom path here)
        # self.tf = T.Compose([
        #     T.Resize(448, antialias=True),
        #     T.CenterCrop(448),
        #     T.ToTensor(),
        #     T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        # ])
        self.input_size = None

        self.previous_info = None

    # ===== Parameter / cache interfaces =====
    def set_params(self, threshold: Optional[float] = None, assistant_id: Optional[int] = None) -> None:
        """Called by backend WS while commentary is paused (WS layer already guarantees idle state)."""
        with self._lock:
            if threshold is not None:
                print(f"Setting threshold to {threshold}")
                # thr = min(1.0, max(0.0, float(threshold)))
                if assistant_id is None:
                    # Apply to all assistants
                    for assistant in self.model.assistants:
                        assistant.set_threshold(float(threshold))
                else:
                    # Apply only to the specified assistant
                    for assistant in self.model.assistants:
                        if assistant.assistant_id == assistant_id:
                            assistant.set_threshold(float(threshold))

    def set_system_prompt(self, prompt: str, *, assistant_id: Optional[int] = None) -> None:
        print(f"Setting system prompt to: {prompt}")
        print(f"Assistant ID: {assistant_id}")
        with self._lock:
            if assistant_id is None:
                # Apply to all assistants
                for assistant in self.model.assistants:
                    assistant.clear_session()
                    assistant.prime_system_prompt(prompt)
            else:
                # Apply only to the specified assistant
                for assistant in self.model.assistants:
                    if assistant.assistant_id == assistant_id:
                        assistant.clear_session()
                        assistant.prime_system_prompt(prompt)

    def set_assistant_count(self, count: int) -> None:
        # print(f"Setting assistant count to: {count}")
        with self._lock:
            # Assume the model exposes a method to set assistant count here
            self.model.set_assistant_count(count)

    def clear_cache(self, assistant_id: Optional[int] = None) -> None:
        with self._lock:
            # Clear runtime state (KV/history context, etc.)
            if assistant_id is None:
                # Apply to all assistants
                for assistant in self.model.assistants:
                    assistant.clear_session()
            else:
                # Apply only to the specified assistant
                for assistant in self.model.assistants:
                    if assistant.assistant_id == assistant_id:
                        assistant.clear_session()

    # ===== Image preprocessing: 0-255 float16, no normalization =====
    def _preprocess(self, frames: List[Image.Image]) -> torch.Tensor:
        tensors = []
        for img in frames:
            if img.mode != "RGB":
                img = img.convert("RGB")
            if self.input_size is not None:
                # Normalize geometry; resize only (no crop)
                img = F.resize(img, self.input_size, antialias=True)
            t = F.pil_to_tensor(img)             # uint8, [C,H,W], 0..255
            t = t.to(dtype=torch.float16)        # float16, 0..255 (do not divide by 255)
            tensors.append(t)
        x = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)  # [B,C,H,W]
        return x

    @torch.inference_mode()
    def infer(self, video: List[Image.Image], query: Optional[str]) -> str:
        """
        video: two PIL.Image frames
        query: optional text
        return: model-generated commentary
        """
        if _settings.USE_DUMMY:
            wh = f"{video[0].width}x{video[0].height}" if (video and video[0]) else "NA"
            q = (query or "").strip() or "(none)"
            # Show current parameters to make it easy to observe whether set_params has taken effect
            return f"[demo] Received 2 frames at resolution {wh}; threshold={self.model.assistants[0].threshold:.2f}; SysPromptLen={len(self.system_prompt)}; Query: {q}"

        # Actual inference path
        x = self._preprocess(video)  # [2,3,448,448]
        q = (query or "").strip()
        print(f"Model infer called with query: {q}")
        # Note: infer_one_chunk parameter signature is kept unchanged; depending on your implementation,
        # videos may need shape [B,T,C,H,W]. Here x.unsqueeze(0) -> [1,2,3,448,448], which is fine if that is expected.
        response, audio = self.model.infer_one_chunk_backend(
            audios=None,
            images=None,
            videos=[x],
            user_query=q,
            previous_assistant_responses=self.previous_info,
            begin_second=0,
        )
        print(response)
        speaker_name = None
        active_commentary = "<|SILENCE|>"
        active_speaker_id = None
        self.previous_info = response
        for a in self.model.assistants:
            id = a.assistant_id
            commentary = response[id].commentary
            commentary = commentary.strip() if commentary else "<|SILENCE|>"
            print(f'Assistant {a.assistant_id} active: {RED}{response[id].active}{RESET}, score: {RED}{response[id].score}{RESET}, response: {GREEN}{commentary}{RESET}')
            if response[id].active:
                active_commentary = commentary
                active_speaker_id = id
                score = response[id].score

        # print(f'commentary: {commentary}, speaker_name: {speaker_name}, audio length: {audio.shape[0] if audio is not None else "NA"}, score threshold: {self.model.assistants[0].state_threshold}, score: {score}')
        audio = numpy_to_wav_base64(audio) if audio is not None and audio.shape[0] != 0 else None
        # print(f'audio shape: {audio if audio is not None else "NA"}, speaker_id: {active_speaker_id}')
        return active_commentary, audio, active_speaker_id

# Model singleton
_model_singleton: Optional[Model] = None
def get_model() -> Model:
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = Model()
    return _model_singleton

# ====== Convert numpy audio to WAV (Base64) ======
def numpy_to_wav_base64(
    audio: np.ndarray,
    sample_rate: int = 24000,
    num_channels: int = 1,
) -> str:
    """
    Encode a numpy audio array as WAV, then convert it to a base64 string.

    Supports:
    - audio: shape (T,), (1, T), (T, 1), and other 1D/2D layouts
    - dtype: float32/float64 (in [-1,1]) or int16
    """
    if audio is None:
        raise ValueError("audio is None")

    # 1) First convert to a 1D layout when appropriate
    audio = np.asarray(audio)
    if audio.ndim == 2:
        # Common layouts: (1, T) or (T, 1)
        if 1 in audio.shape:
            audio = audio.reshape(-1)
        else:
            # If this is truly multi-channel, you can extend it later; for now, treat each column as one channel
            # e.g. (T, C)
            if audio.shape[1] == num_channels:
                pass  # Multi-channel handling is done below
            else:
                # Keep it simple for now: use the first column as mono
                audio = audio[:, 0]
    if audio.ndim == 1:
        # Single-channel
        num_channels = 1
    else:
        # Multi-channel (T, C)
        if audio.shape[1] != num_channels:
            num_channels = audio.shape[1]

    # 2) Convert uniformly to int16 PCM
    if np.issubdtype(audio.dtype, np.floating):
        # Assume values are in [-1, 1] and guard against overflow
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767.0).astype(np.int16)
    elif audio.dtype == np.int16:
        audio_int16 = audio
    else:
        # Force all other types to int16 as well
        audio_int16 = audio.astype(np.int16)

    # Multi-channel case: shape (T, C) -> row-wise interleave
    if audio_int16.ndim == 2:
        # Flatten (T, C) to 1D; the wave module reads it in little-endian order
        interleaved = audio_int16.reshape(-1)
    else:
        interleaved = audio_int16.reshape(-1)

    # 3) Write as WAV using wave + BytesIO
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)           # int16 = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved.tobytes())

    wav_bytes = buf.getvalue()

    # 4) Base64-encode to str
    return base64.b64encode(wav_bytes).decode("ascii")
