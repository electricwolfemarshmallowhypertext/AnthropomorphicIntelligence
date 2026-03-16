from typing import List, Optional
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F

import threading  # ✅ 新增：并发锁

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
        # 设备/精度
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16

        # Concurrent lock and runtime parameters for set_params/set_system_prompt/clear_cache
        self._lock = threading.RLock()
        self.threshold: float = _settings.THRESHOLD
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
            assistant.set_threshold(self.threshold)
        # self.model.assistants[0].prime_system_prompt()
        # # ✅ 确保门控阈值与本地一致
        # self.model.assistants[0].state_threshold = self.threshold

        # 预处理（虽然 infer_one_chunk 里可能自带处理，这里保留你自定义）
        # self.tf = T.Compose([
        #     T.Resize(448, antialias=True),
        #     T.CenterCrop(448),
        #     T.ToTensor(),
        #     T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        # ])
        self.input_size = None

        self.previous_info = None

    # ===== 参数/缓存接口 =====
    def set_params(self, threshold: Optional[float] = None, assistant_id: Optional[int] = None) -> None:
        """在“暂停评论”状态下由后端 WS 调用（已在 ws 层保证空闲时才允许）。"""
        with self._lock:
            if threshold is not None:
                print(f"Setting threshold to {threshold}")
                # thr = min(1.0, max(0.0, float(threshold)))
                if assistant_id is None:
                    # 对所有助手生效
                    for assistant in self.model.assistants:
                        assistant.set_threshold(float(threshold))
                else:
                    # 只对指定助手生效
                    for assistant in self.model.assistants:
                        if assistant.assistant_id == assistant_id:
                            assistant.set_threshold(float(threshold))

    def set_system_prompt(self, prompt: str, *, assistant_id: Optional[int] = None) -> None:
        print(f"Setting system prompt to: {prompt}")
        print(f"Assistant ID: {assistant_id}")
        with self._lock:
            if assistant_id is None:
                # 对所有助手生效
                for assistant in self.model.assistants:
                    assistant.clear_session()
                    assistant.prime_system_prompt(prompt)
            else:
                # 只对指定助手生效
                for assistant in self.model.assistants:
                    if assistant.assistant_id == assistant_id:
                        assistant.clear_session()
                        assistant.prime_system_prompt(prompt)

    def set_assistant_count(self, count: int) -> None:
        # print(f"Setting assistant count to: {count}")
        with self._lock:
            # 这里假设 model 有方法可以设置助手数量
            self.model.set_assistant_count(count)

    def clear_cache(self, assistant_id: Optional[int] = None) -> None:
        with self._lock:
            # 清理运行态（KV/历史上下文等）
            if assistant_id is None:
                # 对所有助手生效
                for assistant in self.model.assistants:
                    assistant.clear_session()
            else:
                # 只对指定助手生效
                for assistant in self.model.assistants:
                    if assistant.assistant_id == assistant_id:
                        assistant.clear_session()

    # ===== 图像预处理：0-255 float16，不归一化 =====
    def _preprocess(self, frames: List[Image.Image]) -> torch.Tensor:
        tensors = []
        for img in frames:
            if img.mode != "RGB":
                img = img.convert("RGB")
            if self.input_size is not None:
                # 统一几何尺寸；只 resize（不裁剪）
                img = F.resize(img, self.input_size, antialias=True)
            t = F.pil_to_tensor(img)             # uint8, [C,H,W], 0..255
            t = t.to(dtype=torch.float16)        # float16, 0..255（不除以 255）
            tensors.append(t)
        x = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)  # [B,C,H,W]
        return x

    @torch.inference_mode()
    def infer(self, video: List[Image.Image], query: Optional[str]) -> str:
        """
        video: 两帧 PIL.Image
        query: 可选文本
        return: 模型生成的 commentary
        """
        if _settings.USE_DUMMY:
            wh = f"{video[0].width}x{video[0].height}" if (video and video[0]) else "NA"
            q = (query or "").strip() or "（无）"
            # 展示当前参数，便于你观测 set_params 是否生效
            return f"[demo] 接收2帧 分辨率{wh}；阈值={self.threshold:.2f}；SysPromptLen={len(self.system_prompt)}；Query：{q}"

        # 你的真实推理路径
        x = self._preprocess(video)  # [2,3,448,448]
        q = (query or "").strip()
        print(f"Model infer called with query: {q}")
        # 注意：infer_one_chunk 的参数签名我保持你原样；根据你内部实现，videos 维度可能需要 [B,T,C,H,W]
        # 你这里是 x.unsqueeze(0) -> [1,2,3,448,448]，若内部期望 [B,T,C,H,W] 则没问题。
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

# 模型单例
_model_singleton: Optional[Model] = None
def get_model() -> Model:
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = Model()
    return _model_singleton

# ====== 把 numpy 音频转成 WAV(Base64) ======
def numpy_to_wav_base64(
    audio: np.ndarray,
    sample_rate: int = 24000,
    num_channels: int = 1,
) -> str:
    """
    将 numpy 音频数组编码成 WAV，再转 base64 字符串。

    支持：
    - audio: shape (T,), (1, T), (T, 1) 等一维/二维
    - dtype: float32/float64（-1~1）或 int16
    """
    if audio is None:
        raise ValueError("audio is None")

    # 1) 先转成一维
    audio = np.asarray(audio)
    if audio.ndim == 2:
        # 常见形式： (1, T) or (T, 1)
        if 1 in audio.shape:
            audio = audio.reshape(-1)
        else:
            # 如果真的是多通道你可以自己扩展，这里先简单做“每列一个通道”
            # 比如 (T, C)
            if audio.shape[1] == num_channels:
                pass  # 下面会按多通道处理
            else:
                # 暂时简单点：直接取第一列当单声道
                audio = audio[:, 0]
    if audio.ndim == 1:
        # 单通道
        num_channels = 1
    else:
        # 多通道 (T, C)
        if audio.shape[1] != num_channels:
            num_channels = audio.shape[1]

    # 2) 统一转成 int16 PCM
    if np.issubdtype(audio.dtype, np.floating):
        # 假设在 [-1, 1]，防一下溢出
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767.0).astype(np.int16)
    elif audio.dtype == np.int16:
        audio_int16 = audio
    else:
        # 其他类型也强制转成 int16
        audio_int16 = audio.astype(np.int16)

    # 多通道情况：shape (T, C) -> 按行 interleave
    if audio_int16.ndim == 2:
        # (T, C) 展成一维，wave 模块按小端序读
        interleaved = audio_int16.reshape(-1)
    else:
        interleaved = audio_int16.reshape(-1)

    # 3) 使用 wave + BytesIO 写成 WAV
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)           # int16 = 2 字节
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved.tobytes())

    wav_bytes = buf.getvalue()

    # 4) base64 编码成 str
    return base64.b64encode(wav_bytes).decode("ascii")
