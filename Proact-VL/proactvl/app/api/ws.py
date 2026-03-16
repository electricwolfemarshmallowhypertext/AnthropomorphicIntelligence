import json
import logging
from typing import List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from proactvl.app.models.model import get_model
from proactvl.app.schemas.messages import (
    Hello, InferRequest,
    SetParamsRequest, ClearCacheRequest, SetSystemPromptRequest, SetAssistantCountRequest,
    CommentResponse, ErrorResponse, StatusResponse
)
from proactvl.app.utils.images import b64jpeg_to_pil

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws", tags=["ws"])

@router.websocket("/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    logger.info("WS connected")

    # 连接级别的“忙碌”标记：有推理在跑时，禁止修改参数
    infer_busy = False

    try:
        while True:
            raw = await ws.receive_text()
            # 先解析 type
            try:
                data = json.loads(raw)
                mtype = data.get("type")
            except Exception as e:
                await ws.send_text(ErrorResponse(text=f"invalid json: {e}").model_dump_json())
                continue

            # ===== hello =====
            if mtype == "hello":
                try:
                    _ = Hello(**data)
                except Exception:
                    # 放宽：hello 校验失败也直接回
                    pass
                await ws.send_text(StatusResponse(text="hello received").model_dump_json())
                continue

            # ===== 推理 =====
            if mtype == "infer":
                try:
                    req = InferRequest(**data)
                except Exception as e:
                    await ws.send_text(ErrorResponse(text=f"bad infer payload: {e}").model_dump_json())
                    continue

                # 解码两帧
                pil_frames: List = []
                try:
                    for f in req.frames[:2]:
                        pil_frames.append(b64jpeg_to_pil(f.data))
                except Exception as e:
                    await ws.send_text(ErrorResponse(request_id=req.request_id, text=f"frame decode error: {e}").model_dump_json())
                    continue

                try:
                    infer_busy = True
                    model = get_model()
                    commentary, audio, speaker_id = await run_in_threadpool(model.infer, pil_frames, req.query)
                    audio_mime = "audio/wav" if audio else None
                except Exception as e:
                    logger.exception("infer failed")
                    await ws.send_text(ErrorResponse(request_id=req.request_id, text=f"infer error: {e}").model_dump_json())
                else:
                    resp = CommentResponse(request_id=req.request_id, text=commentary, audio_mime=audio_mime, audio_base64=audio, speaker=speaker_id)
                    await ws.send_text(resp.model_dump_json())
                finally:
                    infer_busy = False

                continue

            # ===== 控制：设置阈值 =====
            if mtype == "set_params":
                rid = data.get("request_id")
                if infer_busy:
                    await ws.send_text(ErrorResponse(request_id=rid, text="busy: inference in progress").model_dump_json())
                    continue
                try:
                    req = SetParamsRequest(**data)
                except Exception as e:
                    await ws.send_text(ErrorResponse(request_id=rid, text=f"bad set_params: {e}").model_dump_json())
                    continue

                try:
                    model = get_model()
                    await run_in_threadpool(model.set_params, threshold=req.threshold, assistant_id=req.assistant_id)
                    await ws.send_text(StatusResponse(request_id=rid, text="ok").model_dump_json())
                except Exception as e:
                    await ws.send_text(ErrorResponse(request_id=rid, text=f"set_params error: {e}").model_dump_json())
                continue

            # ===== 控制：清空缓存 =====
            if mtype == "clear_cache":
                rid = data.get("request_id")
                if infer_busy:
                    await ws.send_text(ErrorResponse(request_id=rid, text="busy: inference in progress").model_dump_json())
                    continue
                try:
                    req = ClearCacheRequest(**data)
                except Exception as e:
                    await ws.send_text(ErrorResponse(request_id=rid, text=f"bad clear_cache: {e}").model_dump_json())
                    continue

                try:
                    model = get_model()
                    await run_in_threadpool(model.clear_cache, assistant_id=req.assistant_id)
                    await ws.send_text(StatusResponse(request_id=rid, text="ok").model_dump_json())
                except Exception as e:
                    await ws.send_text(ErrorResponse(request_id=rid, text=f"clear_cache error: {e}").model_dump_json())
                continue

            # ===== 控制：设置系统提示 =====
            if mtype == "set_system_prompt":
                rid = data.get("request_id")
                print(data)
                if infer_busy:
                    await ws.send_text(ErrorResponse(request_id=rid, text="busy: inference in progress").model_dump_json())
                    continue
                try:
                    req = SetSystemPromptRequest(**data)
                except Exception as e:
                    await ws.send_text(ErrorResponse(request_id=rid, text=f"bad set_system_prompt: {e}").model_dump_json())
                    continue

                try:
                    model = get_model()
                    await run_in_threadpool(model.set_system_prompt, req.system_prompt, assistant_id=req.assistant_id)
                    await ws.send_text(StatusResponse(request_id=rid, text="ok").model_dump_json())
                except Exception as e:
                    await ws.send_text(ErrorResponse(request_id=rid, text=f"set_system_prompt error: {e}").model_dump_json())
                continue

            # ===== ADDED: 控制：设置 Assistant 数量 =====
            if mtype == "set_assistant_count":
                rid = data.get("request_id")
                print(data)
                if infer_busy:
                    await ws.send_text(ErrorResponse(request_id=rid, text="busy: inference in progress").model_dump_json())
                    continue
                try:
                    # 1. 验证请求
                    req = SetAssistantCountRequest(**data)
                except Exception as e:
                    await ws.send_text(ErrorResponse(request_id=rid, text=f"bad set_assistant_count: {e}").model_dump_json())
                    continue
                try:
                    # 2. 获取模型并调用 (您需要在这里实现 'set_assistant_count' 逻辑)
                    model = get_model()
                    await run_in_threadpool(model.set_assistant_count, req.count)
                    await ws.send_text(StatusResponse(request_id=rid, text="ok").model_dump_json())

                    logger.info(f"Received request to set assistant count to: {req.count}")
                    await ws.send_text(StatusResponse(request_id=rid, text=f"Assistant count set to {req.count}").model_dump_json())
                
                except Exception as e:
                    logger.exception("set_assistant_count failed")
                    await ws.send_text(ErrorResponse(request_id=rid, text=f"set_assistant_count error: {e}").model_dump_json())
                
                continue
            # ===== END ADDED =====
            # 未知类型
            await ws.send_text(ErrorResponse(text=f"unknown type: {mtype}").model_dump_json())

    except WebSocketDisconnect:
        logger.info("WS disconnected")
    except Exception as e:
        logger.exception("WS error: %s", e)
        try:
            await ws.close()
        except Exception:
            pass
