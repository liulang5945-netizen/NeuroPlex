"""
原生态极 — 模型发布 API（精简版）
仅支持原生模型保存，不支持 LoRA 合并 / GGUF 导出
"""
import asyncio
import datetime
import json as _json
import logging
import os
import threading
import queue

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from taiji.core.config import TrainingConfig
from taiji.core.app_state import app_state
from taiji.core.utils import get_external_path

from .common import safe_put

logger = logging.getLogger("ApiServer.Training")
router = APIRouter()


@router.post("/api/model/publish")
async def publish_model():
    """保存当前原生模型（SSE 流式进度）"""
    if not app_state.try_start_publishing():
        raise HTTPException(status_code=400, detail="正在发布中，请勿重复操作")
    if not app_state.startup_complete or app_state.model is None:
        app_state.finish_publishing()
        raise HTTPException(status_code=503, detail="模型未加载完成")

    q = queue.Queue()

    def _publish():
        try:
            safe_put(q, {"type": "progress", "phase": "start", "message": "开始保存原生模型..."})
            from taiji.loader import save_model
            import time
            t0 = time.time()

            output_dir = os.path.join(
                get_external_path("checkpoints"),
                f"published_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            os.makedirs(output_dir, exist_ok=True)
            save_model(app_state.model, app_state.tokenizer, output_dir)
            elapsed = time.time() - t0
            safe_put(q, {"type": "complete", "path": output_dir, "elapsed_seconds": round(elapsed, 1)})
        except Exception as exc:
            safe_put(q, {"type": "error", "message": str(exc)})
        finally:
            app_state.finish_publishing()

    threading.Thread(target=_publish, daemon=True).start()

    async def _stream():
        while True:
            try:
                item = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, lambda: q.get(timeout=0.1)),
                    timeout=0.5,
                )
                yield f"data: {_json.dumps(item, ensure_ascii=False)}\n\n"
                if item.get("type") == "complete" or item.get("type") == "error":
                    break
            except asyncio.TimeoutError:
                yield f"data: {_json.dumps({'type': 'heartbeat'})}\n\n"
                if not threading.active_count():
                    break

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.get("/api/model/published")
def list_published_models():
    """列出已发布（保存）的原生模型"""
    checkpoint_dir = get_external_path("checkpoints")
    published = []
    if os.path.isdir(checkpoint_dir):
        for name in sorted(os.listdir(checkpoint_dir)):
            full = os.path.join(checkpoint_dir, name)
            if os.path.isdir(full) and name.startswith("published_"):
                config_path = os.path.join(full, "config.json")
                published.append({
                    "name": name,
                    "path": full,
                    "has_config": os.path.exists(config_path),
                })
    return {"status": "ok", "published": published}


@router.post("/api/model/export_gguf")
async def export_gguf():
    """GGUF 导出（原生态极不支持）"""
    return StreamingResponse(
        _stream_error("原生态极不支持 GGUF 导出"),
        media_type="text/event-stream",
    )


async def _stream_error(msg: str):
    yield f"data: {_json.dumps({'type': 'error', 'message': msg})}\n\n"


@router.get("/api/model/export_gguf/options")
def get_gguf_export_options():
    """GGUF 量化选项（原生态极不支持）"""
    return {"options": [], "message": "原生态极不支持 GGUF 导出"}
