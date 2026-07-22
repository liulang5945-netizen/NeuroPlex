"""
态极 — 模型发布 API（精简版）
Cortex 神经元架构不适用传统 save_model，仅保留 published 列表查询和 GGUF 不支持消息。
"""
import json as _json
import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from taiji.core.app_state import app_state
from taiji.core.utils import get_external_path

logger = logging.getLogger("ApiServer.Training")
router = APIRouter()


@router.post("/api/model/publish")
async def publish_model():
    """保存当前模型（Cortex 模式下不适用传统 save_model）"""
    raise HTTPException(
        status_code=503,
        detail="Cortex 神经元架构不适用传统模型发布。神经元 ckpt 通过 distill_neurons.py 或 sleep_engine 管理。",
    )


@router.get("/api/model/published")
def list_published_models():
    """列出已发布（保存）的模型"""
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
    """GGUF 导出（态极不支持）"""
    return StreamingResponse(
        _stream_error("态极不支持 GGUF 导出"),
        media_type="text/event-stream",
    )


async def _stream_error(msg: str):
    yield f"data: {_json.dumps({'type': 'error', 'message': msg})}\n\n"


@router.get("/api/model/export_gguf/options")
def get_gguf_export_options():
    """GGUF 量化选项（态极不支持）"""
    return {"options": [], "message": "态极不支持 GGUF 导出"}
