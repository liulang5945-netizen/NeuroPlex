"""
多模态输出 API 路由
==================
让前端能调用态极的多模态输出能力。
"""
import os
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from taiji.core.app_state import app_state

logger = logging.getLogger("ApiServer.Multimodal")
router = APIRouter()


def _get_output_engine():
    """获取多模态输出引擎"""
    try:
        from taiji.multimodal.output_engine import MultimodalOutputEngine
        return MultimodalOutputEngine()
    except Exception as e:
        logger.error(f"Failed to get output engine: {e}")
        return None


# ======================== 语音输出 ========================

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None  # xiaoxiao/yunxi/yunyang/xiaoyi/yunjian


@router.post("/api/multimodal/tts")
async def text_to_speech(request: TTSRequest):
    """文字转语音"""
    engine = _get_output_engine()
    if not engine or not engine.speech:
        raise HTTPException(status_code=503, detail="语音引擎不可用")

    try:
        audio_path = engine.text_to_speech(request.text, request.voice)
        if audio_path and os.path.exists(audio_path):
            return FileResponse(
                audio_path,
                media_type="audio/mpeg",
                filename=os.path.basename(audio_path),
            )
        raise HTTPException(status_code=500, detail="语音生成失败")
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return HTTPException(status_code=500, detail="内部错误，请查看日志")


@router.get("/api/multimodal/voices")
async def list_voices():
    """列出可用语音"""
    engine = _get_output_engine()
    if not engine or not engine.speech:
        return {"voices": {}}
    return {"voices": engine.speech.list_voices()}


# ======================== 图像生成 ========================

class ImageGenRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    steps: int = 30
    guidance_scale: float = 7.5


@router.post("/api/multimodal/generate-image")
async def generate_image(request: ImageGenRequest):
    """文生图"""
    engine = _get_output_engine()
    if not engine or not engine.image:
        raise HTTPException(status_code=503, detail="图像引擎不可用")

    try:
        image_path = engine.generate_image(
            request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
        )
        if image_path and os.path.exists(image_path):
            return FileResponse(
                image_path,
                media_type="image/png",
                filename=os.path.basename(image_path),
            )
        raise HTTPException(status_code=500, detail="图像生成失败")
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return HTTPException(status_code=500, detail="内部错误，请查看日志")


# ======================== 图像描述 ========================

class ImageDescribeRequest(BaseModel):
    image_path: str


@router.post("/api/multimodal/describe-image")
async def describe_image(request: ImageDescribeRequest):
    """描述图片内容"""
    engine = _get_output_engine()
    if not engine or not engine.image:
        raise HTTPException(status_code=503, detail="图像引擎不可用")

    try:
        description = engine.describe_image(request.image_path)
        return {"description": description}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return HTTPException(status_code=500, detail="内部错误，请查看日志")


# ======================== 状态查询 ========================

@router.get("/api/multimodal/status")
async def multimodal_status():
    """获取多模态引擎状态"""
    engine = _get_output_engine()
    if not engine:
        return {"available": False}
    return engine.get_status()
