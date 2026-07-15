"""
多模态高级 API 路由（内部代号：态极）
仅在加载 ModelSelf 原生模型时可用。
其他模型类型下，所有端点返回通用 404，不暴露任何内部信息。
"""
import asyncio
import json
import logging
import os
import time

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import threading
import queue

from taiji.core.app_state import app_state
from taiji.core.utils import get_external_path

logger = logging.getLogger("ApiServer.Taiji")
router = APIRouter()


def _is_available() -> bool:
    """检查多模态高级功能是否可用（静默检查，不暴露内部代号）"""
    return app_state.is_taiji() and app_state.get_taiji_engine() is not None


def _get_engine():
    """获取引擎实例，不可用时返回通用 404"""
    if not _is_available():
        raise HTTPException(status_code=404, detail="接口不存在")
    return app_state.get_taiji_engine()


def _not_found():
    """统一的不可用响应"""
    raise HTTPException(status_code=404, detail="接口不存在")


# ======================== 状态查询 ========================

@router.get("/api/taiji/status")
def taiji_status():
    """获取多模态引擎状态（仅激活时返回有效数据）"""
    if not _is_available():
        raise HTTPException(status_code=404, detail="接口不存在")
    engine = app_state.get_taiji_engine()
    try:
        status = engine.get_status()
        status["available"] = True
        # 对外隐藏内部代号
        status.pop("name", None)
        return status
    except Exception as e:
        return {"available": False}


@router.get("/api/taiji/tools")
def taiji_tools():
    """获取可用的多模态工具列表"""
    if not _is_available():
        raise HTTPException(status_code=404, detail="接口不存在")
    engine = app_state.get_taiji_engine()
    return {"tools": engine.MULTIMODAL_TOOLS}


# ======================== 文件上传 ========================

@router.post("/api/taiji/upload")
async def taiji_upload(file: UploadFile = File(...)):
    """上传文件供多模态处理（图片/音频/视频）"""
    if not _is_available():
        raise HTTPException(status_code=404, detail="接口不存在")

    try:
        upload_dir = get_external_path(os.path.join("user_data", "multimodal_uploads"))
        os.makedirs(upload_dir, exist_ok=True)
        safe_name = f"{int(time.time() * 1000)}_{os.path.basename(file.filename)}"
        file_path = os.path.join(upload_dir, safe_name)
        with open(file_path, "wb") as buffer:
            import shutil
            shutil.copyfileobj(file.file, buffer)

        ext = os.path.splitext(file.filename)[1].lower()
        is_image = ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
        is_audio = ext in {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}
        is_video = ext in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif"}

        modality = "unknown"
        if is_image:
            modality = "image"
        elif is_audio:
            modality = "audio"
        elif is_video:
            modality = "video"

        return {
            "status": "ok",
            "filename": file.filename,
            "saved_path": file_path,
            "public_url": f"/multimodal_media/{safe_name}",
            "modality": modality,
            "file_ext": ext,
        }
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail="文件上传失败，请查看服务端日志")


# ======================== 请求模型 ========================

class ImageRequest(BaseModel):
    image_path: str
    prompt: Optional[str] = ""

class ImageGenRequest(BaseModel):
    prompt: str
    size: Optional[str] = "512x512"

class AudioRequest(BaseModel):
    audio_path: str

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "zh-CN-XiaoxiaoNeural"

class VideoRequest(BaseModel):
    video_path: str

class VideoGenRequest(BaseModel):
    prompt: str
    duration: Optional[int] = 4

class VideoGifRequest(BaseModel):
    video_path: str
    fps: Optional[float] = 4.0

class MultimodalRunRequest(BaseModel):
    task: str
    tools: Optional[list] = None


# ======================== 图像模态 ========================

@router.post("/api/taiji/describe_image")
def describe_image(req: ImageRequest):
    """描述图像内容"""
    engine = _get_engine()
    try:
        result = engine._execute_multimodal_tool("describe_image", {"input": req.image_path})
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"图像描述失败: {e}")
        raise HTTPException(status_code=500, detail="图像描述失败，请查看服务端日志")


@router.post("/api/taiji/generate_image")
def generate_image(req: ImageGenRequest):
    """根据文字描述生成图像"""
    engine = _get_engine()
    try:
        input_str = f"{req.prompt} | {req.size}" if req.size else req.prompt
        result = engine._execute_multimodal_tool("generate_image", {"input": input_str})
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"图像生成失败: {e}")
        raise HTTPException(status_code=500, detail="图像生成失败，请查看服务端日志")


# ======================== 音频模态 ========================

@router.post("/api/taiji/transcribe_audio")
def transcribe_audio(req: AudioRequest):
    """语音识别（音频转文字）"""
    engine = _get_engine()
    try:
        result = engine._execute_multimodal_tool("transcribe_audio", {"input": req.audio_path})
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"语音识别失败: {e}")
        raise HTTPException(status_code=500, detail="语音识别失败，请查看服务端日志")


@router.post("/api/taiji/text_to_speech")
def text_to_speech(req: TTSRequest):
    """语音合成（文字转语音）"""
    engine = _get_engine()
    try:
        input_str = f"{req.text} | {req.voice}" if req.voice else req.text
        result = engine._execute_multimodal_tool("text_to_speech", {"input": input_str})
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"语音合成失败: {e}")
        raise HTTPException(status_code=500, detail="语音合成失败，请查看服务端日志")


# ======================== 视频模态 ========================

@router.post("/api/taiji/understand_video")
def understand_video(req: VideoRequest):
    """理解视频内容"""
    engine = _get_engine()
    try:
        result = engine._execute_multimodal_tool("understand_video", {"input": req.video_path})
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"视频理解失败: {e}")
        raise HTTPException(status_code=500, detail="视频理解失败，请查看服务端日志")


@router.post("/api/taiji/generate_video")
def generate_video(req: VideoGenRequest):
    """根据文字描述生成视频"""
    engine = _get_engine()
    try:
        input_str = f"{req.prompt} | {req.duration}" if req.duration else req.prompt
        result = engine._execute_multimodal_tool("generate_video", {"input": input_str})
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"视频生成失败: {e}")
        raise HTTPException(status_code=500, detail="视频生成失败，请查看服务端日志")


@router.post("/api/taiji/video_to_gif")
def video_to_gif(req: VideoGifRequest):
    """视频转 GIF"""
    engine = _get_engine()
    try:
        input_str = f"{req.video_path} | {req.fps}" if req.fps else req.video_path
        result = engine._execute_multimodal_tool("video_to_gif", {"input": input_str})
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"视频转GIF失败: {e}")
        raise HTTPException(status_code=500, detail="视频转GIF失败，请查看服务端日志")


# ======================== 多模态 Agent 任务 ========================

@router.post("/api/taiji/run")
async def multimodal_run(req: MultimodalRunRequest):
    """运行多模态 Agent 任务（流式返回）"""
    engine = _get_engine()

    async def event_generator():
        try:
            from taiji.agent_ext.tool_registry import registry
            # 用户提供了工具列表时用用户的，否则用全局 registry
            tool_reg = req.tools if req.tools else registry

            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(engine.run, req.task, tool_reg)

                # 分步 yield 步骤信息
                last_step_count = 0
                while not future.done():
                    await asyncio.sleep(0.1)
                    # 如果引擎支持中间结果，可以在这里 yield 进度
                    if hasattr(engine, 'last_result') and engine.last_result:
                        current_steps = len(engine.last_result.steps)
                        if current_steps > last_step_count:
                            for i in range(last_step_count, current_steps):
                                step = engine.last_result.steps[i]
                                step_output = {
                                    "type": "step",
                                    "step_index": i,
                                    "action": getattr(step, 'action', ''),
                                    "thought": getattr(step, 'thought', ''),
                                }
                                yield f"data: {json.dumps(step_output, ensure_ascii=False)}\n\n"
                            last_step_count = current_steps

                result = future.result()

                output = {
                    "type": "result",
                    "status": result.status,
                    "final_answer": result.final_answer,
                    "steps": len(result.steps),
                    "duration_ms": result.total_duration_ms,
                }
                yield f"data: {json.dumps(output, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"多模态任务失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'status': 'error', 'message': '任务执行失败，请查看服务端日志'}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/api/taiji/cancel")
def multimodal_cancel():
    """取消当前多模态任务"""
    engine = _get_engine()
    engine.cancel()
    return {"status": "ok", "message": "已发送取消信号"}


# ======================== 喂养引擎（吃饭）========================

@router.get("/api/taiji/feed/status")
def feed_status():
    """获取喂养引擎状态"""
    try:
        from taiji.life.feed_engine import get_feed_engine
        engine = get_feed_engine()
        return {"status": "ok", "data": engine.get_status(), "summary": engine.get_summary()}
    except Exception as e:
        logger.error(f"获取喂养状态失败: {e}")
        raise HTTPException(status_code=500, detail="获取喂养状态失败")


@router.post("/api/taiji/feed")
def feed_taiji():
    """让态极吃饭 — 自动从各来源收集数据"""
    try:
        from taiji.life.feed_engine import get_feed_engine
        engine = get_feed_engine()
        report = engine.feed(reason="manual")
        return {
            "status": "ok",
            "items_fed": report.items_fed,
            "items_rejected": report.items_rejected,
            "samples_generated": report.samples_generated,
            "avg_quality": report.avg_quality,
            "duration_seconds": report.duration_seconds,
            "recommendations": report.recommendations,
        }
    except Exception as e:
        logger.error(f"喂养失败: {e}")
        raise HTTPException(status_code=500, detail="喂养失败，请查看服务端日志")


@router.post("/api/taiji/feed/text")
def feed_text(request: dict):
    """直接喂态极一段文字（请求体：{text, source?, category?}）"""
    try:
        text = request.get("text", "")
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="text 不能为空")
        source = request.get("source", "manual")
        category = request.get("category", "knowledge")
        from taiji.life.feed_engine import get_feed_engine
        engine = get_feed_engine()
        item = engine.feed_text(text=text, source=source, category=category)
        if item:
            return {
                "status": "ok",
                "quality_score": item.quality_score,
                "sample_count": item.sample_count,
                "item_status": item.status,
            }
        return {"status": "ok", "message": "内容已跳过（重复或质量不达标）"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文字喂养失败: {e}")
        raise HTTPException(status_code=500, detail="文字喂养失败")


def _validate_workspace_path(file_path: str) -> str:
    """验证路径在工作空间内，防止路径穿越"""
    workspace = get_external_path("agent_workspace")
    abs_path = os.path.abspath(file_path)
    abs_workspace = os.path.abspath(workspace)
    if not abs_path.startswith(abs_workspace + os.sep) and abs_path != abs_workspace:
        # 也允许访问 data 目录
        data_dir = get_external_path("data")
        abs_data = os.path.abspath(data_dir)
        if not abs_path.startswith(abs_data + os.sep):
            raise HTTPException(status_code=403, detail="路径超出允许范围")
    return abs_path


@router.post("/api/taiji/feed/file")
def feed_file(request: dict):
    """喂态极吃一个文件（请求体：{file_path, category?}）"""
    try:
        file_path = request.get("file_path", "")
        if not file_path:
            raise HTTPException(status_code=400, detail="file_path 不能为空")
        safe_path = _validate_workspace_path(file_path)
        if not os.path.isfile(safe_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        category = request.get("category", "knowledge")
        from taiji.life.feed_engine import get_feed_engine
        engine = get_feed_engine()
        item = engine.feed_file(file_path=safe_path, category=category)
        if item:
            return {
                "status": "ok",
                "quality_score": item.quality_score,
                "sample_count": item.sample_count,
                "item_status": item.status,
            }
        return {"status": "ok", "message": "文件已跳过（重复或质量不达标）"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件喂养失败: {e}")
        raise HTTPException(status_code=500, detail="文件喂养失败")


@router.post("/api/taiji/feed/directory")
def feed_directory(request: dict):
    """喂态极吃一个目录下的所有文件（请求体：{dir_path, category?}）"""
    try:
        dir_path = request.get("dir_path", "")
        if not dir_path:
            raise HTTPException(status_code=400, detail="dir_path 不能为空")
        safe_path = _validate_workspace_path(dir_path)
        if not os.path.isdir(safe_path):
            raise HTTPException(status_code=404, detail="目录不存在")
        category = request.get("category", "code")
        from taiji.life.feed_engine import get_feed_engine
        engine = get_feed_engine()
        count = engine.feed_directory(dir_path=safe_path, category=category)
        return {"status": "ok", "files_fed": count}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"目录喂养失败: {e}")
        raise HTTPException(status_code=500, detail="目录喂养失败")


@router.get("/api/taiji/feed/plan")
def feed_plan():
    """获取进食计划 — 根据能力短板推荐吃什么"""
    try:
        from taiji.life.feed_engine import get_feed_engine
        engine = get_feed_engine()
        plan = engine.get_feed_plan()
        return {"status": "ok", "plan": plan}
    except Exception as e:
        logger.error(f"获取进食计划失败: {e}")
        raise HTTPException(status_code=500, detail="获取进食计划失败")


# ======================== 睡眠引擎 ========================

@router.get("/api/taiji/sleep/status")
def sleep_status():
    """获取睡眠引擎状态"""
    try:
        from taiji.life.sleep_engine import get_sleep_engine
        engine = get_sleep_engine()
        return {"status": "ok", "data": engine.get_status(), "summary": engine.get_summary()}
    except Exception as e:
        logger.error(f"获取睡眠状态失败: {e}")
        raise HTTPException(status_code=500, detail="获取睡眠状态失败")


@router.post("/api/taiji/sleep")
def sleep_taiji():
    """让态极睡觉"""
    try:
        from taiji.life.sleep_engine import get_sleep_engine
        engine = get_sleep_engine()
        report = engine.sleep(reason="manual")
        return {
            "status": "ok",
            "phases_completed": report.phases_completed,
            "training_samples_used": report.training_samples_used,
            "training_loss": report.training_loss,
            "health_status": report.health_status,
            "duration_seconds": report.duration_seconds,
        }
    except Exception as e:
        logger.error(f"睡眠失败: {e}")
        raise HTTPException(status_code=500, detail="睡眠失败，请查看服务端日志")


# ======================== 玩耍引擎（娱乐）========================

@router.get("/api/taiji/play/status")
def play_status():
    """获取玩耍引擎状态"""
    try:
        from taiji.life.play_engine import get_play_engine
        engine = get_play_engine()
        return {"status": "ok", "data": engine.get_status(), "summary": engine.get_summary()}
    except Exception as e:
        logger.error(f"获取玩耍状态失败: {e}")
        raise HTTPException(status_code=500, detail="获取玩耍状态失败")


@router.post("/api/taiji/play")
def play_taiji():
    """让态极玩耍 — 自由探索和创意实验"""
    try:
        from taiji.life.play_engine import get_play_engine
        engine = get_play_engine()
        report = engine.play(reason="manual")
        activities = []
        for a in report.activities:
            activities.append({
                "type": a.activity_type,
                "topic": a.topic,
                "content": a.content,
                "quality": round(a.quality_score, 2),
                "kept": a.kept,
            })
        return {
            "status": "ok",
            "activities": activities,
            "mood": report.mood,
            "traits_discovered": report.personality_traits_discovered,
            "duration_seconds": report.duration_seconds,
        }
    except Exception as e:
        logger.error(f"玩耍失败: {e}")
        raise HTTPException(status_code=500, detail="玩耍失败")


@router.get("/api/taiji/play/personality")
def play_personality():
    """获取态极的个性档案"""
    try:
        from taiji.life.play_engine import get_play_engine
        engine = get_play_engine()
        return {"status": "ok", "personality": engine.get_personality()}
    except Exception as e:
        logger.error(f"获取个性档案失败: {e}")
        raise HTTPException(status_code=500, detail="获取个性档案失败")


# ======================== 生命调度器 ========================

@router.get("/api/taiji/life/status")
def life_status():
    """获取生命状态（需求、状态、心跳数）"""
    try:
        from taiji.life.life_scheduler import get_life_scheduler
        scheduler = get_life_scheduler()
        return {"status": "ok", "data": scheduler.get_status(), "summary": scheduler.get_summary()}
    except Exception as e:
        logger.error(f"获取生命状态失败: {e}")
        raise HTTPException(status_code=500, detail="获取生命状态失败")


@router.post("/api/taiji/life/start")
def life_start():
    """启动生命（启动心跳循环）"""
    try:
        from taiji.life.life_scheduler import get_life_scheduler
        scheduler = get_life_scheduler()
        scheduler.start()
        return {"status": "ok", "message": "🌱 生命已启动"}
    except Exception as e:
        logger.error(f"启动生命失败: {e}")
        raise HTTPException(status_code=500, detail="启动生命失败")


@router.post("/api/taiji/life/stop")
def life_stop():
    """暂停生命"""
    try:
        from taiji.life.life_scheduler import get_life_scheduler
        scheduler = get_life_scheduler()
        scheduler.stop()
        return {"status": "ok", "message": "⏸️ 生命已暂停"}
    except Exception as e:
        logger.error(f"暂停生命失败: {e}")
        raise HTTPException(status_code=500, detail="暂停生命失败")


@router.post("/api/taiji/life/interact")
def life_interact(success: bool = True, topic: str = ""):
    """记录一次用户交互（影响需求状态）"""
    try:
        from taiji.life.life_scheduler import get_life_scheduler
        scheduler = get_life_scheduler()
        scheduler.record_interaction(success=success, topic=topic)
        return {"status": "ok", "needs": scheduler.needs.to_dict()}
    except Exception as e:
        logger.error(f"记录交互失败: {e}")
        raise HTTPException(status_code=500, detail="记录交互失败")


@router.post("/api/taiji/life/action/{action}")
def life_force_action(action: str):
    """强制执行某个生命活动（feed/sleep/play）"""
    if action not in ("feed", "sleep", "play"):
        raise HTTPException(status_code=400, detail="无效的操作，支持: feed, sleep, play")
    try:
        from taiji.life.life_scheduler import get_life_scheduler
        scheduler = get_life_scheduler()
        result = scheduler.force_action(action)
        return {"status": "ok", "result": result, "needs": scheduler.needs.to_dict()}
    except Exception as e:
        logger.error(f"执行操作失败: {e}")
        raise HTTPException(status_code=500, detail="执行操作失败")


@router.get("/api/taiji/self_mod/status")
def self_mod_status():
    """获取态极自修改引擎状态（自主发现和安装工具的能力）"""
    try:
        from taiji.agent_ext.self_modification import get_self_modification_engine
        engine = get_self_modification_engine()
        return {"status": "ok", **engine.get_status()}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/taiji/self_mod/discover")
async def self_mod_discover(req: dict):
    """态极自主搜索可安装的工具"""
    keyword = req.get("keyword", "").strip()
    if not keyword:
        raise HTTPException(status_code=400, detail="关键词不能为空")
    try:
        from taiji.agent_ext.self_modification import get_self_modification_engine
        engine = get_self_modification_engine()
        matches = engine._discovery.find_matching_tools(keyword)
        return {"status": "ok", "matches": matches, "count": len(matches)}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.post("/api/taiji/self_mod/toggle")
async def self_mod_toggle(req: dict):
    """启用/禁用态极自修改引擎"""
    enabled = req.get("enabled", True)
    try:
        from taiji.agent_ext.self_modification import get_self_modification_engine
        engine = get_self_modification_engine()
        if enabled:
            engine.enable()
        else:
            engine.disable()
        return {"status": "ok", "enabled": engine._enabled}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"status": "error", "message": "内部错误，请查看日志"}


@router.get("/api/taiji/life/timeline")
def life_timeline(hours: int = 24):
    """获取生命时间线"""
    try:
        from taiji.life.life_scheduler import get_life_scheduler
        scheduler = get_life_scheduler()
        timeline = scheduler.get_timeline(hours=hours)
        return {"status": "ok", "timeline": timeline, "hours": hours}
    except Exception as e:
        logger.error(f"获取时间线失败: {e}")
        raise HTTPException(status_code=500, detail="获取时间线失败")


def _load_dataset_files(file_names: list, _safe_put=None) -> tuple:
    """
    加载用户上传的数据集文件，自动区分 ReAct 和对话格式。
    支持 file_parser 的全部格式：JSONL/JSON/CSV/TXT/MD/PDF/DOCX/HTML/EPUB/
    Excel/PPTX/RTF/XML/PY/JS/图片OCR 等 30+ 种格式。

    Returns:
        (conv_data, react_data, file_stats) — 对话数据 + ReAct数据 + 每文件统计
    """
    # 查找文件：检查所有可能的数据目录
    data_dirs = [get_external_path("data")]
    import sys as _sys
    if getattr(_sys, 'frozen', False):
        _project_root = os.path.dirname(_sys.executable)
    else:
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _fallback = os.path.join(_project_root, "data")
    if _fallback not in data_dirs:
        data_dirs.append(_fallback)
    conv_data = []
    react_data = []
    file_stats = []

    for idx, fname in enumerate(file_names):
        # 在所有数据目录中查找文件
        fpath = None
        for data_dir in data_dirs:
            candidate = os.path.abspath(os.path.join(data_dir, fname))
            if not candidate.startswith(os.path.abspath(data_dir) + os.sep):
                continue
            if os.path.exists(candidate):
                fpath = candidate
                break
        if not fpath:
            logger.warning(f"数据集文件不存在: {fname}")
            file_stats.append({"name": fname, "samples": 0, "error": "文件不存在"})
            continue

        file_conv_before = len(conv_data)
        file_react_before = len(react_data)
        ext = os.path.splitext(fname)[1].lower()

        try:
            if ext in (".jsonl", ".json"):
                # JSONL/JSON：自动区分 ReAct 和对话格式
                _load_json_file(fpath, conv_data, react_data)
            else:
                # 所有其他格式统一走 file_parser（含 PDF OCR、DOCX、Excel、图片 OCR 等）
                _load_with_file_parser(fpath, ext, conv_data, _safe_put)

            file_count = (len(conv_data) - file_conv_before) + (len(react_data) - file_react_before)
            file_stats.append({"name": fname, "samples": file_count, "error": None})
            logger.info(f"从 {fname} 加载了 {file_count} 条数据")

            if _safe_put:
                _safe_put(json.dumps({
                    "type": "progress",
                    "fraction": round(0.02 + (idx + 1) / len(file_names) * 0.03, 4),
                    "desc": f"📂 加载数据集 {idx+1}/{len(file_names)}: {fname} ({file_count} 条)",
                    "loss": None, "step": 0,
                }, ensure_ascii=False))

        except Exception as e:
            logger.warning(f"无法解析数据集文件 {fname}: {e}")
            file_stats.append({"name": fname, "samples": 0, "error": str(e)})

    logger.info(f"共加载 {len(conv_data)} 条对话 + {len(react_data)} 条 ReAct 数据用于微调")
    return conv_data, react_data, file_stats


def _load_json_file(fpath: str, conv_data: list, react_data: list = None):
    """加载 JSONL/JSON 文件，自动识别 ReAct 和对话格式"""
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return

    def _process_item(item):
        """处理单条数据，自动分类"""
        # ReAct 格式：task + steps
        if "task" in item and "steps" in item:
            if react_data is not None:
                react_data.append(item)
            return
        # 对话格式：messages
        if "messages" in item:
            conv_data.append(item)
            return
        # instruction/output 格式 → 转为对话
        conv = _convert_to_conversation(item)
        if conv:
            conv_data.append(conv)

    if content.startswith("{") and "\n" in content:
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                _process_item(json.loads(line))
            except json.JSONDecodeError:
                pass
    else:
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    _process_item(item)
        except json.JSONDecodeError:
            pass


def _load_with_file_parser(fpath: str, ext: str, conv_data: list, _safe_put=None):
    """使用 file_parser 统一解析非 JSON 文件，按段落拆分为对话样本"""
    from taiji.tools.file_parser import parse_file_to_text, IMAGE_EXTENSIONS

    def progress_cb(_current, _total, message):
        if _safe_put and message:
            _safe_put(json.dumps({
                "type": "progress",
                "fraction": 0.02,
                "desc": f"📄 解析中: {message}",
                "loss": None, "step": 0,
            }, ensure_ascii=False))

    text = parse_file_to_text(fpath, progress_callback=progress_cb)
    if not text or len(text.strip()) < 10:
        return

    # 图片 OCR：整张图作为一个样本
    if ext in IMAGE_EXTENSIONS:
        conv_data.append({
            "messages": [
                {"role": "user", "content": "请识别并描述这张图片中的文字内容"},
                {"role": "assistant", "content": text.strip()[:2000]},
            ]
        })
        return

    # 其他格式：按段落拆分
    paragraphs = _split_paragraphs(text)
    for para in paragraphs:
        if len(para) > 10:
            conv_data.append({
                "messages": [
                    {"role": "user", "content": f"请学习以下内容并总结要点：\n{para[:500]}"},
                    {"role": "assistant", "content": para[:1000]},
                ]
            })


def _split_paragraphs(text: str) -> list:
    """智能分段：双换行优先，单换行次之，长文本兜底按字数切"""
    # 先按双换行分
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) > 1:
        return paras
    # 单换行
    paras = [p.strip() for p in text.split("\n") if p.strip() and len(p.strip()) > 20]
    if len(paras) > 1:
        return paras
    # 长文本按 500 字切
    if len(text) > 500:
        return [text[i:i+500] for i in range(0, len(text), 500)]
    return [text]


def _convert_to_conversation(item: dict) -> dict:
    """将多种数据格式统一转换为对话格式"""
    # 已经是 messages 格式
    if "messages" in item:
        return item
    # instruction/output 格式
    instruction = item.get("instruction") or item.get("question") or item.get("input") or ""
    output = item.get("output") or item.get("answer") or item.get("response") or ""
    if instruction and output:
        return {
            "messages": [
                {"role": "user", "content": str(instruction)},
                {"role": "assistant", "content": str(output)},
            ]
        }
    return None


# ======================== 态极原生微调 ========================

@router.post("/api/taiji/train")
async def taiji_train(request: dict):
    """
    态极原生模型微调（SSE 流式返回进度）

    使用 ModelSelfTrainer.finetune() 进行 ReAct 工具调用微调。
    仅在加载态极 ModelSelf 模型时可用。

    请求体参数:
        num_epochs: int = 5
        batch_size: int = 4
        learning_rate: float = 1e-4
        max_length: int = 512
        save_steps: int = 50
        log_steps: int = 5
        keep_checkpoints: int = 3  # 保留最近 N 个中间 checkpoint + best
    """
    if not _is_available():
        raise HTTPException(status_code=404, detail="接口不存在")

    # 检查是否已有训练在运行
    if app_state.is_training:
        raise HTTPException(status_code=400, detail="当前已有训练任务在运行")

    from taiji.architecture import ModelSelf
    model = app_state.model
    if not isinstance(model, ModelSelf):
        raise HTTPException(status_code=400, detail="当前模型不是态极原生模型，无法使用此接口")

    tokenizer = app_state.tokenizer
    if tokenizer is None:
        raise HTTPException(status_code=500, detail="分词器未加载")

    # 解析参数
    num_epochs = request.get("num_epochs", 5)
    batch_size = request.get("batch_size", 4)
    learning_rate = request.get("learning_rate", 1e-4)
    max_length = request.get("max_length", 512)
    save_steps = request.get("save_steps", 50)
    log_steps = request.get("log_steps", 5)
    keep_checkpoints = request.get("keep_checkpoints", 3)
    dataset_files = request.get("dataset_files", [])  # 用户上传的数据集文件名列表

    async def event_generator():
        log_queue = queue.Queue(maxsize=256)

        def _safe_put(msg, timeout=5.0):
            try:
                log_queue.put(msg, timeout=timeout)
            except queue.Full:
                pass

        def train_worker():
            training_completed = False
            nonlocal batch_size, max_length
            try:
                app_state.try_start_training()
                # 清除停止标志
                from api.training.taiji_train import clear_stop_request, is_stop_requested
                clear_stop_request()

                from taiji.train.trainer import ModelSelfTrainer, build_dataset

                # 确定设备
                device = "cpu"
                try:
                    device = str(next(model.parameters()).device)
                    if device.startswith("cuda"):
                        device = "cuda"
                    elif device.startswith("mps"):
                        device = "mps"
                except Exception:
                    pass

                # 硬件自适应：根据 VRAM/RAM 自动调整参数
                hw_info = {}
                try:
                    from taiji.core.hardware import analyze_hardware
                    hw_info = analyze_hardware() or {}
                except Exception:
                    pass
                ram_gb = getattr(hw_info, "available_memory_gb", 8) or 8
                vram_gb = getattr(hw_info, "vram_gb", 0) or 0
                if vram_gb >= 8:
                    batch_size = max(batch_size, 8)
                    max_length = min(max_length, 1024)
                elif vram_gb >= 4:
                    batch_size = max(batch_size, 4)
                elif ram_gb >= 16:
                    batch_size = max(batch_size, 4)
                else:
                    batch_size = min(batch_size, 2)

                _safe_put(json.dumps({
                    "type": "hardware_diag",
                    "device": device,
                    "ram_gb": ram_gb,
                    "vram_gb": vram_gb,
                    "batch_size": batch_size,
                    "max_length": max_length,
                }, ensure_ascii=False))

                # 线程优先级（非关键，失败不影响训练）
                try:
                    from api.training.stream import _apply_thread_priority
                    _apply_thread_priority(hw_info)
                except Exception:
                    pass

                # 构建数据集
                _safe_put(json.dumps({
                    "type": "progress",
                    "fraction": 0.0,
                    "desc": "📦 正在构建训练数据集...",
                    "loss": None, "step": 0,
                }, ensure_ascii=False))

                # 自动清洗训练数据（去重、过滤模板、工具名对齐）
                try:
                    from taiji.train.data_cleaner import clean_training_data
                    _base_dir = os.path.dirname(os.path.abspath(__file__))
                    training_data_dir = os.path.join(_base_dir, "..", "taiji", "training_data")
                    if os.path.exists(training_data_dir):
                        _safe_put(json.dumps({
                            "type": "progress",
                            "fraction": 0.01,
                            "desc": "🧹 清洗训练数据（去重、过滤模板）...",
                            "loss": None, "step": 0,
                        }, ensure_ascii=False))
                        clean_stats = clean_training_data(training_data_dir)
                        _safe_put(json.dumps({
                            "type": "progress",
                            "fraction": 0.03,
                            "desc": f"✅ 数据清洗完成: {clean_stats['total_input']} → {clean_stats['total_output']} 条",
                            "loss": None, "step": 0,
                            "clean_stats": clean_stats,
                        }, ensure_ascii=False))
                except Exception as e:
                    logger.warning(f"数据清洗跳过（非关键）: {e}")

                # 加载用户上传的数据集（支持 JSONL/JSON/TXT/PDF，自动区分 ReAct 和对话）
                extra_conv_data = None
                extra_react_data = None
                file_stats = []
                if dataset_files:
                    extra_conv_data, extra_react_data, file_stats = _load_dataset_files(dataset_files, _safe_put=_safe_put)

                dataset = build_dataset(
                    tokenizer,
                    extra_react_data=extra_react_data,
                    extra_conv_data=extra_conv_data,
                    max_length=max_length,
                )

                if len(dataset) == 0:
                    _safe_put(json.dumps({
                        "type": "error",
                        "message": "训练数据为空，请先使用态极对话积累数据",
                    }, ensure_ascii=False))
                    _safe_put("[DONE]")
                    return

                _safe_put(json.dumps({
                    "type": "progress",
                    "fraction": 0.05,
                    "desc": f"✅ 数据集构建完成: {len(dataset)} 条样本",
                    "loss": None, "step": 0,
                    "dataset_info": {
                        "total_samples": len(dataset),
                        "files": file_stats,
                        "user_data": len(extra_conv_data) if extra_conv_data else 0,
                        "seed_data": len(dataset) - (len(extra_conv_data) if extra_conv_data else 0),
                    },
                }, ensure_ascii=False))

                # 创建训练器
                trainer = ModelSelfTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    learning_rate=learning_rate,
                )
                # 注册到 app_state，供暂停/停止 API 使用
                app_state._trainer_ref = trainer

                # 微调 checkpoint 保存到模型目录下（和模型绑定，打包不影响）
                model_path = getattr(app_state, "_loaded_model_name", "") or ""
                if model_path and os.path.isdir(model_path):
                    model_dir = model_path if os.path.exists(os.path.join(model_path, "config.json")) else os.path.dirname(model_path)
                    save_dir = os.path.join(model_dir, "checkpoints")
                else:
                    save_dir = get_external_path(os.path.join("taiji_checkpoints", "finetune"))
                os.makedirs(save_dir, exist_ok=True)

                # 执行微调（generator 模式）
                for fraction, desc, loss_history, metrics in trainer.finetune(
                    dataset=dataset,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    save_dir=save_dir,
                    save_steps=save_steps,
                    log_steps=log_steps,
                    device=device,
                ):
                    # 检查停止请求
                    if is_stop_requested():
                        trainer.stop()
                        _safe_put(json.dumps({
                            "type": "completed",
                            "message": "⏹ 训练已停止",
                            "total_steps": trainer.global_step,
                            "best_loss": trainer.best_loss,
                        }, ensure_ascii=False))
                        _safe_put("[DONE]")
                        return

                    # SSE 事件：透传 trainer 返回的所有 metrics
                    evt = {
                        "type": "progress",
                        "fraction": round(fraction, 4),
                        "desc": desc,
                    }
                    evt.update(metrics)
                    _safe_put(json.dumps(evt, ensure_ascii=False))

                # 训练完成后清理旧 checkpoint
                _cleanup_checkpoints(save_dir, keep=keep_checkpoints)
                training_completed = True

                _safe_put(json.dumps({
                    "type": "completed",
                    "message": f"✅ 态极微调完成！最终 Loss: {loss_history[-1]:.4f}" if loss_history else "✅ 态极微调完成！",
                    "total_steps": trainer.global_step,
                    "best_loss": trainer.best_loss,
                }, ensure_ascii=False))
                _safe_put("[DONE]")

            except Exception as e:
                logger.error(f"态极微调异常: {e}")
                _safe_put(json.dumps({
                    "type": "error",
                    "message": f"态极微调出错: {e}",
                }, ensure_ascii=False))
                _safe_put("[DONE]")
            finally:
                app_state.finish_training()
                model.eval()
                # 训练完成 → 热切换到微调后的模型；中止/失败 → 保持原模型
                if training_completed:
                    try:
                        best_path = os.path.join(save_dir, "best")
                        if os.path.exists(os.path.join(best_path, "config.json")):
                            from taiji.loader import load_model as load_taiji_model
                            new_model, new_tokenizer = load_taiji_model(best_path, device=device)
                            app_state.update_model(new_model, new_tokenizer, None, best_path)
                            logger.info("训练完成，已热切换到微调后的模型")
                    except Exception as e:
                        logger.warning(f"热切换模型失败（原模型仍可用）: {e}")

        t = threading.Thread(target=train_worker, daemon=True)
        t.start()
        app_state.register_background_task(t)

        # SSE 心跳循环
        heartbeat_counter = 0
        while True:
            try:
                has_message = False
                while not log_queue.empty():
                    msg = log_queue.get_nowait()
                    if msg == "[DONE]":
                        yield "data: [DONE]\n\n"
                        return
                    yield f"data: {msg}\n\n"
                    has_message = True
                    heartbeat_counter = 0

                if not has_message:
                    heartbeat_counter += 1
                    if heartbeat_counter >= 50:
                        yield ": heartbeat\n\n"
                        heartbeat_counter = 0

                await asyncio.sleep(0.1)
            except (GeneratorExit, RuntimeError, Exception):
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ======================== 模型信息查询 ========================

@router.get("/api/taiji/model/info")
def taiji_model_info():
    """获取当前态极模型的详细信息"""
    if not _is_available():
        raise HTTPException(status_code=404, detail="接口不存在")

    from taiji.architecture import ModelSelf
    model = app_state.model
    if not isinstance(model, ModelSelf):
        return {"status": "not_taiji", "message": "当前模型不是态极原生模型"}

    params = model.get_num_parameters()
    config = model.config

    # 检测当前模型规模
    total_params = params["total"]
    if total_params >= 3e9:
        current_size = "7B"
    elif total_params >= 1e9:
        current_size = "3B"
    elif total_params >= 500e6:
        current_size = "1B"
    elif total_params >= 200e6:
        current_size = "350M"
    else:
        current_size = "125M"

    # 扫描已有 checkpoint（与微调保存路径保持一致）
    model_path = getattr(app_state, "_loaded_model_name", "") or ""
    if model_path and os.path.isdir(model_path):
        checkpoints_dir = os.path.dirname(model_path)
    else:
        checkpoints_dir = get_external_path(os.path.join("taiji_checkpoints", "finetune"))
    best_path = os.path.join(checkpoints_dir, "best")
    has_best = os.path.exists(os.path.join(best_path, "config.json"))
    step_dirs = []
    if os.path.exists(checkpoints_dir):
        for d in os.listdir(checkpoints_dir):
            if d.startswith("step_") and os.path.isdir(os.path.join(checkpoints_dir, d)):
                try:
                    step_num = int(d.split("_")[1])
                    step_dirs.append(step_num)
                except ValueError:
                    pass
    step_dirs.sort()

    return {
        "status": "active",
        "size": current_size,
        "parameters": params,
        "config": {
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "intermediate_size": config.intermediate_size,
            "vocab_size": config.vocab_size,
            "max_position_embeddings": config.max_position_embeddings,
        },
        "available_sizes": ["125m", "350m", "1b", "3b", "7b"],
        "checkpoints": {
            "has_best": has_best,
            "latest_step": step_dirs[-1] if step_dirs else None,
            "total_checkpoints": len(step_dirs),
            "steps": step_dirs[-10:] if len(step_dirs) > 10 else step_dirs,
        },
    }


@router.post("/api/taiji/checkpoints/cleanup")
def cleanup_checkpoints_api(keep: int = 3):
    """清理旧 checkpoint，保留 best 和最新的 N 个"""
    model_path = getattr(app_state, "_loaded_model_name", "") or ""
    if model_path and os.path.isdir(model_path):
        model_dir = model_path if os.path.exists(os.path.join(model_path, "config.json")) else os.path.dirname(model_path)
        save_dir = os.path.join(model_dir, "checkpoints")
    else:
        save_dir = get_external_path(os.path.join("taiji_checkpoints", "finetune"))
    if not os.path.exists(save_dir):
        return {"status": "ok", "message": "checkpoint 目录不存在", "deleted": 0}

    deleted = _cleanup_checkpoints(save_dir, keep=keep)
    return {
        "status": "ok",
        "message": f"已清理 {deleted} 个旧 checkpoint，保留 best + 最新 {keep} 个",
        "deleted": deleted,
    }


def _cleanup_checkpoints(save_dir: str, keep: int = 3) -> int:
    """
    清理旧 checkpoint，保留 best/ 和最新的 keep 个 step_* 目录。

    Returns:
        删除的 checkpoint 数量
    """
    import shutil

    if not os.path.exists(save_dir):
        return 0

    step_dirs = []
    for d in os.listdir(save_dir):
        if d.startswith("step_") and os.path.isdir(os.path.join(save_dir, d)):
            try:
                step_num = int(d.split("_")[1])
                step_dirs.append((step_num, os.path.join(save_dir, d)))
            except ValueError:
                pass

    step_dirs.sort(key=lambda x: x[0])

    if len(step_dirs) <= keep:
        return 0

    # 删除旧的（保留最后 keep 个）
    to_delete = step_dirs[:-keep]
    deleted = 0
    for step_num, dir_path in to_delete:
        try:
            shutil.rmtree(dir_path)
            deleted += 1
            logger.info(f"已删除旧 checkpoint: {dir_path}")
        except Exception as e:
            logger.warning(f"删除 checkpoint 失败 {dir_path}: {e}")

    return deleted
