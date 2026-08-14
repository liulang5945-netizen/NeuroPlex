"""
Cortex 神经元架构 API 路由（内部代号：态极）。
所有端点基于 Cortex 神经元架构认知主体。
"""
import json
import logging
import os
import time

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Union

from taiji.core.app_state import app_state
from taiji.core.utils import get_external_path

logger = logging.getLogger("ApiServer.Taiji")
router = APIRouter()


def _is_available() -> bool:
    """检查 Cortex 神经元架构是否可用。"""
    return app_state.is_taiji()


def _is_cortex(model) -> bool:
    """P2-6: 判断当前模型是否为 Cortex（新认知主体）。

    使用类型名判断避免硬导入 Cortex（可能在某些启动路径下未加载）。
    """
    if model is None:
        return False
    return type(model).__name__ == 'Cortex'


def _cortex_model_info(cortex) -> dict:
    """P2-6: 返回 Cortex 神经元架构信息。"""
    neurons = getattr(cortex, 'neurons', {}) or {}
    neuron_specs = []
    total_params = 0
    for nid, neuron in neurons.items():
        try:
            n_params = sum(p.numel() for p in neuron.parameters())
            total_params += n_params
            cfg = getattr(neuron, 'config', None)
            spec = getattr(cfg, 'spec', 'unknown') if cfg else 'unknown'
            n_type = getattr(cfg, 'neuron_type', 'unknown') if cfg else 'unknown'
            neuron_specs.append({
                "id": nid,
                "spec": spec,
                "neuron_type": n_type,
                "params": n_params,
            })
        except Exception:
            continue

    field_dim = getattr(getattr(cortex, 'field', None), 'dim', None)
    is_fallback = any(
        s.get('spec') == 'general-fallback' for s in neuron_specs
    )

    # 多模态 codec 状态
    modalities = []
    hub = getattr(cortex, '_tokenizer_hub', None)
    if hub is not None:
        for mod in hub.list_modalities():
            codec = hub.modal_encoders.get(mod)
            if codec is not None:
                vocab = codec.vocab_size() if hasattr(codec, 'vocab_size') else 0
                # 检查 checkpoint 是否存在
                ckpt_map = {"image": "data/vqvae/vqvae_latest.pt",
                           "audio": "data/encodec/encodec_latest.pt",
                           "video": "data/video/video_latest.pt"}
                ckpt_path = ckpt_map.get(mod, "")
                modalities.append({
                    "modality": mod,
                    "vocab_size": vocab,
                    "trained": os.path.isfile(ckpt_path),
                    "checkpoint": ckpt_path if os.path.isfile(ckpt_path) else None,
                })

    return {
        "status": "active",
        "architecture": "cortex",
        "neuron_count": len(neurons),
        "field_dim": field_dim,
        "total_params": total_params,
        "is_fallback_mode": is_fallback,
        "neurons": neuron_specs,
        "max_rounds": getattr(cortex, 'max_rounds', None),
        "modalities": modalities,
        "message": (
            "单神经元 fallback 模式（未蒸馏）" if is_fallback
            else f"Cortex 已加载 {len(neurons)} 个神经元"
        ),
    }


# ======================== 状态查询 ========================

@router.get("/api/taiji/status")
def taiji_status():
    """获取 Cortex 神经元架构状态。"""
    if not _is_available():
        raise HTTPException(status_code=404, detail="接口不存在")
    model = app_state.model
    return _cortex_model_info(model)


@router.get("/api/taiji/tools")
def taiji_tools():
    """获取可用的多模态工具列表（旧 TaijiMultimodalEngine 已移除）。"""
    raise HTTPException(status_code=404, detail="接口不存在（请使用 /api/taiji/cortex/generate）")


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


@router.post("/api/taiji/feed/multimodal")
def feed_multimodal(request: dict):
    """喂态极多模态资料并触发小睡消化。

    请求体：{file_path, modality, nap?}
    - file_path: 已上传的文件路径（通过 /api/taiji/upload 上传）
    - modality: "image" / "audio" / "video"
    - nap: 是否触发小睡消化（默认 true）

    流程：文件 → codec encode → 训练样本入队 → (可选)nap 睡眠训练
    """
    try:
        file_path = request.get("file_path", "")
        modality = request.get("modality", "")
        trigger_nap = request.get("nap", True)

        if not file_path:
            raise HTTPException(status_code=400, detail="file_path 不能为空")
        if modality not in ("image", "audio", "video"):
            raise HTTPException(status_code=400, detail="modality 必须是 image/audio/video")

        safe_path = _validate_workspace_path(file_path)
        if not os.path.isfile(safe_path):
            raise HTTPException(status_code=404, detail="文件不存在")

        # 从 cortex 获取 tokenizer_hub
        model = app_state.model
        if model is None or not _is_cortex(model):
            raise HTTPException(status_code=503, detail="Cortex 未加载，无法处理多模态资料")
        hub = getattr(model, "_tokenizer_hub", None)
        if hub is None:
            raise HTTPException(status_code=503, detail="TokenizerHub 未初始化")
        if modality not in hub.list_modalities():
            raise HTTPException(
                status_code=503,
                detail=f"模态 '{modality}' 的 codec 未注册（已注册: {hub.list_modalities()})",
            )

        # 喂养
        from taiji.life.feed_engine import get_feed_engine
        engine = get_feed_engine()
        item = engine.feed_multimodal(modality, file_path=safe_path, tokenizer_hub=hub)
        if not item:
            return {"status": "ok", "message": "内容已跳过（重复或 codec 不可用）"}

        result = {
            "status": "ok",
            "modality": modality,
            "quality_score": item.quality_score,
            "sample_count": item.sample_count,
        }

        # 触发小睡消化
        if trigger_nap:
            from taiji.life.sleep_engine import get_sleep_engine
            sleep_engine = get_sleep_engine()
            sleep_engine.set_brain_interfaces(cortex=model)
            report = sleep_engine.nap(duration_minutes=1)
            if report:
                result["nap"] = {
                    "phases": report.phases_completed,
                    "samples": report.training_samples_used,
                }
            else:
                result["nap"] = "skipped"

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"多模态喂养失败: {e}")
        raise HTTPException(status_code=500, detail="多模态喂养失败")


class CortexGenRequest(BaseModel):
    """Cortex 原生多模态生成请求。"""
    modality: str  # "image" / "audio" / "video"
    input_path: Optional[str] = None  # 参考文件路径（模仿生成）
    max_tokens: Optional[int] = 0  # 0=自动用 codec 网格大小
    temperature: Optional[float] = 1.0
    top_k: Optional[int] = 0
    seed: Optional[int] = None
    output_path: Optional[str] = None  # 保存目录


@router.post("/api/taiji/cortex/generate")
def cortex_generate(req: CortexGenRequest):
    """使用 Cortex 原生 codec pipeline 生成多模态内容。

    两种模式：
    1. 模仿生成：提供 input_path（参考文件）→ codec encode → ensemble 共振 → 生成
    2. 随机生成：不提供 input_path → 用随机 latent 作为种子 → 生成
    """
    try:
        model = app_state.model
        if model is None or not _is_cortex(model):
            raise HTTPException(status_code=503, detail="Cortex 未加载")
        hub = getattr(model, "_tokenizer_hub", None)
        if hub is None:
            raise HTTPException(status_code=503, detail="TokenizerHub 未初始化")

        modality = req.modality
        if modality not in ("image", "audio", "video"):
            raise HTTPException(status_code=400, detail="modality 必须是 image/audio/video")
        if modality not in hub.list_modalities():
            raise HTTPException(status_code=503, detail=f"模态 '{modality}' 的 codec 未注册")

        import torch
        if req.seed is not None:
            torch.manual_seed(req.seed)

        device = next(model.neurons[next(iter(model.neurons))].parameters()).device
        codec = hub.modal_encoders.get(modality)

        # 构造输入 latent features
        if req.input_path:
            # 模仿模式：从文件加载 → codec encode → latent features
            safe_path = _validate_workspace_path(req.input_path)
            if not os.path.isfile(safe_path):
                raise HTTPException(status_code=404, detail="参考文件不存在")
            from taiji.multimodal.io import load_image, load_audio, load_video
            if modality == "image":
                data = load_image(safe_path).to(device)
            elif modality == "audio":
                data = load_audio(safe_path).to(device)
            else:
                data = load_video(safe_path).to(device)
            x = data.unsqueeze(0).to(device)
        else:
            # 随机模式：合成随机数据 → codec encode
            if modality == "image":
                x = torch.rand(1, 3, 32, 32, device=device)
            elif modality == "audio":
                x = torch.rand(1, 1, 16000, device=device) * 0.5
            else:
                x = torch.rand(1, 3, 16, 32, 32, device=device)

        with torch.no_grad():
            z = codec.model.encoder(x)
            # 展平为 [B, L, D]
            if modality == "image":
                B, D, Hz, Wz = z.shape
                z_seq = z.permute(0, 2, 3, 1).contiguous().view(B, Hz * Wz, D)
            elif modality == "audio":
                B, D, Tz = z.shape
                z_seq = z.permute(0, 2, 1).contiguous().view(B, Tz, D)
            else:
                B, D, Tz, Hz, Wz = z.shape
                z_seq = z.permute(0, 2, 3, 4, 1).contiguous().view(B, Tz * Hz * Wz, D)

        # max_tokens 默认用 codec 实际网格大小（避免 decode 时 reshape 失败）
        # image: Hz*Wz（正方形网格）；audio/video: 同样用 z_seq 长度
        actual_max_tokens = req.max_tokens if req.max_tokens and req.max_tokens > 0 else z_seq.shape[1]

        # 生成
        generated_ids = model.generate_multimodal(
            {"modality": modality, "data": z_seq, "domain": "general"},
            max_tokens=actual_max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            modality=modality,
        )

        # 解码
        recon = hub.decode(generated_ids, modality=modality)

        # 保存
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = req.output_path or f"data/{modality}/generated"
        os.makedirs(out_dir, exist_ok=True)

        result = {
            "status": "ok",
            "modality": modality,
            "token_count": len(generated_ids),
            "token_range": [min(generated_ids), max(generated_ids)],
        }

        if modality == "image":
            from taiji.multimodal.io import save_image
            path = os.path.join(out_dir, f"cortex_{ts}.png")
            img = recon if recon.dim() == 3 else recon[0]
            save_image(img, path)
            result["file"] = path
        elif modality == "audio":
            from taiji.multimodal.io import save_audio
            path = os.path.join(out_dir, f"cortex_{ts}.wav")
            # audio decode 返回 1D [samples]，不要索引
            aud = recon if recon.dim() <= 1 else recon[0]
            save_audio(aud, path, sample_rate=16000)
            result["file"] = path
        elif modality == "video":
            from taiji.multimodal.io import save_video
            path = os.path.join(out_dir, f"cortex_{ts}.mp4")
            vid = recon if recon.dim() == 4 else recon[0]
            # video decode 返回 [C, T, H, W]，save_video 需要 [T, C, H, W]
            if vid.dim() == 4 and vid.shape[0] == 3:
                vid = vid.permute(1, 0, 2, 3)
            save_video(vid, path, fps=8, fallback_png=True)
            result["file"] = path

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cortex 多模态生成失败: {e}")
        raise HTTPException(status_code=500, detail="Cortex 多模态生成失败")


class CortexChatRequest(BaseModel):
    """Cortex 文本对话请求。"""
    prompt: str
    # 2026-08-04：默认值改为保守参数（验证过短问答质量最佳）
    # 原 256/0.8/50 在 5 神经元异构 ensemble 上长序列生成易崩坏
    max_tokens: Optional[int] = 60
    temperature: Optional[float] = 0.55
    top_k: Optional[int] = 15
    domain: Optional[str] = None  # "zh"/"en"/"code"/"math"/"general"，None=自动推断


@router.post("/api/taiji/cortex/chat")
def cortex_chat(req: CortexChatRequest):
    """使用 Cortex 神经元架构进行文本对话。

    走 P7 路径：域专用 tokenizer encode → shared_embedding → ensemble 共振 → 域 lm_head decode。
    domain=None 时自动推断域（code/math/zh/en/general）。
    """
    try:
        model = app_state.model
        if model is None or not _is_cortex(model):
            raise HTTPException(status_code=503, detail="Cortex 未加载")
        if not req.prompt.strip():
            raise HTTPException(status_code=400, detail="prompt 不能为空")

        # 调用 Cortex.generate（内部会获取 train_lock，与 sleep 训练互斥）
        text = model.generate(
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            domain=req.domain,
        )

        # R10（REMEDIATION_PLAN 2026-08-14）：生产接线——对话成功后把共振场
        # 快照沉淀为记忆候选（睡眠 Phase 1.5 统一固化 + WriteGate 可学习过滤）。
        # 此前 record_field_memory 仅 verify 脚本调用 → 生产记忆固化是空操作
        # （审计发现）。读侧（auto_memory 检索注入生成）已由 sleep_engine
        # set_brain_interfaces → cortex.set_field_memory 接通。
        try:
            from taiji.life.sleep_engine import get_sleep_engine
            engine = get_sleep_engine()
            fstate = model.get_last_field_state()
            if engine is not None and fstate is not None:
                label = (req.prompt.strip() or "chat")[:40]
                engine.record_field_memory(fstate, label, text=text)
        except Exception as e:
            logger.debug(f"场记忆记录失败（不影响对话响应）: {e}")

        # 返回推断的域（用于前端展示路由信息）
        inferred_domain = req.domain or model._infer_domain(req.prompt)

        return {
            "status": "ok",
            "response": text,
            "domain": inferred_domain,
            "neurons": list(model.neurons.keys()),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cortex 文本对话失败: {e}")
        raise HTTPException(status_code=500, detail="Cortex 文本对话失败")


class TaskChainStageRequest(BaseModel):
    """C26 增量八：任务链阶段（task-set）请求体。"""
    prompt: str
    mode: str = "continuous"
    domain: Optional[str] = None
    active_nids: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    quality_gate: bool = True
    record_memory: bool = False
    memory_label: Optional[str] = None


class CortexTaskChainRequest(BaseModel):
    """C26 增量八：多阶段任务模式链 v2 请求体。"""
    stages: List[TaskChainStageRequest]
    max_tokens_per_stage: Optional[int] = 32


@router.post("/api/taiji/cortex/task_chain")
def cortex_task_chain(req: CortexTaskChainRequest):
    """多阶段任务模式链 v2（TaskSet 序列，人脑任务集切换，C26 增量八）。

    每阶段 = task-set（prompt/mode/domain/active_nids + 质量门），阶段间
    三重传递（文本 prev + 场状态 seed_memories + 记忆写入）。生产入口。
    """
    try:
        model = app_state.model
        if model is None or not _is_cortex(model):
            raise HTTPException(status_code=503, detail="Cortex 未加载")
        if not req.stages:
            raise HTTPException(status_code=400, detail="stages 不能为空")
        from taiji.brain.cortex import TaskSet
        stages = [TaskSet(
            prompt=s.prompt,
            mode=s.mode,
            domain=s.domain,
            active_nids=s.active_nids,
            max_tokens=s.max_tokens,
            temperature=s.temperature,
            quality_gate=s.quality_gate,
            record_memory=s.record_memory,
            memory_label=s.memory_label,
        ) for s in req.stages]
        result = model.generate_task_chain(
            stages,
            max_tokens_per_stage=req.max_tokens_per_stage or 32,
        )
        return {
            "status": "ok",
            "outputs": result["outputs"],
            "gates": result["gates"],
            "neurons": list(model.neurons.keys()),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cortex 任务链失败: {e}")
        raise HTTPException(status_code=500, detail="Cortex 任务链失败")


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


# ======================== 模型信息查询 ========================

@router.get("/api/taiji/model/info")
def taiji_model_info():
    """获取当前 Cortex 神经元架构信息。"""
    if not _is_available():
        raise HTTPException(status_code=404, detail="接口不存在")
    model = app_state.model
    # Cortex 是唯一认知主体；非 Cortex 视为未加载
    if not _is_cortex(model):
        raise HTTPException(status_code=503, detail="Cortex 未加载")
    return _cortex_model_info(model)


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
