"""
Seed — 模型路由（精简版）
Cortex 神经元架构是唯一认知主体，不依赖外部模型市场/下载/GGUF。
"""
import logging

from fastapi import APIRouter

from neuroplex.core.app_state import app_state

logger = logging.getLogger("ApiServer.Models")
router = APIRouter()


# ======================== Cortex 模型信息 ========================

@router.get("/api/models/installed")
def list_installed_models():
    """列出已安装的 Cortex 神经元架构。"""
    loaded = getattr(app_state, "_loaded_model_name", "") or ""
    models = [{"name": loaded, "type": "cortex", "status": "loaded"}] if loaded else []
    return {"models": models}


@router.get("/api/models/list")
def list_available_models():
    """Seed使用 Cortex 神经元架构，不依赖外部模型市场。"""
    return {"models": [], "message": "Seed使用 Cortex 神经元架构，无需模型市场"}


@router.get("/api/models/downloaded")
def list_downloaded_models():
    """列出本地模型文件。"""
    loaded = getattr(app_state, "_loaded_model_name", "") or ""
    return {"models": [{"name": loaded, "type": "cortex"}] if loaded else []}


@router.get("/api/model/gguf_quants")
def get_gguf_quants():
    """GGUF 量化选项（Seed不支持 GGUF）。"""
    return {"options": [], "message": "Seed使用 Cortex 神经元架构，不支持 GGUF 量化"}


@router.get("/api/models/recommend")
def recommend_models():
    """推荐模型（Seed使用 Cortex）。"""
    return {"models": [], "recommended": "Cortex（神经元架构）", "message": "Seed使用 Cortex 神经元架构"}


@router.get("/api/models/tags")
def list_model_tags():
    return {"tags": []}


@router.get("/api/models/families")
def list_model_families():
    return {"families": []}


@router.get("/api/models/info")
def get_model_info():
    return {"info": {"type": "cortex", "message": "Seed Cortex 神经元架构"}}


# 外部模型下载/管理端点 — 返回不支持
@router.post("/api/models/download_hf")
def download_hf_model():
    return {"status": "error", "message": "Seed不支持 HuggingFace 模型下载"}


@router.post("/api/models/download")
def download_model():
    return {"status": "error", "message": "Seed不支持外部模型下载"}


@router.post("/api/models/download_cancel")
def cancel_download():
    return {"status": "ok"}


@router.post("/api/models/download_pause")
def pause_download():
    return {"status": "ok"}


@router.post("/api/models/download_resume")
def resume_download():
    return {"status": "error", "message": "Seed不支持外部模型下载"}


@router.get("/api/models/download_progress")
def get_download_progress():
    return {"active": False}


@router.delete("/api/models/installed")
def delete_installed_model():
    return {"status": "error", "message": "Seed暂不支持通过 API 删除模型"}


@router.post("/api/models/delete")
def delete_model():
    return {"status": "error", "message": "Seed暂不支持通过 API 删除模型"}


@router.post("/api/models/select")
def select_model():
    """选择模型（Seed自动使用 Cortex 神经元架构）。"""
    return {"status": "ok", "model_type": "cortex", "message": "Seed使用 Cortex 神经元架构"}
