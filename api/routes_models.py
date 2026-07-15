"""
原生态极 — 模型路由（精简版）
不需要模型市场/下载/GGUF，仅提供基本模型信息和列表
"""
import logging
import os

from fastapi import APIRouter

from taiji.core.app_state import app_state

logger = logging.getLogger("ApiServer.Models")
router = APIRouter()


# ======================== 原生模型信息 ========================

@router.get("/api/models/installed")
def list_installed_models():
    """列出已安装的原生 Taiji 模型。"""
    loaded = getattr(app_state, "_loaded_model_name", "") or ""
    models = [{"name": loaded, "type": "self", "status": "loaded"}] if loaded else []
    return {"models": models}


@router.get("/api/models/list")
def list_available_models():
    """原生态极不依赖外部模型市场。"""
    return {"models": [], "message": "原生态极使用本地 ModelSelf 模型，无需模型市场"}


@router.get("/api/models/downloaded")
def list_downloaded_models():
    """列出本地模型文件。"""
    loaded = getattr(app_state, "_loaded_model_name", "") or ""
    return {"models": [{"name": loaded, "type": "self"}] if loaded else []}


@router.get("/api/model/gguf_quants")
def get_gguf_quants():
    """GGUF 量化选项（原生态极不支持 GGUF）。"""
    return {"options": [], "message": "原生态极不支持 GGUF 量化"}


@router.get("/api/models/recommend")
def recommend_models():
    """推荐模型（原生态极使用 ModelSelf）。"""
    return {"models": [], "recommended": "ModelSelf（原生）", "message": "原生态极使用本地 ModelSelf 模型"}


@router.get("/api/models/tags")
def list_model_tags():
    return {"tags": []}


@router.get("/api/models/families")
def list_model_families():
    return {"families": []}


@router.get("/api/models/info")
def get_model_info():
    return {"info": {"type": "self", "message": "原生态极 ModelSelf"}}


# 外部模型下载/管理端点 — 返回不支持
@router.post("/api/models/download_hf")
def download_hf_model():
    return {"status": "error", "message": "原生态极不支持 HuggingFace 模型下载"}


@router.post("/api/models/download")
def download_model():
    return {"status": "error", "message": "原生态极不支持外部模型下载"}


@router.post("/api/models/download_cancel")
def cancel_download():
    return {"status": "ok"}


@router.post("/api/models/download_pause")
def pause_download():
    return {"status": "ok"}


@router.post("/api/models/download_resume")
def resume_download():
    return {"status": "error", "message": "原生态极不支持外部模型下载"}


@router.get("/api/models/download_progress")
def get_download_progress():
    return {"active": False}


@router.delete("/api/models/installed")
def delete_installed_model():
    return {"status": "error", "message": "原生态极暂不支持通过 API 删除模型"}


@router.post("/api/models/delete")
def delete_model():
    return {"status": "error", "message": "原生态极暂不支持通过 API 删除模型"}


@router.post("/api/models/select")
def select_model():
    """选择模型（原生态极自动使用 ModelSelf）。"""
    return {"status": "ok", "model_type": "self", "message": "原生态极使用本地 ModelSelf 模型"}
