"""Settings management API routes."""

import logging
import os

from fastapi import APIRouter, HTTPException

from taiji.core.app_state import app_state
from taiji.core.config import get_config
from taiji.core.memory_watchdog import force_memory_refresh, get_memory_status_dict
from taiji.services.settings_service import load_settings, save_settings, update_settings

logger = logging.getLogger("ApiServer.Settings")
router = APIRouter()


@router.get("/api/settings")
def get_all_settings():
    """Return all persisted settings."""
    try:
        return load_settings()
    except Exception as exc:
        logger.warning(f"Failed to read settings: {exc}")
        return {}


@router.post("/api/settings")
async def save_all_settings(req: dict):
    """Merge and persist settings."""
    try:
        update_settings(req)
        return {"status": "ok", "message": "Settings saved"}
    except Exception as exc:
        logger.warning(f"Failed to save settings: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {exc}")


@router.post("/api/settings/model")
async def set_model(req: dict):
    """Update the configured model path or name."""
    model_name = req.get("model_name", "")
    if not model_name.strip():
        raise HTTPException(status_code=400, detail="Model name cannot be empty")
    update_settings({"model_name": model_name.strip()})
    return {"status": "ok", "message": f"Model set to {model_name}; restart required"}


@router.post("/api/settings/device")
async def set_device(req: dict):
    """Update the configured inference device."""
    device = req.get("device", "auto")
    update_settings({"device": device})
    return {"status": "ok", "message": f"Device set to {device}; restart required"}


@router.post("/api/settings/quant")
async def set_quant(req: dict):
    """Update quantization settings."""
    load_in_4bit = req.get("load_in_4bit", False)
    load_in_8bit = req.get("load_in_8bit", False)
    update_settings({
        "load_in_4bit": load_in_4bit,
        "load_in_8bit": load_in_8bit,
    })
    quant_mode = "4-bit" if load_in_4bit else "8-bit" if load_in_8bit else "disabled"
    return {"status": "ok", "message": f"Quantization mode set to {quant_mode}; restart required"}


@router.post("/api/settings/gguf")
async def set_gguf_settings(req: dict):
    """Update GGUF-related settings."""
    updates = {key: req[key] for key in ("model_type", "gguf_path", "n_gpu_layers", "n_ctx") if key in req}
    if req.get("gguf_path"):
        updates["model_type"] = "gguf"
    update_settings(updates)
    return {"status": "ok", "message": "GGUF settings saved; restart required"}


@router.get("/api/settings/gguf_models")
async def list_gguf_models():
    """GGUF models not supported in native Taiji stack."""
    return {"models": [], "error": "原生态极不支持 GGUF 模型"}


@router.post("/api/settings/download_gguf")
async def download_gguf(req: dict):
    """GGUF download not supported in native Taiji stack."""
    raise HTTPException(status_code=400, detail="原生态极不支持 GGUF 模型下载")


@router.get("/api/system/current_model")
def get_current_model():
    """Return effective and pending model configuration."""
    try:
        config = get_config()
        model_type = getattr(config, "model_type", None)
        gguf_path = getattr(config, "gguf_path", "")
        model_name = getattr(config, "model_name", "")
        n_gpu_layers = getattr(config, "n_gpu_layers", 0)
        device = getattr(config, "device", "auto")
        load_in_4bit = getattr(config, "load_in_4bit", False)
        load_in_8bit = getattr(config, "load_in_8bit", False)

        actual_loaded_name = getattr(app_state, "_loaded_model_name", "") or ""
        if actual_loaded_name:
            effective_path = actual_loaded_name
            effective_type = "self"  # 原生态极 — 单一模型类型
        else:
            effective_path = model_name
            effective_type = "self"

        saved = load_settings()
        pending_model_type = saved.get("model_type", "")
        pending_gguf_path = saved.get("gguf_path", "")
        pending_model_name = saved.get("model_name", "")
        pending_path = pending_gguf_path or pending_model_name

        is_pending = False
        if pending_model_type and pending_model_type != (model_type or ""):
            if not (pending_model_type == "gguf" and effective_type == "gguf"):
                if not (pending_model_type == "self" and effective_type == "self"):
                    is_pending = True
        if pending_path and pending_path != effective_path:
            is_pending = True
        if is_pending and app_state.startup_complete and effective_path and pending_path == effective_path:
            is_pending = False

        pending_settings = {
            "has_pending": is_pending,
            "pending_model_type": pending_model_type or effective_type,
            "pending_model_path": pending_path or effective_path,
            "pending_n_gpu_layers": saved.get("n_gpu_layers", n_gpu_layers),
            "pending_n_ctx": saved.get("n_ctx", getattr(config, "n_ctx", 2048)),
            "needs_restart": is_pending,
        }

        return {
            "status": "ok",
            "model_type": effective_type,
            "model_path": effective_path,
            "model_name": model_name,
            "gguf_path": gguf_path,
            "device": device,
            "n_gpu_layers": n_gpu_layers,
            "load_in_4bit": load_in_4bit,
            "load_in_8bit": load_in_8bit,
            "loaded": bool(effective_path and app_state.startup_complete),
            "pending_settings": pending_settings,
        }
    except Exception as exc:
        logger.warning(f"Failed to get current model info: {exc}")
        logger.error(f"Memory status failed: {exc}")
        return {"status": "error", "message": "内部错误，请查看日志", "loaded": False}


@router.get("/api/system/memory")
def get_memory_status():
    """Return the current memory status."""
    return get_memory_status_dict()


@router.post("/api/system/memory/refresh")
def refresh_memory_status():
    """Force-refresh memory status."""
    force_memory_refresh()
    return get_memory_status_dict()
