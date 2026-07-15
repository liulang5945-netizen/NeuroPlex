"""Canonical model switch routes for runtime model lifecycle operations."""

from __future__ import annotations

import gc
import logging
import os
import threading
from typing import Any

from fastapi import APIRouter

from taiji.core.app_state import app_state
from taiji.core.config import TrainingConfig, save_config
from taiji.core.memory_watchdog import MemoryWatchdog, force_memory_refresh
from taiji.core.utils import get_external_path
from taiji.services.settings_service import load_settings, save_settings

logger = logging.getLogger("ApiServer.ModelSwitch")
router = APIRouter()

_switch_lock = threading.Lock()
_switch_thread: threading.Thread | None = None


@router.post("/api/system/reload_model")
def reload_model() -> dict[str, Any]:
    settings = load_settings()
    gguf_path = settings.get("gguf_path", "")
    model_name = settings.get("model_name", "")
    model_type = settings.get("model_type", "gguf" if gguf_path else "huggingface")
    return _do_switch_model(str(model_type), str(gguf_path), str(model_name))


@router.post("/api/system/switch_model")
def switch_model(req: dict[str, Any]) -> dict[str, Any]:
    global _switch_thread

    model_type = str(req.get("model_type", "") or "").lower()
    gguf_path = str(req.get("gguf_path", "") or "")
    model_name = str(req.get("model_name", "") or "")

    if not model_type:
        return {"status": "error", "message": "Missing model_type parameter"}

    if not _switch_lock.acquire(blocking=False):
        current = app_state.get_switch_status()
        return {
            "status": "switching_in_progress",
            "message": f"Model switch already in progress ({current.get('message') or 'loading'})",
        }

    try:
        current = app_state.get_switch_status()
        if current["status"] == "switching":
            _switch_lock.release()
            return {
                "status": "switching_in_progress",
                "message": f"Model switch already in progress ({current.get('message') or 'loading'})",
            }

        model_display = os.path.basename(gguf_path or model_name) or model_type
        app_state.update_switch_status("switching", f"Starting model switch: {model_display}")

        def _do_switch_async() -> None:
            try:
                result = _do_switch_model(model_type, gguf_path, model_name, async_mode=True)
                if result.get("status") == "ok":
                    app_state.update_switch_status("success", result.get("message", "Model switch complete"))
                else:
                    app_state.update_switch_status("error", "", result.get("message", "Model switch failed"))
            except Exception as exc:
                logger.exception("Async model switch failed")
                app_state.mark_startup_failed(str(exc))
                app_state.update_switch_status("error", "", f"Model switch failed: {exc}")
            finally:
                _switch_lock.release()

        _switch_thread = threading.Thread(target=_do_switch_async, daemon=True)
        _switch_thread.start()
        return {
            "status": "ok",
            "message": f"Starting model switch: {model_display}",
            "model_type": model_type,
        }
    except Exception as exc:
        _switch_lock.release()
        logger.error(f"Model switch start failed: {exc}")
        return {"status": "error", "message": "Failed to start model switch"}


@router.get("/api/system/switch_status")
def get_switch_status() -> dict[str, Any]:
    state = app_state.get_switch_status()
    return {
        "status": state["status"],
        "message": state["message"],
        "error": state["error"],
    }


@router.post("/api/system/pub_reset")
def force_reset_publishing() -> dict[str, Any]:
    result = app_state.force_reset_publishing()
    return {"status": "ok", **result}


def _do_switch_model(
    model_type: str,
    gguf_path: str,
    model_name: str,
    *,
    async_mode: bool = False,
) -> dict[str, Any]:
    import traceback

    try:
        if async_mode:
            app_state.update_switch_status("switching", "Checking memory state...")
            force_memory_refresh()
            status = MemoryWatchdog().status
            if status.avail_pct < 0.15:
                return {
                    "status": "error",
                    "message": (
                        f"Insufficient available memory ({status.avail_gb:.1f}GB / {status.total_gb:.1f}GB). "
                        "Close other applications and retry."
                    ),
                }

        if async_mode:
            app_state.update_switch_status("switching", "Saving model settings...")
        settings = load_settings()
        settings["model_type"] = model_type

        if model_type == "gguf":
            if not gguf_path or not os.path.exists(gguf_path):
                return {"status": "error", "message": f"GGUF model file does not exist: {gguf_path}"}
            settings["gguf_path"] = gguf_path
            settings["model_name"] = gguf_path
        else:
            if not model_name:
                return {"status": "error", "message": "Missing model_name parameter"}
            settings["model_name"] = model_name
            settings["gguf_path"] = ""

        settings["n_gpu_layers"] = settings.get("n_gpu_layers", -1)
        settings["n_ctx"] = settings.get("n_ctx", 2048)
        save_settings(settings)

        if async_mode:
            app_state.update_switch_status("switching", "Unloading current model...")
        app_state.unload_model()
        gc.collect()

        if async_mode:
            force_memory_refresh()
            status = MemoryWatchdog().status
            if status.avail_pct < 0.25:
                message = (
                    f"Available memory is still too low after unload ({status.avail_gb:.1f}GB). "
                    "Restart the runtime and try again."
                )
                app_state.mark_startup_failed(message)
                return {"status": "error", "message": message}

        if async_mode:
            app_state.update_switch_status("switching", "Loading target model...")

        config = TrainingConfig()
        config.cache_dir = get_external_path("model_cache")
        config.model_type = model_type
        config.n_gpu_layers = settings.get("n_gpu_layers", -1)
        config.n_ctx = settings.get("n_ctx", 2048)

        if model_type == "self":
            if async_mode:
                app_state.update_switch_status("switching", "Loading ModelSelf native model...")
            _load_self_model_switch(config, model_name or gguf_path)
        else:
            # 原生态极：仅支持 self 模型类型
            return {"status": "error", "message": f"不支持的模型类型: {model_type}，原生态极仅支持 self 类型"}

        app_state.mark_started()
        save_config(config, os.path.join(get_external_path("checkpoints"), "training_config.json"))
        return {
            "status": "ok",
            "message": f"Model switch complete: {os.path.basename(gguf_path or model_name)}",
            "model_type": model_type,
            "model_name": app_state._loaded_model_name,
        }
    except Exception as exc:
        logger.error("Model switch failed: %s", traceback.format_exc())
        app_state.mark_startup_failed(str(exc))
        logger.error(f"Model switch failed: {exc}")
        return {"status": "error", "message": "Model switch failed"}


def _load_self_model_switch(config: TrainingConfig, model_path: str) -> None:
    from taiji import NativeInferenceEngine, TaijiMultimodalEngine, load_model
    from taiji.agent_ext.tool_registry import registry
    from taiji.core.config import get_external_path
    from taiji.core.taiji_bridge import get_taiji_bridge

    if not model_path or not os.path.isdir(model_path):
        raise RuntimeError(f"ModelSelf model directory does not exist: {model_path}")

    device = config.resolve_device()
    model, tokenizer = load_model(model_path, device=device)

    tool_names = [tool.name for tool in registry.list_tools(enabled_only=True)]
    if hasattr(tokenizer, "register_tool"):
        for name in tool_names:
            tokenizer.register_tool(name)
        if hasattr(tokenizer, "_tool_name_to_id"):
            model.set_num_tools(len(tokenizer._tool_name_to_id))
    elif hasattr(model, "set_num_tools"):
        model.set_num_tools(len(tool_names))

    trainer = NativeInferenceEngine(model, tokenizer, device)
    app_state.update_model(model, tokenizer, trainer, model_path)

    taiji = TaijiMultimodalEngine(
        model,
        tokenizer,
        device=device,
        workspace_path=get_external_path("agent_workspace"),
        memory_save_path=get_external_path(os.path.join("taiji", "user_data", "memory.json")),
    )
    taiji.register_tools(tool_names)
    app_state.set_taiji_engine(taiji)

    bridge = get_taiji_bridge()
    bridge.initialize(model=model, tokenizer=tokenizer, device=str(device))
    bridge.start_life()
    app_state.set_taiji_bridge(bridge)
