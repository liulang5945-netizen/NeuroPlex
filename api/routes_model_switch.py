"""Canonical model switch routes for runtime model lifecycle operations.

认知主体可切换：Cortex（neuroplex，冻结对照）与 Seed（taiji 原生）。
switch_model(model_type="cortex") 重载 Cortex；model_type="seed" 激活原生运行时。
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from typing import Any

from fastapi import APIRouter

from neuroplex.core.app_state import app_state
from neuroplex.core.utils import get_external_path

logger = logging.getLogger("ApiServer.ModelSwitch")
router = APIRouter()

_switch_lock = threading.Lock()
_switch_thread: threading.Thread | None = None


@router.post("/api/system/reload_model")
def reload_model() -> dict[str, Any]:
    """重载 Cortex 神经元架构（从磁盘重新装配）。"""
    return _do_switch_model(async_mode=False)


@router.post("/api/system/switch_model")
def switch_model(req: dict[str, Any]) -> dict[str, Any]:
    """切换/重载模型。

    支持的 model_type：
    - "cortex"（默认）：重载 Cortex 神经元架构（同时卸载 Seed 运行时）；
    - "seed"：加载并激活 taiji 原生 Seed 检查点（可选参数 "checkpoint"）。
    旧 model_type="self" 已废弃，会被自动路由到 Cortex 重载。
    """
    global _switch_thread

    model_type = str(req.get("model_type", "") or "").lower()

    if model_type == "seed":
        return _switch_to_seed(req)

    if model_type and model_type not in ("cortex", "self"):
        return {"status": "error", "message": f"不支持的模型类型: {model_type}，支持 cortex / seed"}

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

        app_state.update_switch_status("switching", "Reloading Cortex neuron architecture...")

        def _do_switch_async() -> None:
            try:
                result = _do_switch_model(async_mode=True)
                if result.get("status") == "ok":
                    app_state.update_switch_status("success", result.get("message", "Cortex reload complete"))
                else:
                    app_state.update_switch_status("error", "", result.get("message", "Cortex reload failed"))
            except Exception as exc:
                logger.exception("Async Cortex reload failed")
                app_state.mark_startup_failed(str(exc))
                app_state.update_switch_status("error", "", f"Cortex reload failed: {exc}")
            finally:
                _switch_lock.release()

        _switch_thread = threading.Thread(target=_do_switch_async, daemon=True)
        _switch_thread.start()
        return {
            "status": "ok",
            "message": "Starting Cortex reload...",
            "model_type": "cortex",
        }
    except Exception as exc:
        _switch_lock.release()
        logger.error(f"Cortex reload start failed: {exc}")
        return {"status": "error", "message": "Failed to start Cortex reload"}


@router.get("/api/system/switch_status")
def get_switch_status() -> dict[str, Any]:
    state = app_state.get_switch_status()
    return {
        "status": state["status"],
        "message": state["message"],
        "error": state["error"],
    }


def _switch_to_seed(req: dict[str, Any]) -> dict[str, Any]:
    """激活 Seed 原生运行时（加载检查点 + 接管聊天主路由）。"""
    from api.seed_runtime import activate_seed

    try:
        runtime = activate_seed(req.get("checkpoint"))
        app_state.update_switch_status("success", f"Seed runtime active ({runtime.name})")
        return {
            "status": "ok",
            "message": f"Seed runtime active: {runtime.name}",
            "model_type": "seed",
            "seed": runtime.status(),
        }
    except Exception as exc:
        logger.error(f"Seed activation failed: {exc}")
        app_state.update_switch_status("error", "", f"Seed activation failed: {exc}")
        return {"status": "error", "message": f"Seed 激活失败: {exc}"}


@router.post("/api/system/pub_reset")
def force_reset_publishing() -> dict[str, Any]:
    result = app_state.force_reset_publishing()
    return {"status": "ok", **result}


def _do_switch_model(*, async_mode: bool = False) -> dict[str, Any]:
    """重载 Cortex 神经元架构。

    流程：
    1. 卸载当前 model（释放引用）
    2. 调用 load_model_on_startup() 重新装配 Cortex
    """
    import traceback

    try:
        if async_mode:
            app_state.update_switch_status("switching", "Unloading current Cortex...")

        # 切回 Cortex 主路径：Seed 运行时与 Cortex 互斥，先卸载。
        from api.seed_runtime import deactivate_seed
        deactivate_seed()

        app_state.unload_model()
        gc.collect()

        if async_mode:
            app_state.update_switch_status("switching", "Loading Cortex neuron architecture...")

        from neuroplex.core.model_loader import load_model_on_startup
        load_model_on_startup()

        if app_state.startup_error:
            return {"status": "error", "message": f"Cortex reload failed: {app_state.startup_error}"}

        n_neurons = len(getattr(app_state.model, 'neurons', {}))
        return {
            "status": "ok",
            "message": f"Cortex reload complete: {n_neurons} neurons",
            "model_type": "cortex",
            "model_name": app_state._loaded_model_name,
        }
    except Exception as exc:
        logger.error("Cortex reload failed: %s", traceback.format_exc())
        app_state.mark_startup_failed(str(exc))
        return {"status": "error", "message": f"Cortex reload failed: {exc}"}
