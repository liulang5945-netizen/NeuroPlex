"""
态极训练辅助 API 路由
POST /api/taiji/train/stop   — 停止态极训练
POST /api/taiji/train/pause  — 暂停态极训练
POST /api/taiji/train/resume — 恢复态极训练
GET  /api/taiji/train/status — 查询训练状态

训练主端点 POST /api/taiji/train 在 routes_taiji.py 中定义。
"""
import logging

from fastapi import APIRouter

from taiji.core.app_state import app_state

logger = logging.getLogger("ApiServer.TaijiTrain")
router = APIRouter()

# 停止标志（由 routes_taiji.py 的训练循环检查）
_taiji_stop_requested = False


def is_stop_requested() -> bool:
    """供 routes_taiji.py 检查是否请求了停止"""
    return _taiji_stop_requested


def clear_stop_request():
    """训练开始时清除停止标志"""
    global _taiji_stop_requested
    _taiji_stop_requested = False


def _get_trainer():
    """获取当前训练器实例"""
    return getattr(app_state, "_trainer_ref", None)


@router.post("/api/taiji/train/stop")
async def taiji_train_stop():
    """停止态极训练（先暂停再停止，确保尽快响应）"""
    global _taiji_stop_requested
    _taiji_stop_requested = True
    # 立即暂停训练器，让训练循环在下一个 batch 间隙快速退出
    trainer = _get_trainer()
    if trainer and hasattr(trainer, "pause"):
        trainer.pause()
    return {"status": "stopping"}


@router.post("/api/taiji/train/pause")
async def taiji_train_pause():
    """暂停态极训练"""
    trainer = _get_trainer()
    if trainer and hasattr(trainer, "pause"):
        trainer.pause()
        return {"status": "paused"}
    return {"status": "no_trainer"}


@router.post("/api/taiji/train/resume")
async def taiji_train_resume():
    """恢复态极训练"""
    trainer = _get_trainer()
    if trainer and hasattr(trainer, "resume"):
        trainer.resume()
        return {"status": "resumed"}
    return {"status": "no_trainer"}


@router.get("/api/taiji/train/status")
async def taiji_train_status():
    """查询态极训练状态"""
    trainer = _get_trainer()
    return {
        "running": app_state.is_training,
        "stop_requested": _taiji_stop_requested,
        "paused": getattr(trainer, "is_paused", False) if trainer else False,
    }


@router.post("/api/taiji/train/reset")
async def taiji_train_reset():
    """紧急重置：强制停止训练并释放锁"""
    global _taiji_stop_requested
    _taiji_stop_requested = True
    trainer = _get_trainer()
    if trainer:
        trainer.stop()
    app_state.finish_training()
    return {"status": "reset", "message": "训练已强制重置"}