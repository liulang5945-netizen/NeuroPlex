"""
训练控制 API 路由
================
Cortex 模式下训练走 sleep_engine，这些端点控制 app_state 训练标志。
stop/pause/resume 设置标志（sleep_engine 可选检查），reset 紧急释放 train_lock。
"""
import logging

from fastapi import APIRouter

from neuroplex.core.app_state import app_state

logger = logging.getLogger("ApiServer.Training")
router = APIRouter()


@router.post("/api/train/pause")
def pause_training():
    """暂停正在进行的训练（设置标志）"""
    if not app_state.is_training:
        return {"status": "ok", "message": "当前无训练任务"}
    app_state.pause_training_requested = True
    return {"status": "ok", "message": "训练暂停请求已发送"}


@router.post("/api/train/resume")
def resume_training():
    """恢复已暂停的训练"""
    if not app_state.is_training:
        return {"status": "ok", "message": "当前无训练任务"}
    app_state.pause_training_requested = False
    return {"status": "ok", "message": "训练已恢复"}


@router.post("/api/train/stop")
def stop_training():
    """停止正在进行的训练"""
    if not app_state.is_training:
        return {"status": "ok", "message": "当前无训练任务"}
    app_state.stop_training_requested = True
    # 立即释放训练锁，避免用户再次触发训练时被阻塞
    app_state.finish_training()
    return {"status": "ok", "message": "正在请求停止训练..."}


@router.post("/api/train/reset")
def force_reset_training():
    """紧急强制重置训练状态（释放 train_lock）"""
    was_training = app_state.is_training
    locked = app_state.train_lock.locked()
    app_state.stop_training_requested = True
    app_state.is_training = False
    app_state.stop_training_requested = False
    app_state._trainer_ref = None
    if locked:
        try:
            app_state.train_lock.release()
        except RuntimeError:
            pass
    logger.warning(f"训练状态已强制重置 (was_training={was_training}, lock_was_held={locked})")
    return {
        "status": "ok",
        "message": "训练状态已强制重置",
        "was_training": was_training,
        "lock_was_held": locked,
    }
