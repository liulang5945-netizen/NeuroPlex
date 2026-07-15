"""
训练控制 API 路由
暂停/恢复/停止/强制重置训练状态
"""
import logging

from fastapi import APIRouter

from taiji.core.app_state import app_state

logger = logging.getLogger("ApiServer.Training")
router = APIRouter()


@router.post("/api/train/pause")
def pause_training():
    """暂停正在进行的训练"""
    if not app_state.is_training:
        return {"status": "ok", "message": "当前无训练任务"}
    app_state.pause_training_requested = True
    if app_state._trainer_ref is not None:
        app_state._trainer_ref.is_paused = True
    return {"status": "ok", "message": "训练已暂停"}


@router.post("/api/train/resume")
def resume_training():
    """恢复已暂停的训练"""
    if not app_state.is_training:
        return {"status": "ok", "message": "当前无训练任务"}
    app_state.pause_training_requested = False
    if app_state._trainer_ref is not None:
        app_state._trainer_ref.is_paused = False
    return {"status": "ok", "message": "训练已恢复"}


@router.post("/api/train/stop")
def stop_training():
    """停止正在进行的训练"""
    if not app_state.is_training:
        return {"status": "ok", "message": "当前无训练任务"}
    app_state.stop_training_requested = True
    if app_state._trainer_ref is not None:
        try:
            app_state._trainer_ref.is_stopped = True
        except Exception:
            pass
    # 立即释放训练锁，避免用户再次点击训练时被阻塞。
    # daemon 线程检测到 is_stopped 后会自行退出，不再需要持有锁。
    app_state.finish_training()
    return {"status": "ok", "message": "正在请求停止训练..."}


@router.post("/api/train/reset")
def force_reset_training():
    """紧急强制重置训练状态"""
    was_training = app_state.is_training
    locked = app_state.train_lock.locked()
    app_state.stop_training_requested = True
    if app_state._trainer_ref is not None:
        try:
            app_state._trainer_ref.is_stopped = True
        except Exception:
            pass
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
