"""
态极生命状态 API
================
提供生命系统的 REST 接口：状态查询、手动触发生命活动。
"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger("RoutesLife")
router = APIRouter(prefix="/api/life", tags=["life"])


class LifeActionResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None


@router.get("/status")
async def get_life_status():
    """获取态极完整生命状态"""
    try:
        from taiji.life.life_scheduler import get_life_scheduler
        scheduler = get_life_scheduler()
        status = scheduler.get_status()
        needs = scheduler.needs.to_dict() if hasattr(scheduler, 'needs') else {}
        return {
            "is_running": status.get("is_running", False),
            "needs": needs,
            "total_interactions": status.get("total_interactions", 0),
            "uptime_seconds": status.get("uptime_seconds", 0),
        }
    except Exception as e:
        logger.error(f"获取生命状态失败: {e}")
        logger.error(f"Request failed: {e}")
        return HTTPException(status_code=500, detail="内部错误，请查看日志")


@router.post("/feed")
async def feed_taiji(reason: str = "manual"):
    """手动触发喂养（数据摄取）"""
    try:
        from taiji.life.life_scheduler import get_life_scheduler
        scheduler = get_life_scheduler()
        scheduler.record_interaction(success=True, topic="feed")

        from taiji.life.feed_engine import get_feed_engine
        engine = get_feed_engine()
        report = engine.feed(reason=reason)
        return LifeActionResponse(
            success=True,
            message=f"喂养完成: {report.items_fed} 项, {report.samples_generated} 样本",
            data={
                "items_fed": report.items_fed,
                "samples_generated": report.samples_generated,
                "avg_quality": report.avg_quality,
            }
        )
    except Exception as e:
        logger.error(f"喂养失败: {e}")
        return LifeActionResponse(success=False, message=str(e))


@router.post("/sleep")
async def sleep_taiji(reason: str = "manual"):
    """手动触发睡眠（训练/整合）"""
    try:
        from taiji.life.life_scheduler import get_life_scheduler
        scheduler = get_life_scheduler()
        scheduler.record_interaction(success=True, topic="sleep")

        from taiji.life.sleep_engine import get_sleep_engine
        engine = get_sleep_engine()
        report = engine.sleep(reason=reason)
        loss_str = f"{report.training_loss:.4f}" if report.training_loss is not None else "N/A"
        return LifeActionResponse(
            success=True,
            message=f"睡眠完成: {report.phases_completed} 阶段, loss={loss_str}",
            data={
                "phases": report.phases_completed,
                "training_loss": report.training_loss,
                "health": report.health_status,
            }
        )
    except Exception as e:
        logger.error(f"睡眠失败: {e}")
        return LifeActionResponse(success=False, message=str(e))


@router.post("/play")
async def play_taiji(reason: str = "manual"):
    """手动触发玩耍（探索/创造）"""
    try:
        from taiji.life.life_scheduler import get_life_scheduler
        scheduler = get_life_scheduler()
        scheduler.record_interaction(success=True, topic="play")

        from taiji.life.play_engine import get_play_engine
        engine = get_play_engine()
        report = engine.play(reason=reason)
        return LifeActionResponse(
            success=True,
            message=f"玩耍完成: {len(report.activities)} 活动, 心情={report.mood}",
            data={
                "activities": len(report.activities),
                "mood": report.mood,
                "traits": report.personality_traits_discovered,
            }
        )
    except Exception as e:
        logger.error(f"玩耍失败: {e}")
        return LifeActionResponse(success=False, message=str(e))


@router.post("/evolve")
async def evolve_taiji():
    """手动触发进化"""
    try:
        from taiji.life.evolution_engine import get_evolution_engine
        engine = get_evolution_engine()
        return LifeActionResponse(
            success=True,
            message=f"进化阶段: {engine.metrics.current_phase}",
            data={
                "phase": engine.metrics.current_phase,
                "tasks_completed": engine.metrics.tasks_completed,
            }
        )
    except Exception as e:
        logger.error(f"进化查询失败: {e}")
        return LifeActionResponse(success=False, message=str(e))


@router.post("/interaction")
async def record_interaction(success: bool = True, topic: str = ""):
    """记录一次用户交互（影响需求状态）"""
    try:
        from taiji.life.life_scheduler import get_life_scheduler
        scheduler = get_life_scheduler()
        scheduler.record_interaction(success=success, topic=topic)
        return LifeActionResponse(success=True, message="交互已记录")
    except Exception as e:
        logger.error(f"记录交互失败: {e}")
        return LifeActionResponse(success=False, message=str(e))
