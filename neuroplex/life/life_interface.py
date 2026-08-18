"""
TaijiCore 生命系统线程安全API接口 (ThreadSafeLifeInterface)
============================================================

为 taiji.life 生命系统提供线程安全的并发访问接口，支持：
- 异步/同步双接口
- 多线程安全的状态管理
- 生命周期事件的原子性操作
- 优雅的错误处理与日志
"""

import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger("ThreadSafeLifeInterface")


@dataclass
class LifeStateSnapshot:
    """生命状态快照（用于线程安全的状态读取）"""
    timestamp: str
    hunger: float
    fatigue: float
    boredom: float
    stress: float
    curiosity: float
    activity_log: List[str]
    last_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ThreadSafeLifeInterface:
    """
    TaijiCore 生命系统的线程安全API接口
    
    提供对生命系统各引擎的原子性访问，确保并发操作的一致性。
    
    使用示例：
        interface = ThreadSafeLifeInterface()
        
        # 同步接口
        state = interface.get_life_state()
        interface.feed()
        
        # 异步接口
        await interface.async_sleep(duration=3600)
        await interface.async_play()
    """

    def __init__(self, scheduler=None):
        """
        初始化线程安全生命接口
        
        Args:
            scheduler: 可选的life_scheduler实例，若为None则延迟初始化
        """
        self._scheduler = scheduler
        self._lock = threading.RLock()  # 统一互斥锁，同步和异步方法共用
        self._activity_log: List[str] = []
        self._event_handlers: Dict[str, List[Callable]] = {
            'on_feed': [],
            'on_sleep': [],
            'on_play': [],
            'on_evolve': [],
            'on_stress': [],
        }

        logger.info("ThreadSafeLifeInterface initialized")

    # ===== 同步接口 =====

    def get_life_state(self) -> LifeStateSnapshot:
        """
        原子性地读取当前生命状态
        
        Returns:
            LifeStateSnapshot: 包含所有需求状态的快照
        """
        with self._lock:
            if not self._scheduler:
                logger.warning("Scheduler not initialized")
                return self._create_empty_snapshot()

            try:
                needs = self._scheduler.needs
                snapshot = LifeStateSnapshot(
                    timestamp=datetime.now().isoformat(),
                    hunger=needs.hunger,
                    fatigue=needs.fatigue,
                    boredom=needs.boredom,
                    stress=needs.stress,
                    curiosity=needs.curiosity,
                    activity_log=self._activity_log[-10:],  # 最后10条日志
                    last_action=self._activity_log[-1] if self._activity_log else None,
                )
                return snapshot
            except Exception as e:
                logger.error(f"Failed to get life state: {e}")
                return self._create_empty_snapshot()

    def feed(self, amount: float = 30.0) -> bool:
        """
        原子性地执行吃饭操作，委托给 LifeScheduler。

        Args:
            amount: 降低的饥饿度数值 (0-100)

        Returns:
            bool: 操作是否成功
        """
        with self._lock:
            if not self._scheduler:
                logger.error("Scheduler not initialized")
                return False

            try:
                result = self._scheduler.force_action("feed")
                success = result.get("success", False) if isinstance(result, dict) else result
                success = bool(success)
                action = f"feed(amount={amount})"
                self._log_action(action)
                if success:
                    self._trigger_event('on_feed', amount)
                logger.info(f"Feed operation {'successful' if success else 'failed'}: {action}")
                return success
            except Exception as e:
                logger.error(f"Feed operation failed: {e}")
                return False

    def sleep(self, duration: float = 3600) -> bool:
        """
        原子性地执行睡眠操作，委托给 LifeScheduler。

        Args:
            duration: 睡眠时长（秒，用于计算疲劳度恢复）

        Returns:
            bool: 操作是否成功
        """
        with self._lock:
            if not self._scheduler:
                logger.error("Scheduler not initialized")
                return False

            try:
                result = self._scheduler.force_action("sleep")
                success = result.get("success", False) if isinstance(result, dict) else result
                success = bool(success)
                action = f"sleep(duration={duration}s)"
                self._log_action(action)
                if success:
                    self._trigger_event('on_sleep', duration)
                logger.info(f"Sleep operation {'successful' if success else 'failed'}: {action}")
                return success
            except Exception as e:
                logger.error(f"Sleep operation failed: {e}")
                return False

    def play(self, enjoyment: float = 20.0) -> bool:
        """
        原子性地执行玩耍操作，委托给 LifeScheduler。

        Args:
            enjoyment: 玩耍的快乐度 (0-100)

        Returns:
            bool: 操作是否成功
        """
        with self._lock:
            if not self._scheduler:
                logger.error("Scheduler not initialized")
                return False

            try:
                result = self._scheduler.force_action("play")
                success = result.get("success", False) if isinstance(result, dict) else result
                success = bool(success)
                action = f"play(enjoyment={enjoyment})"
                self._log_action(action)
                if success:
                    self._trigger_event('on_play', enjoyment)
                logger.info(f"Play operation {'successful' if success else 'failed'}: {action}")
                return success
            except Exception as e:
                logger.error(f"Play operation failed: {e}")
                return False

    def evolve(self, improvement: Dict[str, float]) -> bool:
        """
        原子性地执行进化操作（改进多个需求指标）
        
        Args:
            improvement: 改进字典，如 {'hunger': -10, 'stress': -5}
            
        Returns:
            bool: 操作是否成功
        """
        with self._lock:
            if not self._scheduler:
                logger.error("Scheduler not initialized")
                return False

            try:
                needs = self._scheduler.needs
                for key, value in improvement.items():
                    if hasattr(needs, key):
                        current = getattr(needs, key)
                        new_value = max(0, min(100, current + value))
                        setattr(needs, key, new_value)

                action = f"evolve({improvement})"
                self._log_action(action)
                self._trigger_event('on_evolve', improvement)
                logger.info(f"Evolution successful: {action}")
                return True
            except Exception as e:
                logger.error(f"Evolution failed: {e}")
                return False

    def apply_stress(self, stress_level: float = 10.0) -> bool:
        """
        原子性地应用压力
        
        Args:
            stress_level: 施加的压力等级 (0-100)
            
        Returns:
            bool: 操作是否成功
        """
        with self._lock:
            if not self._scheduler:
                logger.error("Scheduler not initialized")
                return False

            try:
                self._scheduler.needs.stress = min(100, self._scheduler.needs.stress + stress_level)
                action = f"apply_stress(level={stress_level})"
                self._log_action(action)
                self._trigger_event('on_stress', stress_level)
                logger.info(f"Stress applied: {action}")
                return True
            except Exception as e:
                logger.error(f"Failed to apply stress: {e}")
                return False

    # ===== 异步接口 =====

    async def async_feed(self, amount: float = 30.0) -> bool:
        """异步吃饭操作"""
        with self._lock:
            return self.feed(amount)

    async def async_sleep(self, duration: float = 3600) -> bool:
        """异步睡眠操作（支持实际延迟）"""
        with self._lock:
            result = self.sleep(duration)
            if result:
                # 模拟异步睡眠（可选）
                # await asyncio.sleep(min(1, duration))
                pass
            return result

    async def async_play(self, enjoyment: float = 20.0) -> bool:
        """异步玩耍操作"""
        with self._lock:
            return self.play(enjoyment)

    async def async_evolve(self, improvement: Dict[str, float]) -> bool:
        """异步进化操作"""
        with self._lock:
            return self.evolve(improvement)

    async def async_get_state(self) -> LifeStateSnapshot:
        """异步获取生命状态"""
        with self._lock:
            return self.get_life_state()

    # ===== 事件系统 =====

    def on_event(self, event_name: str, handler: Callable) -> None:
        """
        注册事件处理器
        
        支持的事件：on_feed, on_sleep, on_play, on_evolve, on_stress
        """
        if event_name in self._event_handlers:
            self._event_handlers[event_name].append(handler)
            logger.debug(f"Event handler registered for {event_name}")

    def off_event(self, event_name: str, handler: Callable) -> None:
        """移除事件处理器"""
        if event_name in self._event_handlers and handler in self._event_handlers[event_name]:
            self._event_handlers[event_name].remove(handler)
            logger.debug(f"Event handler removed for {event_name}")

    # ===== 内部方法 =====

    def _log_action(self, action: str) -> None:
        """记录操作到日志"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {action}"
        self._activity_log.append(log_entry)
        # 限制日志大小（保持最后1000条）
        if len(self._activity_log) > 1000:
            self._activity_log = self._activity_log[-1000:]

    def _trigger_event(self, event_name: str, data: Any = None) -> None:
        """触发事件处理器"""
        try:
            for handler in self._event_handlers.get(event_name, []):
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_name}: {e}")
        except Exception as e:
            logger.error(f"Error triggering event {event_name}: {e}")

    def _create_empty_snapshot(self) -> LifeStateSnapshot:
        """创建空的状态快照"""
        return LifeStateSnapshot(
            timestamp=datetime.now().isoformat(),
            hunger=50.0,
            fatigue=50.0,
            boredom=50.0,
            stress=50.0,
            curiosity=50.0,
            activity_log=[],
            last_action=None,
        )

    def set_scheduler(self, scheduler) -> None:
        """
        设置life_scheduler实例（用于延迟初始化）
        
        Args:
            scheduler: life_scheduler实例
        """
        with self._lock:
            self._scheduler = scheduler
            logger.info("Scheduler set for ThreadSafeLifeInterface")

    def get_activity_log(self, limit: int = 50) -> List[str]:
        """获取活动日志"""
        with self._lock:
            return self._activity_log[-limit:]

    def clear_activity_log(self) -> None:
        """清空活动日志"""
        with self._lock:
            self._activity_log.clear()
            logger.info("Activity log cleared")


# 全局接口实例（可选）
_global_interface: Optional[ThreadSafeLifeInterface] = None


def get_global_interface() -> ThreadSafeLifeInterface:
    """获取全局线程安全接口实例"""
    global _global_interface
    if _global_interface is None:
        _global_interface = ThreadSafeLifeInterface()
    return _global_interface


def set_global_scheduler(scheduler) -> None:
    """设置全局接口的调度器"""
    interface = get_global_interface()
    interface.set_scheduler(scheduler)


__all__ = [
    'ThreadSafeLifeInterface',
    'LifeStateSnapshot',
    'get_global_interface',
    'set_global_scheduler',
]
