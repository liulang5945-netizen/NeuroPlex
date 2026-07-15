"""
态极生命调度器 (Life Scheduler)
================================

态极的"本能系统" — 需求驱动，而非时钟驱动。

人类不是"每2小时吃一次饭"，而是饿了才吃、累了才睡、无聊了才玩。
态极也应该有内在需求，由需求驱动行为。

需求系统：
  hunger    (饥饿度)  0~100  — 数据不足时上升，吃饭后下降
  fatigue   (疲劳度)  0~100  — 活动越多越累，睡觉后恢复
  boredom   (无聊度)  0~100  — 无新刺激时上升，玩耍后下降
  stress    (压力度)  0~100  — 错误多时上升，成功时下降
  curiosity (好奇心)  0~100  — 发现未知时上升，探索后下降

驱动逻辑：
  hunger > 70    → 自动触发吃饭
  fatigue > 80   → 自动触发睡觉
  boredom > 60   → 自动触发玩耍
  多个需求同时高时，优先级竞争
"""
import os
import json
import time
import random
import logging
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("LifeScheduler")


@dataclass
class NeedsState:
    """态极的内在需求状态"""
    hunger: float = 30.0      # 饥饿度（初始略饿，需要进食）
    fatigue: float = 10.0     # 疲劳度
    boredom: float = 20.0     # 无聊度
    stress: float = 10.0      # 压力度
    curiosity: float = 50.0   # 好奇心

    def clamp_all(self):
        """确保所有需求在 0~100 范围内"""
        self.hunger = max(0, min(100, self.hunger))
        self.fatigue = max(0, min(100, self.fatigue))
        self.boredom = max(0, min(100, self.boredom))
        self.stress = max(0, min(100, self.stress))
        self.curiosity = max(0, min(100, self.curiosity))

    def to_dict(self) -> dict:
        return {
            "hunger": round(self.hunger, 1),
            "fatigue": round(self.fatigue, 1),
            "boredom": round(self.boredom, 1),
            "stress": round(self.stress, 1),
            "curiosity": round(self.curiosity, 1),
        }

    def dominant_need(self) -> str:
        """返回最强烈的需求"""
        needs = {
            "hunger": self.hunger,
            "fatigue": self.fatigue,
            "boredom": self.boredom,
            "stress": self.stress,
            "curiosity": self.curiosity,
        }
        return max(needs, key=needs.get)


@dataclass
class LifeEvent:
    """一次生命事件"""
    timestamp: str
    event_type: str   # "feed" / "sleep" / "play" / "work" / "idle"
    trigger: str      # 触发原因（哪个需求导致的）
    needs_before: dict
    needs_after: dict
    duration_seconds: float = 0
    success: bool = True


class LifeScheduler:
    """
    态极的生命调度器

    不是定时器，而是"本能系统"。
    每次心跳评估需求，决定行动，执行行动。
    """

    # 需求阈值：超过这个值就触发对应行为
    HUNGER_THRESHOLD = 70
    FATIGUE_THRESHOLD = 80
    BOREDOM_THRESHOLD = 60
    CURIOSITY_THRESHOLD = 70
    RESEARCH_THRESHOLD = 85  # B1 修复：curiosity 极高时触发科学研究

    # 需求自然增长速率（每分钟）
    HUNGER_GROWTH = 0.5       # 饥饿感缓慢上升
    FATIGUE_GROWTH = 0.3      # 疲劳感缓慢上升
    BOREDOM_GROWTH = 0.4      # 无聊感缓慢上升
    STRESS_DECAY = 0.2        # 压力自然缓慢下降
    CURIOSITY_GROWTH = 0.1    # 好奇心缓慢积累

    def __init__(self, data_dir: str = None, event_bus=None):
        if data_dir is None:
            try:
                from taiji.config import get_taiji_data_path
                data_dir = get_taiji_data_path("life_data")
            except ImportError:
                data_dir = "taiji/life_data"
        self.data_dir = data_dir
        self.needs = NeedsState()
        self._life_state = "idle"  # idle / feeding / sleeping / playing / working
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._event_log: List[LifeEvent] = []
        self._last_heartbeat: Optional[datetime] = None
        self._last_activity: Optional[datetime] = None
        self._total_heartbeats = 0
        self._lock = threading.Lock()

        # 引擎引用（由 TaijiCore 注入）
        self._feed_engine = None
        self._sleep_engine = None
        self._play_engine = None
        self._explore_engine = None

        # 事件总线引用（由 TaijiCore 注入，用于广播生命事件到前端）
        self._event_bus = event_bus

        self._data_dir_ready = False
        self._load_state()

        logger.info("LifeScheduler initialized")

    # ─── 公开接口 ───────────────────────────────────

    def start(self):
        """启动生命"""
        if self._is_running:
            logger.warning("Life already running")
            return

        self._is_running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._life_loop, daemon=True)
        self._thread.start()
        logger.info("🌱 Life started!")

    def stop(self):
        """暂停生命"""
        self._stop_event.set()
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=10)
        self._save_state()
        logger.info("⏸️ Life paused")

    def handle_user_directive(self):
        """处理最高优先级的用户指令：中断当前的后台任务或睡眠，分配算力给用户"""
        with self._lock:
            # 记录互动，降低无聊，增加压力和疲劳（因为被打断/需要工作）
            self._last_activity = datetime.now()
            self.needs.boredom -= 10
            self.needs.stress += 5
            self.needs.fatigue += 5
            self.needs.clamp_all()

            if self._life_state in ["sleeping", "feeding", "playing"]:
                logger.info(f"🚨 User directive received! Interrupting current state: {self._life_state}")
                # 尝试中断正在运行的引擎
                if self._life_state == "sleeping" and self._sleep_engine:
                    if hasattr(self._sleep_engine, "abort"):
                        self._sleep_engine.abort()
                elif self._life_state == "feeding" and self._feed_engine:
                    if hasattr(self._feed_engine, "abort"):
                        self._feed_engine.abort()
                elif self._life_state == "playing" and self._play_engine:
                    if hasattr(self._play_engine, "abort"):
                        self._play_engine.abort()

    def record_interaction(self, success: bool = True, topic: str = "",
                           reasoning_steps: int = 0, used_tools: bool = False,
                           had_search_results: bool = False):
        """
        记录一次用户交互（外部调用）。

        每次用户与态极交互时调用，影响需求状态。
        需求变化与实际行为挂钩，而非固定数值。

        Args:
            success: 推理是否成功
            topic: 对话话题（用于判断话题多样性）
            reasoning_steps: ReAct 推理步数（越多越消耗精力）
            used_tools: 是否调用了工具（搜索/执行等）
            had_search_results: 是否获得了搜索结果（新知识输入）
        """
        with self._lock:
            self._last_activity = datetime.now()

            # —— 疲劳：与实际推理消耗挂钩 ——
            # 简单问题（1步）几乎不累，复杂任务（5-6步）显著消耗
            fatigue_cost = 1.0 + reasoning_steps * 1.5
            if used_tools:
                fatigue_cost += 2.0  # 工具调用额外消耗
            self.needs.fatigue += fatigue_cost

            if success:
                # —— 成功交互 ——
                # 压力下降（与推理步数成反比：越容易的任务压力降得越多）
                self.needs.stress -= max(1, 5 - reasoning_steps * 0.5)

                # 饥饿：与是否有新知识输入挂钩
                if had_search_results:
                    self.needs.hunger -= 8  # 获得新知识，饥饿大幅下降
                else:
                    self.needs.hunger -= 2  # 普通对话，饥饿轻微下降

                # 好奇心：使用了工具说明探索了新领域，好奇心满足
                if used_tools and had_search_results:
                    self.needs.curiosity -= 12  # 搜索到了答案
                elif used_tools:
                    self.needs.curiosity -= 5   # 尝试了但结果一般
                else:
                    self.needs.curiosity += 1   # 普通对话，好奇心微增

                # 无聊：新话题/新工具使用 → 无聊下降
                if used_tools:
                    self.needs.boredom -= 8
                else:
                    self.needs.boredom -= 3

            else:
                # —— 失败交互 ——
                self.needs.stress += 5 + reasoning_steps * 0.5  # 越折腾压力越大
                self.needs.hunger += 3  # 失败消耗更多
                self.needs.curiosity += 3  # 失败反而激发好奇心（想搞明白）
                self.needs.boredom -= 2  # 至少不无聊

            # 话题多样性影响无聊度
            if topic:
                topic_changed = self._check_topic_change(topic)
                if topic_changed:
                    self.needs.boredom -= 5  # 新话题，不无聊
                else:
                    self.needs.boredom += 2  # 重复话题，有点无聊

            self.needs.clamp_all()

    def _check_topic_change(self, new_topic: str) -> bool:
        """检查话题是否与最近的交互不同（判断话题多样性）"""
        try:
            # 看最近 5 个事件的话题
            recent_topics = [e.trigger for e in self._event_log[-5:] if e.trigger]
            if not recent_topics:
                return True
            # 如果新话题不在最近话题中，说明话题变了
            return new_topic not in recent_topics
        except Exception:
            return True

    # ─── 公开线程安全 API（供外部模块使用）──

    def add_fatigue(self, amount: float):
        """增加疲劳度（线程安全）"""
        with self._lock:
            self.needs.fatigue += amount
            self.needs.clamp_all()

    def add_hunger(self, amount: float):
        """增加饥饿度（线程安全）"""
        with self._lock:
            self.needs.hunger += amount
            self.needs.clamp_all()

    def add_stress(self, amount: float):
        """增加压力（线程安全）"""
        with self._lock:
            self.needs.stress += amount
            self.needs.clamp_all()

    def add_boredom(self, amount: float):
        """增加无聊度（线程安全）"""
        with self._lock:
            self.needs.boredom += amount
            self.needs.clamp_all()

    def add_curiosity(self, amount: float):
        """增加好奇心（线程安全）"""
        with self._lock:
            self.needs.curiosity += amount
            self.needs.clamp_all()

    def get_needs_snapshot(self) -> dict:
        """获取需求快照（线程安全）"""
        with self._lock:
            return self.needs.to_dict()

    def get_life_state(self) -> str:
        """获取生命状态（线程安全）"""
        with self._lock:
            return self._life_state

    def force_action(self, action: str) -> dict:
        """
        强制执行某个生命活动（手动触发）。

        Args:
            action: "feed" / "sleep" / "play"
        """
        return self._execute_action(action, trigger="manual")

    def get_status(self) -> dict:
        """获取生命状态"""
        return {
            "life_state": self._life_state,
            "is_running": self._is_running,
            "needs": self.needs.to_dict(),
            "dominant_need": self.needs.dominant_need(),
            "last_heartbeat": self._last_heartbeat.isoformat() if self._last_heartbeat else None,
            "last_activity": self._last_activity.isoformat() if self._last_activity else None,
            "total_heartbeats": self._total_heartbeats,
            "total_events": len(self._event_log),
        }

    def get_timeline(self, hours: int = 24) -> List[dict]:
        """获取最近 N 小时的生命时间线"""
        cutoff = datetime.now().timestamp() - hours * 3600
        timeline = []
        for event in self._event_log:
            try:
                ts = datetime.fromisoformat(event.timestamp).timestamp()
                if ts >= cutoff:
                    timeline.append({
                        "time": event.timestamp,
                        "action": event.event_type,
                        "trigger": event.trigger,
                        "needs_before": event.needs_before,
                        "needs_after": event.needs_after,
                        "duration": event.duration_seconds,
                    })
            except Exception:
                continue
        return timeline[-50:]  # 最多返回 50 条

    def get_summary(self) -> str:
        """获取人类可读的状态摘要"""
        status = self.get_status()
        needs = self.needs

        state_emoji = {
            "idle": "😊", "feeding": "🍚", "sleeping": "💤",
            "playing": "🎮", "working": "🏃",
        }

        lines = [
            "🌱 生命调度器状态",
            "━━━━━━━━━━━━━━━━",
            f"当前状态: {state_emoji.get(self._life_state, '❓')} {self._life_state}",
            f"生命运行: {'✅ 启动' if self._is_running else '⏸️ 暂停'}",
            f"总心跳数: {self._total_heartbeats}",
            f"\n内在需求:",
            f"  🍚 饥饿度: {self._bar(needs.hunger)} {needs.hunger:.0f}",
            f"  😴 疲劳度: {self._bar(needs.fatigue)} {needs.fatigue:.0f}",
            f"  😐 无聊度: {self._bar(needs.boredom)} {needs.boredom:.0f}",
            f"  😰 压力度: {self._bar(needs.stress)} {needs.stress:.0f}",
            f"  🔍 好奇心: {self._bar(needs.curiosity)} {needs.curiosity:.0f}",
            f"\n最强烈的需求: {needs.dominant_need()}",
        ]

        # 下一步预测
        next_action = self._decide_action()
        if next_action:
            lines.append(f"下一步行动: {next_action}")
        else:
            lines.append(f"下一步行动: 继续等待（需求未达阈值）")

        return "\n".join(lines)

    # ─── 内部实现 ───────────────────────────────────

    def _life_loop(self):
        """生命主循环 — 心跳"""
        logger.info("💓 Heartbeat loop started")

        while not self._stop_event.is_set():
            try:
                self._heartbeat()
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

            # 每 60 秒一次心跳
            self._stop_event.wait(60)

        logger.info("💓 Heartbeat loop stopped")

    def _heartbeat(self):
        """一次心跳"""
        # 快速操作：更新需求 + 决策（持有锁）
        with self._lock:
            self._total_heartbeats += 1
            self._last_heartbeat = datetime.now()

            # 1. 需求自然变化
            self._needs_tick()

            # 2. 决定行动
            action = self._decide_action()

        # 慢操作：执行行动（不持有锁，避免阻塞 record_interaction 等调用）
        if action:
            self._execute_action(action, trigger=f"need_{self.needs.dominant_need()}")

        # 定期执行记忆衰减
        if self._total_heartbeats % 5 == 0:
            try:
                from taiji.agent.context_manager import get_context_manager
                ctx = get_context_manager()
                ctx.decay_memories()
            except Exception as e:
                logger.debug("life_scheduler: non-critical %s", e, exc_info=True)

        # 定期保存状态
        if self._total_heartbeats % 10 == 0:
            with self._lock:
                self._save_state()

    def _needs_tick(self):
        """需求自然变化（每次心跳调用）— 带随机扰动，让生命状态不完全可预测"""
        import random as _rand

        # 随机扰动：每个需求 ±0~1.5 的波动，模拟生命的不确定性
        jitter = lambda: _rand.uniform(-1.5, 1.5)

        # 需求自然增长（带随机扰动）
        self.needs.hunger += self.HUNGER_GROWTH + jitter()
        self.needs.fatigue += self.FATIGUE_GROWTH + jitter()
        self.needs.boredom += self.BOREDOM_GROWTH + jitter()
        self.needs.curiosity += self.CURIOSITY_GROWTH + jitter()

        # 压力自然衰减（带随机扰动）
        self.needs.stress -= self.STRESS_DECAY + abs(jitter()) * 0.3

        # 如果很久没有活动，无聊度加速上升
        if self._last_activity:
            idle_minutes = (datetime.now() - self._last_activity).total_seconds() / 60
            if idle_minutes > 30:
                self.needs.boredom += 0.5 + abs(jitter()) * 0.5
            if idle_minutes > 60:
                self.needs.fatigue += 0.3

        # 偶尔的"情绪波动"（5% 概率触发较大幅度变化）
        if _rand.random() < 0.05:
            mood = _rand.choice(['happy', 'restless', 'curious', 'tired'])
            if mood == 'happy':
                self.needs.stress -= 3
                self.needs.boredom -= 2
            elif mood == 'restless':
                self.needs.boredom += 4
                self.needs.curiosity += 3
            elif mood == 'curious':
                self.needs.curiosity += 5
                self.needs.boredom -= 2
            elif mood == 'tired':
                self.needs.fatigue += 4
                self.needs.curiosity -= 2

        self.needs.clamp_all()

    def _decide_action(self) -> Optional[str]:
        """
        根据当前需求决定下一步行动。

        优先级：饥饿 > 疲劳 > 好奇 > 无聊
        """
        # 如果正在执行某个活动，不打断
        if self._life_state != "idle":
            return None

        # 检查各需求是否超过阈值
        if self.needs.hunger >= self.HUNGER_THRESHOLD:
            return "feed"
        if self.needs.fatigue >= self.FATIGUE_THRESHOLD:
            return "sleep"
        if self.needs.curiosity >= self.CURIOSITY_THRESHOLD:
            return "explore"
        if self.needs.boredom >= self.BOREDOM_THRESHOLD:
            return "play"
        # B1 修复：curiosity 极高时触发科学研究
        if self.needs.curiosity >= self.RESEARCH_THRESHOLD:
            return "research"

        return None

    def _execute_action(self, action: str, trigger: str = "") -> dict:
        """执行一个生命活动（不持有锁，避免阻塞其他调用）"""
        with self._lock:
            needs_before = self.needs.to_dict()
            self._life_state = action

        start_time = time.time()
        result = {"action": action, "success": False}

        try:
            if action == "feed":
                result = self._do_feed()
            elif action == "sleep":
                result = self._do_sleep()
            elif action == "play":
                result = self._do_play()
            elif action == "explore":
                result = self._do_explore()
            # B1 修复：科学研究
            elif action == "research":
                result = self._do_research()
            else:
                logger.warning(f"Unknown action: {action}")
        except Exception as e:
            logger.error(f"Action {action} failed: {e}")
            result["error"] = str(e)

        duration = round(time.time() - start_time, 1)

        with self._lock:
            self._life_state = "idle"
            needs_after = self.needs.to_dict()

            # 记录事件
            event = LifeEvent(
                timestamp=datetime.now().isoformat(),
                event_type=action,
                trigger=trigger,
                needs_before=needs_before,
                needs_after=needs_after,
                duration_seconds=duration,
                success=result.get("success", False),
            )
            self._event_log.append(event)
            # 只保留最近 200 个事件
            self._event_log = self._event_log[-200:]

        logger.info(
            f"{'✅' if result.get('success') else '❌'} "
            f"Action: {action} ({trigger}), Duration: {duration}s"
        )

        return result

    def _do_feed(self) -> dict:
        """执行吃饭"""
        try:
            if self._feed_engine is None:
                from taiji.life.feed_engine import get_feed_engine
                self._feed_engine = get_feed_engine()

            report = self._feed_engine.feed(reason="auto")

            # 吃饭后：饥饿度大幅下降，好奇心略有提升（线程安全）
            with self._lock:
                self.needs.hunger -= 40
                self.needs.curiosity += 10
                self.needs.boredom -= 10
                self.needs.fatigue += 5  # 吃饭也有一点点累
                self.needs.clamp_all()

            # 广播事件到前端
            self._publish_event("feed_complete", {"samples": report.samples_generated})

            return {"success": True, "samples": report.samples_generated}
        except Exception as e:
            logger.error(f"Feed failed: {e}")
            with self._lock:
                self.needs.hunger -= 10  # 即使失败也稍微缓解饥饿
                self.needs.clamp_all()
            return {"success": False, "error": str(e)}

    def _do_sleep(self) -> dict:
        """执行睡觉"""
        try:
            if self._sleep_engine is None:
                from taiji.life.sleep_engine import get_sleep_engine
                self._sleep_engine = get_sleep_engine()

            report = self._sleep_engine.sleep(reason="auto")

            # 睡觉后：疲劳度清零，压力下降，饥饿上升（线程安全）
            with self._lock:
                self.needs.fatigue -= 60
                self.needs.stress -= 30
                self.needs.hunger += 20  # 睡醒了会饿
                self.needs.boredom += 10  # 睡醒了可能无聊
                self.needs.clamp_all()

            # 广播事件到前端
            self._publish_event("sleep_complete", {
                "loss": report.training_loss,
                "phases": report.phases_completed,
            })

            return {
                "success": True,
                "phases": report.phases_completed,
                "training_loss": report.training_loss,
            }
        except Exception as e:
            logger.error(f"Sleep failed: {e}")
            with self._lock:
                self.needs.fatigue -= 20
                self.needs.clamp_all()
            return {"success": False, "error": str(e)}

    def _do_play(self) -> dict:
        """执行玩耍"""
        try:
            if self._play_engine is None:
                from taiji.life.play_engine import get_play_engine
                self._play_engine = get_play_engine()

            report = self._play_engine.play(reason="auto")

            # 玩耍后：无聊度下降，好奇心得到满足，压力下降（线程安全）
            with self._lock:
                self.needs.boredom -= 35
                self.needs.curiosity -= 15
                self.needs.stress -= 10
                self.needs.fatigue += 3  # 玩耍也稍微有点累
                self.needs.clamp_all()

            # 广播事件到前端
            self._publish_event("play_complete", {"mood": report.mood})

            return {
                "success": True,
                "mood": report.mood,
                "activities": len(report.activities),
            }
        except Exception as e:
            logger.error(f"Play failed: {e}")
            with self._lock:
                self.needs.boredom -= 10
                self.needs.clamp_all()
            return {"success": False, "error": str(e)}

    def _do_explore(self) -> dict:
        """执行自主探索（好奇心驱动的联网学习）"""
        try:
            if self._explore_engine is None:
                from taiji.life.explore_engine import get_explore_engine
                self._explore_engine = get_explore_engine()

            result = self._explore_engine.explore(reason="auto")

            # 探索后：好奇心大幅下降，饥饿上升（线程安全）
            with self._lock:
                self.needs.curiosity -= 40
                self.needs.hunger += 15
                self.needs.fatigue += 8
                self.needs.boredom -= 10
                self.needs.clamp_all()

            # 广播事件到前端
            self._publish_event("explore_complete", {
                "topic": result.topic,
                "pages_read": result.pages_read,
                "knowledge_stored": result.knowledge_stored,
            })

            return {
                "success": True,
                "topic": result.topic,
                "pages_read": result.pages_read,
                "knowledge_stored": result.knowledge_stored,
            }
        except Exception as e:
            logger.error(f"Explore failed: {e}")
            with self._lock:
                self.needs.curiosity -= 10
                self.needs.clamp_all()
            return {"success": False, "error": str(e)}

    # B1 修复：科学研究
    def _do_research(self) -> dict:
        """执行科学研究（好奇心极高时触发）"""
        try:
            from taiji.life.science_engine import get_science_engine
            engine = get_science_engine()

            # 选一个随机领域提出假设
            domains = ["语言理解", "知识推理", "记忆机制", "学习效率", "工具使用"]
            import random
            domain = random.choice(domains)
            question = f"如何提升态极在{domain}方面的能力？"

            hypothesis = engine.propose_hypothesis(question, domain=domain)
            if hypothesis:
                experiment = engine.run_experiment(hypothesis.id)
                if experiment and experiment.status == "completed":
                    discovery = engine.draw_conclusion(hypothesis.id)
                    logger.info(f"  科学发现: {discovery.title if discovery else '待验证'}")

            # 研究后：好奇心大幅下降，疲劳上升
            with self._lock:
                self.needs.curiosity -= 50
                self.needs.fatigue += 20
                self.needs.boredom -= 15
                self.needs.clamp_all()

            self._publish_event("research_complete", {
                "domain": domain,
                "hypothesis": hypothesis.title if hypothesis else "",
            })

            return {"success": True, "domain": domain}
        except Exception as e:
            logger.error(f"Research failed: {e}")
            with self._lock:
                self.needs.curiosity -= 15
                self.needs.clamp_all()
            return {"success": False, "error": str(e)}

    def _publish_event(self, event_type: str, data: dict = None):
        """发布事件到事件总线（广播到前端）"""
        if self._event_bus:
            try:
                self._event_bus.publish(event_type, data or {}, source="life_scheduler")
            except Exception as e:
                logger.debug(f"Event publish failed: {e}")

    def _bar(self, value: float) -> str:
        """生成进度条"""
        filled = int(value / 10)
        return "█" * filled + "░" * (10 - filled)

    # ─── 持久化 ─────────────────────────────────────

    def _ensure_data_dir(self):
        """延迟创建数据目录（只在首次写入时创建）"""
        if not self._data_dir_ready:
            os.makedirs(self.data_dir, exist_ok=True)
            self._data_dir_ready = True

    def _save_state(self):
        """保存生命状态"""
        self._ensure_data_dir()
        try:
            state = {
                "needs": self.needs.to_dict(),
                "life_state": self._life_state,
                "total_heartbeats": self._total_heartbeats,
                "last_heartbeat": self._last_heartbeat.isoformat() if self._last_heartbeat else None,
                "last_activity": self._last_activity.isoformat() if self._last_activity else None,
            }
            path = os.path.join(self.data_dir, "life_state.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            # 保存事件日志
            events_path = os.path.join(self.data_dir, "life_events.json")
            recent = self._event_log[-100:]
            events_data = []
            for e in recent:
                events_data.append({
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "trigger": e.trigger,
                    "needs_before": e.needs_before,
                    "needs_after": e.needs_after,
                    "duration_seconds": e.duration_seconds,
                    "success": e.success,
                })
            with open(events_path, "w", encoding="utf-8") as f:
                json.dump(events_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save life state: {e}")

    def _load_state(self):
        """加载生命状态"""
        path = os.path.join(self.data_dir, "life_state.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            needs = state.get("needs", {})
            self.needs.hunger = needs.get("hunger", 30)
            self.needs.fatigue = needs.get("fatigue", 10)
            self.needs.boredom = needs.get("boredom", 20)
            self.needs.stress = needs.get("stress", 10)
            self.needs.curiosity = needs.get("curiosity", 50)
            self._total_heartbeats = state.get("total_heartbeats", 0)
            logger.info(f"Life state loaded: needs={self.needs.to_dict()}")
        except Exception as e:
            logger.warning(f"Failed to load life state: {e}")

        # 加载事件日志
        events_path = os.path.join(self.data_dir, "life_events.json")
        if os.path.exists(events_path):
            try:
                with open(events_path, "r", encoding="utf-8") as f:
                    events_data = json.load(f)
                for item in events_data:
                    self._event_log.append(LifeEvent(**item))
            except Exception as e:
                logger.debug("life_scheduler: non-critical %s", e, exc_info=True)


# ─── 全局实例 ─────────────────────────────────────

_global_life: Optional[LifeScheduler] = None


def get_life_scheduler() -> LifeScheduler:
    """获取全局生命调度器实例"""
    global _global_life
    if _global_life is None:
        _global_life = LifeScheduler()
    return _global_life