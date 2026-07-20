"""
态极睡眠引擎 (Sleep Engine)
============================

态极最独特的能力：睡觉。

就像人脑在睡眠中巩固记忆、修剪突触、整合经验，
态极在用户不活跃时自动进入"睡眠"状态，
整理收集的数据、微调模型、更新用户画像。

睡眠周期：
Phase 1 (浅睡眠): 记忆整理 — 清理 WorkingMemory
Phase 2 (深睡眠): 模型训练 — 用收集的数据在线微调
Phase 3 (REM): 知识整合 — 进化引擎 + 用户画像更新
Phase 4 (清醒): 自我评估 — 检查模型健康状态
Phase 5 (梦境): 进化语料生成 — 态极生成下一代训练数据（递归蒸馏）
"""
import os
import json
import time
import logging
import threading
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger("SleepEngine")

# 神经元架构组件（try/except 守住，避免循环导入）
try:
    from taiji.brain.cortex import Cortex
except ImportError:
    Cortex = None  # type: ignore

try:
    from taiji.resonance.lifecycle import LifecycleManager
except ImportError:
    LifecycleManager = None  # type: ignore

try:
    from taiji.resonance.neuro_modulation import SleepConsolidator, NeuromodulatorState
except ImportError:
    SleepConsolidator = None  # type: ignore
    NeuromodulatorState = None  # type: ignore

try:
    from taiji.resonance.stdp import STDPTracker
except ImportError:
    STDPTracker = None  # type: ignore

try:
    from taiji.resonance.tribal import CoactivationTracker
except ImportError:
    CoactivationTracker = None  # type: ignore

try:
    from taiji.resonance.quality import QualityFilter
except ImportError:
    QualityFilter = None  # type: ignore


@dataclass
class SleepReport:
    """一次睡眠的报告"""
    timestamp: str
    duration_seconds: float
    phases_completed: List[str] = field(default_factory=list)
    memory_entries_cleared: int = 0
    training_samples_used: int = 0
    training_loss: Optional[float] = None
    evolution_events: int = 0
    user_patterns_updated: int = 0
    health_status: str = "unknown"
    recommendations: List[str] = field(default_factory=list)
    # P6-7: 自主进化 encoder 训练统计
    self_evolve_loss: Optional[float] = None
    self_evolve_steps: int = 0


@dataclass
class SleepConfig:
    """睡眠配置"""
    auto_sleep_enabled: bool = True
    sleep_interval_hours: float = 4.0       # 每 4 小时自动睡眠一次
    min_idle_minutes: int = 30               # 空闲 30 分钟后才触发
    max_cpu_percent: float = 80.0            # CPU < 80% 才睡眠
    max_memory_percent: float = 90.0         # 内存 < 90% 才睡眠
    training_enabled: bool = True            # 睡眠时是否训练
    max_training_steps: int = 50             # 睡眠时最大训练步数
    save_checkpoints: bool = True            # 睡眠时保存 checkpoint
    # P6-7: 自主进化 encoder 配置
    self_evolve_enabled: bool = True         # 是否启用自主进化（P6-6）
    self_evolve_steps: int = 30              # 每次 sleep 自主进化训练步数
    self_evolve_lr: float = 1e-4             # 自主进化学习率
    self_evolve_encoder_path: str = "data/distill/shared_context_encoder.pt"
    auto_generation_transition: bool = False  # 代际迁移（需手动开启，默认关闭）


class SleepEngine:
    """
    态极的睡眠引擎
    
    核心理念：
    - 睡眠不是浪费时间，而是成长的关键
    - 就像人脑在睡眠中巩固记忆、整合经验
    - 态极在用户休息时自动整理、学习、进化
    
    睡眠触发条件：
    1. 定时触发（每 N 小时）
    2. 空闲触发（用户超过 M 分钟没有交互）
    3. 手动触发（用户/系统主动调用）
    """
    
    def __init__(self, config: Optional[SleepConfig] = None, data_dir: str = None,
                 model_provider=None, tokenizer_provider=None):
        """
        Args:
            config: 睡眠配置
            data_dir: 数据目录（默认使用外部持久化路径）
            model_provider: 模型获取回调（解耦 core.app_state）
            tokenizer_provider: 分词器获取回调
        """
        self.config = config or SleepConfig()
        if data_dir is None:
            try:
                from taiji.config import get_taiji_data_path
                data_dir = get_taiji_data_path("sleep_data")
            except ImportError:
                data_dir = "taiji/sleep_data"
        self.data_dir = data_dir
        self._model_provider = model_provider
        self._tokenizer_provider = tokenizer_provider
        self._last_sleep_time: Optional[datetime] = None
        self._last_activity_time: Optional[datetime] = None
        self._sleep_history: List[SleepReport] = []
        self._is_sleeping = False
        self._auto_sleep_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # 神经元架构接口（由 set_brain_interfaces 注入）
        self.cortex: Optional[Any] = None  # Cortex 实例
        self._lifecycle: Optional[Any] = None  # LifecycleManager
        self._sleep_consolidator: Optional[Any] = None  # SleepConsolidator
        self._stdp_tracker: Optional[Any] = None  # STDPTracker
        self._coaction: Optional[Any] = None  # CoactivationTracker
        self._feed_engine: Optional[Any] = None  # FeedEngine
        self._recursive_improver: Optional[Any] = None  # RecursiveImprover
        self._quality_filter: Optional[Any] = None  # QualityFilter
        self._neurogenesis_creator: Optional[Any] = None  # NeurogenesisCreator（P2-7）
        # P6-7: 自主进化 encoder + evolver
        self._self_evolver: Optional[Any] = None  # SelfEvolver
        self._context_encoder: Optional[Any] = None  # SharedContextEncoder
        self._current_step: int = 0  # 全局步数计数器（供 SleepConsolidator）
        # P1-2: 神经调质状态（多巴胺/血清素/去甲肾上腺素）
        self._neuromodulator: Optional[Any] = None

        self._data_dir_ready = False
        self._load_history()

        logger.info(f"SleepEngine initialized: auto={self.config.auto_sleep_enabled}, interval={self.config.sleep_interval_hours}h")

    # ─── 神经元架构接口 ───────────────────────────────

    def set_brain_interfaces(
        self,
        cortex: Optional[Any] = None,
        lifecycle: Optional[Any] = None,
        sleep_consolidator: Optional[Any] = None,
        stdp_tracker: Optional[Any] = None,
        coactivation: Optional[Any] = None,
        feed_engine: Optional[Any] = None,
        recursive_improver: Optional[Any] = None,
        quality_filter: Optional[Any] = None,
        neuromodulator: Optional[Any] = None,
    ):
        """
        注入神经元架构组件（Cortex + ResonanceEnsemble 体系）。

        替代旧的 model_provider/tokenizer_provider 模式。
        注入后，睡眠各 Phase 将训练 Cortex 神经元而非单体 ModelSelf。

        Args:
            cortex: Cortex 实例（含 neurons + ensemble）
            lifecycle: LifecycleManager（apoptosis/neurogenesis/maturity）
            sleep_consolidator: SleepConsolidator（睡眠巩固）
            stdp_tracker: STDPTracker（局部学习）
            coactivation: CoactivationTracker（共激活追踪）
            feed_engine: FeedEngine（数据喂养）
            recursive_improver: RecursiveImprover（策略改进）
            quality_filter: QualityFilter（质量过滤）
            neuromodulator: NeuromodulatorState（P1-2，多巴胺/血清素/去甲肾上腺素）
        """
        if cortex is not None:
            self.cortex = cortex
        if lifecycle is not None:
            self._lifecycle = lifecycle
        if sleep_consolidator is not None:
            self._sleep_consolidator = sleep_consolidator
        if stdp_tracker is not None:
            self._stdp_tracker = stdp_tracker
        if coactivation is not None:
            self._coaction = coactivation
        if feed_engine is not None:
            self._feed_engine = feed_engine
        if recursive_improver is not None:
            self._recursive_improver = recursive_improver
        if quality_filter is not None:
            self._quality_filter = quality_filter

        # P1-2: 神经调质状态（若未提供则自动创建默认实例）
        if neuromodulator is not None:
            self._neuromodulator = neuromodulator
        elif not hasattr(self, '_neuromodulator') or self._neuromodulator is None:
            if NeuromodulatorState is not None:
                try:
                    self._neuromodulator = NeuromodulatorState()
                except Exception as e:
                    logger.debug(f"NeuromodulatorState 默认创建失败: {e}")
                    self._neuromodulator = None
            else:
                self._neuromodulator = None

        # P1-2: 将 neuromodulator 注入 cortex.ensemble（驱动 refractory/field_write_scale）
        if self.cortex is not None and self._neuromodulator is not None:
            try:
                self.cortex.set_neuromodulator(self._neuromodulator)
            except Exception as e:
                logger.debug(f"cortex.set_neuromodulator 失败（非关键）: {e}")

        # P2-7: 自动创建 NeurogenesisCreator（若 cortex + lifecycle 可用）
        if self.cortex is not None and self._lifecycle is not None:
            try:
                from taiji.resonance import NeurogenesisCreator
                # 推断 1.5B 教师目录
                teacher_dir = self._infer_teacher_1_5b_dir()
                self._neurogenesis_creator = NeurogenesisCreator(
                    cortex=self.cortex,
                    lifecycle=self._lifecycle,
                    feed_engine=self._feed_engine,
                    teacher_1_5b_dir=teacher_dir,
                    device=getattr(self.cortex, 'device', 'cpu'),
                )
            except Exception as e:
                logger.debug(f"NeurogenesisCreator 创建失败（非关键）: {e}")
                self._neurogenesis_creator = None

        # P0-4 fix (C3): 自动初始化 SelfEvolver（若 cortex 已有 _context_encoder）
        # 原 bug：set_self_evolver 从未被调用，Phase 2.5 永远是死代码
        if self.cortex is not None and self._self_evolver is None:
            ctx_encoder = getattr(self.cortex, '_context_encoder', None)
            if ctx_encoder is not None:
                try:
                    self.set_self_evolver(encoder=ctx_encoder)
                except Exception as e:
                    logger.debug(f"SelfEvolver 自动初始化失败（非关键）: {e}")

        logger.info(
            f"Brain interfaces set: cortex={'✓' if self.cortex else '✗'}, "
            f"lifecycle={'✓' if self._lifecycle else '✗'}, "
            f"sleep_consolidator={'✓' if self._sleep_consolidator else '✗'}, "
            f"stdp={'✓' if self._stdp_tracker else '✗'}, "
            f"coaction={'✓' if self._coaction else '✗'}, "
            f"feed_engine={'✓' if self._feed_engine else '✗'}, "
            f"neurogenesis_creator={'✓' if self._neurogenesis_creator else '✗'}, "
            f"self_evolver={'✓' if self._self_evolver else '✗'}, "
            f"neuromodulator={'✓' if self._neuromodulator else '✗'}"
        )

    def set_self_evolver(
        self,
        evolver: Optional[Any] = None,
        encoder: Optional[Any] = None,
    ) -> None:
        """P6-7: 注入自主进化 encoder + evolver.

        若不传 evolver 但传 encoder，则自动构建 SelfEvolver。
        若都不传，则尝试自动初始化（需要 cortex._shared_embedding 已注册）。

        Args:
            evolver: SelfEvolver 实例（可选）
            encoder: SharedContextEncoder 实例（可选，若 evolver 未提供则用此构建）
        """
        if evolver is not None:
            self._self_evolver = evolver
            self._context_encoder = encoder or getattr(evolver, 'encoder', None)
        elif encoder is not None:
            try:
                from taiji.resonance import (
                    SharedContextEncoder, HebbianUpdater,
                    ContrastiveLoss, MLMLoss, SelfEvolver,
                )
                hebbian = HebbianUpdater(vocab_size=encoder.vocab_size)
                contrastive = ContrastiveLoss()
                mlm = MLMLoss(mask_token_id=4)
                self._self_evolver = SelfEvolver(encoder, hebbian, contrastive, mlm)
                self._context_encoder = encoder
            except Exception as e:
                logger.warning(f"SelfEvolver 自动构建失败: {e}")
                self._self_evolver = None
                self._context_encoder = None
        else:
            logger.warning("set_self_evolver 需要至少提供 evolver 或 encoder")

        logger.info(
            f"Self-evolver set: evolver={'✓' if self._self_evolver else '✗'}, "
            f"encoder={'✓' if self._context_encoder else '✗'}"
        )

    def _infer_teacher_1_5b_dir(self) -> Optional[str]:
        """推断 1.5B gen1 教师模型目录。

        按优先级查找：
        1. 环境变量 TAIJI_TEACHER_1_5B_DIR
        2. 默认路径 e:/taiji/checkpoint-400000
        3. data/teacher_1_5b
        """
        import os
        # 1. 环境变量
        env_dir = os.environ.get('TAIJI_TEACHER_1_5B_DIR')
        if env_dir and os.path.exists(env_dir):
            return env_dir
        # 2. 默认 gen1 路径
        default_path = 'e:/taiji/checkpoint-400000'
        if os.path.exists(default_path):
            return default_path
        # 3. 本地 data 目录
        local_path = 'data/teacher_1_5b'
        if os.path.exists(local_path):
            return local_path
        return None
    
    # ─── 公开接口 ───────────────────────────────────
    
    def sleep(self, reason: str = "manual") -> SleepReport:
        """
        让态极进入睡眠。
        
        Args:
            reason: 睡眠原因（"manual", "auto", "scheduled"）
            
        Returns:
            SleepReport 睡眠报告
        """
        if self._is_sleeping:
            logger.warning("Already sleeping, skipping")
            return SleepReport(timestamp=datetime.now().isoformat(), duration_seconds=0)
        
        self._is_sleeping = True
        start_time = time.time()
        
        logger.info(f"💤 Taiji is going to sleep... (reason: {reason})")
        
        report = SleepReport(
            timestamp=datetime.now().isoformat(),
            duration_seconds=0,
        )
        
        # Phase 1: 浅睡眠 — 记忆整理
        try:
            self._sleep_phase_memory_consolidation(report)
            report.phases_completed.append("memory_consolidation")
            logger.info("  Phase 1: Memory consolidation ✅")
        except Exception as e:
            logger.warning(f"  Phase 1 failed: {e}")
        
        # Phase 2: 深睡眠 — 模型训练
        if self.config.training_enabled:
            try:
                self._sleep_phase_model_training(report)
                report.phases_completed.append("model_training")
                logger.info("  Phase 2: Model training ✅")
            except Exception as e:
                logger.warning(f"  Phase 2 failed: {e}")

        # Phase 2.5 (P6-7): 自主进化 — encoder/embedding 自组织更新
        if self.config.self_evolve_enabled and self._self_evolver is not None:
            try:
                self._sleep_phase_self_evolve(report)
                report.phases_completed.append("self_evolve")
                logger.info("  Phase 2.5: Self-evolve (P6-7) ✅")
            except Exception as e:
                logger.warning(f"  Phase 2.5 failed: {e}")

        # Phase 3: REM — 知识整合
        try:
            self._sleep_phase_knowledge_integration(report)
            report.phases_completed.append("knowledge_integration")
            logger.info("  Phase 3: Knowledge integration ✅")
        except Exception as e:
            logger.warning(f"  Phase 3 failed: {e}")

        # Phase 3.5: 知识蒸馏 — 将累积知识转化为训练数据
        try:
            self._sleep_phase_knowledge_distillation(report)
            report.phases_completed.append("knowledge_distillation")
            logger.info("  Phase 3.5: Knowledge distillation ✅")
        except Exception as e:
            logger.warning(f"  Phase 3.5 failed: {e}")
        
        # Phase 4: 清醒准备 — 自我评估
        try:
            health = self._sleep_phase_evaluation(report)
            report.health_status = health.get("status", "unknown")
            report.phases_completed.append("evaluation")
            logger.info("  Phase 4: Evaluation ✅")
        except Exception as e:
            logger.warning(f"  Phase 4 failed: {e}")

        # Phase 5: 梦境 — 递归改进（策略优化 + 进化语料生成）
        try:
            self._sleep_phase_recursive_improvement(report)
            report.phases_completed.append("recursive_improvement")
            logger.info("  Phase 5: Recursive improvement ✅")
        except Exception as e:
            logger.warning(f"  Phase 5 failed: {e}")
        
        # P0-4 fix (C1): 所有 Phase 完成后统一清空 feed_engine 样本
        # （Phase 2 和 Phase 2.5 共享同一批样本，之前 Phase 2 清空导致 Phase 2.5 无数据）
        if self._feed_engine is not None:
            try:
                self._feed_engine.clear_pending_samples()
            except Exception as e:
                logger.debug(f"  最终清空样本失败: {e}")

        # 计算睡眠时长
        report.duration_seconds = round(time.time() - start_time, 1)
        self._last_sleep_time = datetime.now()
        self._is_sleeping = False

        # 保存报告
        self._sleep_history.append(report)
        self._save_history()

        logger.info(f"⏰ Taiji woke up! Duration: {report.duration_seconds}s, Phases: {len(report.phases_completed)}")

        return report
    
    def wake(self):
        """唤醒态极"""
        self._is_sleeping = False
        logger.info("☀️ Taiji is awake!")
    
    def record_activity(self):
        """记录用户活动（用于判断是否空闲）"""
        self._last_activity_time = datetime.now()

    def nap(self, duration_minutes: int = 2):
        """Deep Coupling: 短睡——快速消化新知识。

        由 FeedEngine 喂食完成后通过 EventBus 触发。
        只跑 Phase 2（微调），不跑完整的 6 阶段。
        """
        if self._is_sleeping:
            return
        self._is_sleeping = True
        report = SleepReport()
        try:
            logger.info(f"Nap: {duration_minutes}min 短睡消化...")
            self._sleep_phase_model_training(report)
            logger.info(f"Nap complete: loss={report.training_loss}")
        except Exception as e:
            logger.debug(f"Nap failed: {e}")
        finally:
            self._is_sleeping = False
            self._last_sleep_time = time.time()
    
    def start_auto_sleep(self):
        """启动自动睡眠线程"""
        if not self.config.auto_sleep_enabled:
            return
        
        if self._auto_sleep_thread and self._auto_sleep_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._auto_sleep_thread = threading.Thread(target=self._auto_sleep_loop, daemon=True)
        self._auto_sleep_thread.start()
        logger.info("Auto-sleep thread started")
    
    def stop_auto_sleep(self):
        """停止自动睡眠"""
        self._stop_event.set()
        if self._auto_sleep_thread:
            self._auto_sleep_thread.join(timeout=5)
        logger.info("Auto-sleep thread stopped")
    
    def _auto_sleep_loop(self):
        """自动睡眠循环"""
        while not self._stop_event.is_set():
            time.sleep(60)  # 每分钟检查一次
            
            if self._should_auto_sleep():
                self.sleep(reason="auto")
    
    def _should_auto_sleep(self) -> bool:
        """检查是否应该自动睡眠"""
        if self._is_sleeping:
            return False
        
        # 检查距上次睡眠的时间
        if self._last_sleep_time:
            hours_since_last = (datetime.now() - self._last_sleep_time).total_seconds() / 3600
            if hours_since_last < self.config.sleep_interval_hours:
                return False
        
        # 检查空闲时间
        if self._last_activity_time:
            idle_minutes = (datetime.now() - self._last_activity_time).total_seconds() / 60
            if idle_minutes < self.config.min_idle_minutes:
                return False
        
        return True
    
    # ─── 睡眠阶段实现 ──────────────────────────────
    
    def _sleep_phase_memory_consolidation(self, report: SleepReport):
        """Phase 1: 记忆整理 — 整合上下文管理器 + WorkingMemory"""
        try:
            # 整合上下文管理器
            from taiji.agent.context_manager import get_context_manager
            ctx = get_context_manager()
            ctx.consolidate_for_sleep()
            logger.info("  ContextManager consolidated")
        except Exception as e:
            logger.debug(f"  ContextManager consolidation skipped: {e}")

        try:
            from taiji.agent.working_memory import get_working_memory
            wm = get_working_memory()

            modified = wm.get_modified_keys()
            report.memory_entries_cleared = len(modified)

            if modified:
                logger.info(f"  Consolidating {len(modified)} modified memory entries")

            # 导出修改过的内容
            for key in modified:
                content = wm.export(key)
                if content:
                    safe_name = key.replace("/", "_").replace("\\", "_")
                    save_path = os.path.join(self.data_dir, f"memory_{safe_name}.txt")
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(content)

            # 清理工作记忆
            wm.clear()
            logger.info("  Working memory cleared")

        except ImportError:
            logger.info("  WorkingMemory not available, skipping")
    
    def _sleep_phase_model_training(self, report: SleepReport):
        """Phase 2: 神经元训练 - 用收集的数据训练 Cortex 神经元（低秩残差，W_base 冻结）"""
        # 神经元架构路径：训练 Cortex 神经元
        if self.cortex is not None and hasattr(self.cortex, 'neurons') and self.cortex.neurons:
            self._train_cortex_neurons(report)
            return

        # 回退路径：旧单体模型训练（向后兼容，cortex 未注入时使用）
        logger.warning("Cortex 未注入，回退到旧单体模型训练路径")
        try:
            from taiji.agent_ext.data_collector import get_collector
            from taiji.data.data_generator import generate_bulk_react_data, generate_bulk_conversation_data
            logger.warning("taiji.agent_ext/taiji.data 模块用于旧模型回退路径")
            
            collector = get_collector()
            react_data, conv_data = collector.load_as_training_data()

            # B5 修复：加载弱项针对性训练数据
            try:
                weakness_dir = os.path.join(self.data_dir, "weakness_training_data")
                if os.path.isdir(weakness_dir):
                    for fname in os.listdir(weakness_dir):
                        if fname.endswith(".json"):
                            with open(os.path.join(weakness_dir, fname), "r", encoding="utf-8") as f:
                                weakness_samples = json.load(f)
                            if isinstance(weakness_samples, list):
                                for ws in weakness_samples:
                                    if ws.get("type") == "react":
                                        react_data.append(ws)
                                    elif ws.get("type") == "conversation":
                                        conv_data.append(ws)
                            logger.info(f"  Loaded {len(weakness_samples) if isinstance(weakness_samples, list) else 0} weakness-training samples from {fname}")
            except Exception as e:
                logger.debug(f"  Weakness data load skipped: {e}")
            
            report.training_samples_used = len(react_data) + len(conv_data)
            
            if not react_data and not conv_data:
                logger.info("  No new training data, using generated data")
                react_data = generate_bulk_react_data(50)
                conv_data = generate_bulk_conversation_data(20)
                report.training_samples_used = len(react_data) + len(conv_data)
            
            logger.info(f"  Training with {report.training_samples_used} samples")
            
            # 保存训练数据供后续使用
            train_data_path = os.path.join(self.data_dir, "sleep_training_data.jsonl")
            with open(train_data_path, "w", encoding="utf-8") as f:
                for item in react_data:
                    f.write(json.dumps({"type": "react", **item}, ensure_ascii=False) + "\n")
                for item in conv_data:
                    f.write(json.dumps({"type": "conversation", **item}, ensure_ascii=False) + "\n")
            
            # 从喂养引擎获取待消化的训练样本
            try:
                from taiji.life.feed_engine import get_feed_engine
                feed_engine = get_feed_engine()
                pending_samples = feed_engine.get_pending_samples()
                if pending_samples:
                    logger.info(f"  Got {len(pending_samples)} pending samples from feed engine")
                    # 将喂养引擎的样本也加入训练
                    for sample in pending_samples:
                        if sample.get("type") == "react":
                            react_data.append(sample)
                        elif sample.get("type") == "conversation":
                            conv_data.append(sample)
                    report.training_samples_used += len(pending_samples)
            except Exception as e:
                logger.debug(f"  Feed engine integration skipped: {e}")

            # 实际调用态极训练器进行在线微调
            training_loss = self._run_sleep_training(react_data, conv_data)
            if training_loss is not None:
                report.training_loss = training_loss
                logger.info(f"  Sleep training completed, loss={training_loss:.4f}")
            
            # 标记数据已使用
            collector.flush()

            # 清除喂养引擎已消化的样本
            try:
                from taiji.life.feed_engine import get_feed_engine
                feed_engine = get_feed_engine()
                feed_engine.clear_pending_samples()
                logger.info("  Feed engine pending samples cleared")
            except Exception as e:
                logger.debug("sleep_engine: non-critical %s", e, exc_info=True)
            
        except ImportError:
            logger.warning("  DataCollector not available, skipping legacy training")

    def _train_cortex_neurons(self, report: SleepReport):
        """
        神经元架构：训练 Cortex 中每个域的神经元。

        核心流程：
        1. 从 feed_engine.get_pending_samples_by_domain() 获取按域分类的样本
        2. 对每个 domain 的 neuron 训练 lm_head_delta_u/v（低秩残差），W_base 冻结
        3. 记录 PPL 到 lifecycle.apoptosis.record_ppl
        4. 调用 stdp_tracker.apply_all_updates(cortex.neurons) 应用 STDP
        5. 检查 lifecycle.neurogenesis 触发条件
        """
        # 获取按域分类的训练样本
        domain_samples: Dict[str, list] = {}
        if self._feed_engine is not None:
            domain_samples = self._feed_engine.get_pending_samples_by_domain()
        else:
            try:
                from taiji.life.feed_engine import get_feed_engine
                self._feed_engine = get_feed_engine()
                domain_samples = self._feed_engine.get_pending_samples_by_domain()
            except Exception as e:
                logger.warning(f"  FeedEngine 不可用: {e}")

        total_samples = sum(len(s) for s in domain_samples.values())
        report.training_samples_used = total_samples
        logger.info(f"  Cortex 训练: {len(domain_samples)} 个域, {total_samples} 条样本")

        if not domain_samples:
            logger.info("  无训练样本，跳过 Cortex 训练")
            return

        # 获取共享 embedding 和 tokenizer（从 Cortex 提取）
        shared_embedding = getattr(self.cortex, '_shared_embedding', None)
        tokenizer = getattr(self.cortex, '_tokenizer', None)
        embed_pipeline = getattr(self.cortex, '_embed_pipeline', None)

        if shared_embedding is None and embed_pipeline is None:
            logger.warning("  Cortex 未设置 shared_embedding/embed_pipeline，跳过训练")
            return

        ppl_results: Dict[str, float] = {}
        total_loss = 0.0
        trained_count = 0

        for domain, samples in domain_samples.items():
            # 找到对应域的神经元
            neuron = self.cortex.neurons.get(domain)
            if neuron is None:
                logger.debug(f"  域 '{domain}' 无对应神经元，跳过")
                continue

            if not samples:
                continue

            # 训练该神经元的低秩残差（W_base 冻结）
            avg_loss, ppl = self._train_single_neuron(
                neuron, domain, samples, shared_embedding, embed_pipeline, tokenizer
            )

            if avg_loss is not None:
                total_loss += avg_loss
                trained_count += 1
                ppl_results[domain] = ppl
                logger.info(f"  域 '{domain}' 训练完成: loss={avg_loss:.4f}, PPL={ppl:.1f}")

                # 记录 PPL 到凋亡追踪器
                if self._lifecycle is not None:
                    try:
                        self._lifecycle.apoptosis.record_ppl(domain, ppl)
                    except Exception as e:
                        logger.debug(f"  apoptosis.record_ppl 失败: {e}")

        # 应用 STDP 更新（局部学习规则）
        if self._stdp_tracker is not None:
            try:
                updates = self._stdp_tracker.apply_all_updates(self.cortex.neurons)
                if updates:
                    logger.info(f"  STDP 更新: {len(updates)} 个神经元")
            except Exception as e:
                logger.warning(f"  STDP 更新失败: {e}")

        # 检查 neurogenesis 触发条件
        if self._lifecycle is not None and self._feed_engine is not None:
            try:
                error_rates = self._feed_engine.get_domain_error_rates()
                for domain, error_rate in error_rates.items():
                    triggered = self._lifecycle.neurogenesis.record_domain_error(domain, error_rate)
                    if triggered:
                        logger.info(f"  域 '{domain}' 触发 neurogenesis（错误率 {error_rate:.0%}）")
                        report.recommendations.append(
                            f"[神经新生] 域 '{domain}' 错误率过高，建议创建新神经元"
                        )
                        # P2-7: 实际创建新神经元（分场景教师选择）
                        if self._neurogenesis_creator is not None:
                            try:
                                result = self._neurogenesis_creator.create_neuron_for_domain(domain)
                                if result.get("success"):
                                    logger.info(
                                        f"  [神经新生] 域 '{domain}' 新神经元已创建: "
                                        f"id='{result.get('neuron_id')}', "
                                        f"教师类型='{result.get('teacher_type')}'"
                                    )
                                    report.recommendations.append(
                                        f"[神经新生完成] 新神经元 id='{result.get('neuron_id')}'，"
                                        f"教师={result.get('teacher_type')}"
                                    )
                                else:
                                    logger.warning(
                                        f"  [神经新生] 域 '{domain}' 创建失败: "
                                        f"{result.get('error')}"
                                    )
                            except Exception as e:
                                logger.error(f"  [神经新生] 创建异常: {e}", exc_info=True)
            except Exception as e:
                logger.debug(f"  neurogenesis 检查失败: {e}")

        # 递增成熟度
        if self._lifecycle is not None:
            try:
                self._lifecycle.maturity.tick_all()
            except Exception as e:
                logger.debug(f"  maturity.tick_all 失败: {e}")

        # 记录训练损失
        if trained_count > 0:
            report.training_loss = total_loss / trained_count

        # P0-4 fix (C1): 不在此清空样本 — Phase 2.5 (self-evolve) 也要用这批样本
        # 清空操作移到 sleep() 主流程末尾统一执行
        # （原 bug：Phase 2 清空后 Phase 2.5 拿不到数据）

        # 步数递增
        self._current_step += 1

    def _sleep_phase_self_evolve(self, report: SleepReport):
        """P6-7: 自主进化阶段 — 在 sleep cycle 中训练 SharedContextEncoder.

        流程：
        1. 从 feed_engine 收集训练样本（按 domain 分组，用于 contrastive 弱监督）
        2. 每个 step: sample batch → evolver.training_step → backward → optimizer.step
        3. 定期 apply_hebbian_to_embedding（离线 Hebbian 更新）
        4. 训练完成后保存 encoder 到磁盘

        三机制训练：
        - MLM: 自监督主信号（mask 15% token 预测）
        - Contrastive: 用 domain 作弱监督（同 domain 拉近）
        - Hebbian: 离线更新 embedding（共激活 token 拉近）
        """
        if self._self_evolver is None:
            logger.info("  Self-evolver 未注入，跳过")
            return

        # 1. 收集训练样本
        domain_samples: Dict[str, list] = {}
        if self._feed_engine is not None:
            try:
                domain_samples = self._feed_engine.get_pending_samples_by_domain()
            except Exception as e:
                logger.warning(f"  FeedEngine 获取样本失败: {e}")

        if not domain_samples and self.cortex is not None:
            # 回退：尝试从 domain_datasets.pt 加载（蒸馏数据）
            try:
                import torch
                domain_data = torch.load(
                    "data/distill/domain_datasets.pt",
                    map_location="cpu", weights_only=False,
                )
                tokenizer = getattr(self.cortex, '_tokenizer', None)
                if tokenizer is not None:
                    for nid, ids_tensor in domain_data.items():
                        # 取前 50 个 sample 的 token ids 转 text
                        ids_list = ids_tensor[:50].tolist()
                        domain_samples[nid] = [
                            tokenizer.encode(tokenizer.decode(ids)) if False else ids
                            for ids in ids_list
                        ]
            except Exception as e:
                logger.debug(f"  domain_datasets.pt 加载失败: {e}")

        total_samples = sum(len(s) for s in domain_samples.values())
        if total_samples == 0:
            logger.info("  无训练样本，跳过 self-evolve")
            return

        logger.info(f"  Self-evolve: {len(domain_samples)} 个域, {total_samples} 条样本")

        # 2. 准备训练数据：flatten 成 (input_ids, domain_id) 对
        import torch
        tokenizer = getattr(self.cortex, '_tokenizer', None)
        domain_to_id: Dict[str, int] = {
            domain: i for i, domain in enumerate(domain_samples.keys())
        }

        # 把样本转成 token ids（若 samples 已是 ids 则直接用）
        # P0-4 fix (C2): 支持 feed_engine 的 dict 样本格式
        all_samples: List[tuple] = []  # [(input_ids, domain_id), ...]
        for domain, samples in domain_samples.items():
            did = domain_to_id[domain]
            for s in samples:
                ids = None
                if isinstance(s, str) and tokenizer is not None:
                    ids = tokenizer.encode(s)
                elif isinstance(s, list):
                    ids = s
                elif hasattr(s, 'tolist'):
                    ids = s.tolist()
                elif isinstance(s, dict) and tokenizer is not None:
                    # P0-4 fix (C2): feed_engine 的 dict 样本
                    # 提取文本：优先 task/content/messages
                    text = (
                        s.get('task') or s.get('content')
                        or s.get('text') or s.get('prompt')
                    )
                    if text is None and 'messages' in s:
                        # 从 messages 列表提取 user/assistant 文本
                        msgs = s['messages']
                        if isinstance(msgs, list):
                            text = ' '.join(
                                m.get('content', '') for m in msgs
                                if isinstance(m, dict)
                            )
                    if text:
                        ids = tokenizer.encode(text)
                else:
                    continue
                if isinstance(ids, list) and len(ids) > 0:
                    all_samples.append((ids, did))

        if not all_samples:
            logger.warning("  无有效训练样本（tokenize 失败）")
            return

        # 3. 训练循环
        encoder = self._context_encoder
        device = next(encoder.parameters()).device
        encoder.train()
        optimizer = torch.optim.AdamW(
            encoder.parameters(), lr=self.config.self_evolve_lr,
        )

        max_steps = min(self.config.self_evolve_steps, len(all_samples))
        batch_size = 4
        seq_len = 64  # 截断到 64 token 避免 OOM
        total_loss = 0.0
        hebbian_apply_interval = 10  # 每 10 步应用一次 Hebbian

        import random
        random.shuffle(all_samples)

        for step in range(max_steps):
            # 采样 batch
            batch_samples = all_samples[step * batch_size:(step + 1) * batch_size]
            if len(batch_samples) < 2:
                continue  # contrastive 至少需要 2 个样本

            # 构造 batch tensor
            batch_ids = []
            domain_labels = []
            for ids, did in batch_samples:
                # 截断或 padding 到 seq_len
                truncated = ids[:seq_len]
                if len(truncated) < seq_len:
                    truncated = truncated + [0] * (seq_len - len(truncated))
                batch_ids.append(truncated)
                domain_labels.append(did)

            input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
            domain_tensor = torch.tensor(domain_labels, dtype=torch.long, device=device)

            # training_step
            optimizer.zero_grad()
            loss = self._self_evolver.training_step(
                batch={"input_ids": input_ids},
                domain_labels=domain_tensor,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())

            # 定期 Hebbian 更新
            if (step + 1) % hebbian_apply_interval == 0:
                with torch.no_grad():
                    self._self_evolver.apply_hebbian_to_embedding()

            if (step + 1) % 5 == 0:
                avg = total_loss / (step + 1)
                logger.info(f"  self-evolve step {step+1}/{max_steps}, loss={avg:.4f}")

        encoder.eval()

        # 4. 保存 encoder
        try:
            encoder.save(self.config.self_evolve_encoder_path)
        except Exception as e:
            logger.warning(f"  encoder 保存失败: {e}")

        # 5. 记录到 report
        avg_loss = total_loss / max(max_steps, 1)
        report.self_evolve_loss = avg_loss
        report.self_evolve_steps = max_steps

        # 6. 记录 evolver 的 loss 分解（诊断）
        summary = self._self_evolver.get_loss_summary()
        logger.info(
            f"  Self-evolve 完成: avg_loss={avg_loss:.4f}, "
            f"final_mlm={summary.get('mlm', 0):.4f}, "
            f"final_contrastive={summary.get('contrastive', 0):.4f}, "
            f"hebbian_update={summary.get('hebbian', 0):.6f}"
        )

    def _train_single_neuron(self, neuron, domain: str, samples: list,
                             shared_embedding, embed_pipeline, tokenizer) -> tuple:
        """
        训练单个 Cortex 神经元的低秩残差（lm_head_delta_u/v），W_base 冻结。

        Returns:
            (avg_loss, ppl) 元组，失败时为 (None, None)
        """
        import torch
        import torch.nn.functional as F

        # 收集可训练参数：仅低秩残差 u/v
        trainable_params = []
        if hasattr(neuron, 'lm_head_delta_u'):
            trainable_params.extend(
                p for p in neuron.lm_head_delta_u.parameters() if p.requires_grad
            )
        if hasattr(neuron, 'lm_head_delta_v'):
            trainable_params.extend(
                p for p in neuron.lm_head_delta_v.parameters() if p.requires_grad
            )

        # 传统模式（lm_head_rank=0）：回退到训练完整 lm_head
        if not trainable_params and hasattr(neuron, 'lm_head'):
            trainable_params = [p for p in neuron.lm_head.parameters() if p.requires_grad]

        if not trainable_params:
            logger.debug(f"  神经元 '{domain}' 无可训练参数")
            return None, None

        # 准备训练文本
        texts = []
        for sample in samples:
            if isinstance(sample, dict):
                text = sample.get("task") or sample.get("content") or ""
                if sample.get("messages"):
                    text = " ".join(
                        m.get("content", "") for m in sample["messages"]
                        if m.get("role") != "system"
                    )
                if text and len(text.strip()) > 5:
                    texts.append(text.strip())

        if not texts:
            return None, None

        device = next(neuron.parameters()).device
        # P1-2: 神经调质调整学习率（多巴胺驱动）
        # 高多巴胺 → lr 倍数↑（奖励信号，强化学习）
        # 低多巴胺 → lr 倍数↓（错误信号，保守更新）
        base_lr = 5e-5
        lr_mult = 1.0
        if getattr(self, '_neuromodulator', None) is not None:
            try:
                lr_mult = self._neuromodulator.get_lr_multiplier()
            except Exception:
                pass
        optimizer = torch.optim.AdamW(
            trainable_params, lr=base_lr * lr_mult, weight_decay=0.01
        )

        max_steps = min(self.config.max_training_steps, len(texts))
        neuron.train()
        total_loss = 0.0
        step_count = 0
        loss_history = []

        for i in range(max_steps):
            text = texts[i % len(texts)]
            if tokenizer is None:
                break

            # 编码
            try:
                ids = tokenizer.encode(text)
            except Exception:
                continue
            if len(ids) < 5:
                continue
            ids = ids[:512]

            input_ids = torch.tensor([ids], dtype=torch.long, device=device)

            # 获取共享 embedding
            with torch.no_grad():
                if shared_embedding is not None:
                    shared_emb = shared_embedding(input_ids)
                elif embed_pipeline is not None:
                    shared_emb = embed_pipeline(input_ids)
                else:
                    continue

            # 前向传播（带 logits）
            output = neuron.forward(shared_emb, return_logits=True)
            logits = output.get("logits")
            if logits is None:
                continue

            # 计算损失（next token prediction）
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
            )

            if loss.item() == 0:
                continue

            # 反向传播
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            final_loss = loss.item()
            total_loss += final_loss
            loss_history.append(final_loss)
            step_count += 1

            if step_count % 10 == 0:
                logger.info(f"  [{domain}] 训练 step {step_count}/{max_steps}, loss={final_loss:.4f}")

            # 早停：loss 连续 5 步上升
            if len(loss_history) >= 7:
                recent = loss_history[-5:]
                if all(recent[j] > recent[j-1] for j in range(1, len(recent))):
                    logger.info(f"  [{domain}] 早停：loss 连续上升，step={step_count}")
                    break

        neuron.eval()

        if step_count == 0:
            return None, None

        avg_loss = total_loss / step_count
        ppl = math.exp(avg_loss) if avg_loss < 20 else 999.0
        return avg_loss, ppl

    def _run_sleep_training(self, react_data: list, conv_data: list):
        """
        执行睡眠训练：在线微调态极 ModelSelf 模型

        直接使用 PyTorch 进行轻量级微调，限制步数以控制睡眠时长。
        通过注入的 model_provider/tokenizer_provider 获取模型。
        """
        import torch
        import torch.nn.functional as F
        _app_state = None
        try:
            # 获取训练锁，避免与其他训练进程并发操作同一模型
            try:
                from taiji.core.app_state import app_state
                _app_state = app_state
                if hasattr(app_state, 'try_start_training'):
                    if not app_state.try_start_training():
                        logger.info("  其他训练正在进行，跳过睡眠训练以避免权重冲突")
                        return None
            except ImportError:
                logger.debug("  taiji.core.app_state 不可用")
            except Exception as e:
                logger.debug(f"  训练锁获取失败: {e}")

            # 优先使用注入的 provider（解耦 core.app_state）
            model = None
            tokenizer = None
            if self._model_provider:
                try:
                    model = self._model_provider()
                except Exception as e:
                    logger.debug(f"  model_provider 调用失败: {e}")
            if self._tokenizer_provider:
                try:
                    tokenizer = self._tokenizer_provider()
                except Exception as e:
                    logger.debug(f"  tokenizer_provider 调用失败: {e}")

            # 回退：尝试从 taiji.core.app_state 获取
            if model is None or tokenizer is None:
                try:
                    from taiji.core.app_state import app_state
                    if model is None and hasattr(app_state, 'model'):
                        model = app_state.model
                    if tokenizer is None and hasattr(app_state, 'tokenizer'):
                        tokenizer = app_state.tokenizer
                except ImportError:
                    pass

            if model is None or tokenizer is None:
                logger.info("  No model available, skipping training")
                return None

            # P2-6: 判断是否为态极模型
            # Cortex 模式下不执行传统 ModelSelf 睡眠训练（由 _train_cortex_neurons 负责）
            try:
                from taiji.architecture import ModelSelf
            except ImportError:
                ModelSelf = None  # ModelSelf 已完全移除

            if ModelSelf is None or not isinstance(model, ModelSelf):
                if type(model).__name__ == 'Cortex':
                    logger.info("  Cortex 模式：传统睡眠训练已跳过（由 _train_cortex_neurons 负责）")
                else:
                    logger.info("  Current model is not ModelSelf, skipping sleep training")
                return None

            # 合并训练数据
            all_texts = []
            for item in react_data:
                task = item.get("task", "")
                if task:
                    all_texts.append(task)
            for item in conv_data:
                if isinstance(item, dict) and "messages" in item:
                    text = " ".join(m.get("content", "") for m in item["messages"] if m.get("role") != "system")
                    if text.strip():
                        all_texts.append(text)

            if not all_texts:
                logger.info("  No valid training texts, skipping")
                return None

            # 限制训练步数（睡眠时轻量训练）
            max_steps = min(self.config.max_training_steps, len(all_texts))
            device = next(model.parameters()).device

            # 创建优化器（睡眠时用较小学习率，避免灾难性遗忘）
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            if hasattr(model, '_kv_cache'):
                model._kv_cache = None
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            model.train()

            logger.info(f"  Starting sleep training: {max_steps} steps on {device}")

            final_loss = None
            step_count = 0
            loss_history = []

            for i in range(max_steps):
                text = all_texts[i % len(all_texts)]

                # 编码
                ids = tokenizer.encode(text)
                if len(ids) < 5:
                    continue
                ids = ids[:512]

                # 构建 input_ids 和 labels（自回归）
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                labels = input_ids.clone()

                # 前向
                output = model(input_ids, targets=labels)
                loss = output.loss

                if loss is None or loss.item() == 0:
                    continue

                # 反向
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                final_loss = loss.item()
                loss_history.append(final_loss)
                step_count += 1

                if step_count % 10 == 0:
                    logger.info(f"  Sleep training step {step_count}/{max_steps}, loss={final_loss:.4f}")

                # 早停：如果 loss 连续 5 步上升，停止训练（防止灾难性遗忘）
                if len(loss_history) >= 7:
                    recent = loss_history[-5:]
                    if all(recent[j] > recent[j-1] for j in range(1, len(recent))):
                        logger.info(f"  早停：loss 连续上升，step={step_count}")
                        break

            model.eval()

            # 训练后验证：简单对话测试
            if final_loss is not None:
                self._validate_training_effect(model, tokenizer, device, final_loss, loss_history)

            # 训练后保存 checkpoint
            if self.config.save_checkpoints and final_loss is not None:
                try:
                    from taiji.loader import save_model
                    checkpoint_dir = None
                    try:
                        from taiji.core.app_state import app_state
                        model_path = getattr(app_state, "_loaded_model_name", "") or ""
                        if model_path and os.path.isdir(model_path):
                            checkpoint_dir = model_path
                    except Exception as e:
                        logger.debug("sleep_engine: non-critical %s", e, exc_info=True)
                    if not checkpoint_dir:
                        checkpoint_dir = os.path.join(self.data_dir, "checkpoints")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    save_model(model, tokenizer, checkpoint_dir)
                    logger.info(f"  Checkpoint saved to {checkpoint_dir}")
                except Exception as e:
                    logger.warning(f"  Checkpoint save failed: {e}")

            return final_loss

        except Exception as e:
            logger.warning(f"  Sleep training failed: {e}")
            return None
        finally:
            if _app_state is not None:
                try:
                    if hasattr(_app_state, 'finish_training'):
                        _app_state.finish_training()
                    # B6 修复：训练后将新模型热加载到推理服务
                    if final_loss is not None and model is not None:
                        _app_state.update_model(model, tokenizer, trainer=None, model_name="sleep-trained")
                        logger.info("  训练后模型已热加载到推理服务")
                except Exception as e:
                    logger.debug("sleep_engine: non-critical %s", e, exc_info=True)

    def _validate_training_effect(self, model, tokenizer, device,
                                   final_loss: float, loss_history: list):
        """
        训练后验证：检查是否发生灾难性遗忘。

        1. 检查 loss 是否从初始值下降
        2. 尝试简单生成测试（如果 tokenizer 支持）
        """
        # 检查 loss 趋势
        if len(loss_history) >= 2:
            initial_loss = loss_history[0]
            if final_loss > initial_loss * 2:
                logger.warning(
                    f"  ⚠️ 训练后 loss ({final_loss:.4f}) 比初始 ({initial_loss:.4f}) 高 2x 以上，"
                    f"可能发生灾难性遗忘"
                )
            elif final_loss < initial_loss * 0.5:
                logger.info(f"  ✅ 训练效果良好: loss {initial_loss:.4f} → {final_loss:.4f}")

        # 简单生成测试
        try:
            test_prompt = "[用户] 你好\n[助手]"
            ids = tokenizer.encode(test_prompt)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                output = model(input_ids)
                logits = output.logits[:, -1, :]
                top_tokens = torch.topk(logits, 5, dim=-1).indices[0].tolist()
            logger.info(f"  训练后生成测试: top-5 token IDs = {top_tokens}")
        except Exception as e:
            logger.debug(f"  生成测试跳过: {e}")
    
    def _sleep_phase_knowledge_integration(self, report: SleepReport):
        """Phase 3: 知识整合 - 睡眠巩固（SleepConsolidator）+ 进化引擎 + 用户画像"""
        # 神经元架构：调用 SleepConsolidator 执行睡眠巩固
        if self._sleep_consolidator is not None and self.cortex is not None:
            try:
                stats = self._sleep_consolidator.consolidate(
                    self.cortex.neurons,
                    coactivation_tracker=self._coaction,
                    current_step=self._current_step,
                )
                logger.info(f"  睡眠巩固完成: {stats}")
                report.user_patterns_updated = stats.get("channels_reinforced", 0)
            except Exception as e:
                logger.warning(f"  睡眠巩固失败: {e}")
        else:
            logger.debug("  SleepConsolidator 或 Cortex 未注入，跳过睡眠巩固")

        # 进化引擎（向后兼容）
        try:
            from taiji.life.evolution_engine import get_evolution_engine
            engine = get_evolution_engine()

            # 将睡眠训练结果同步到进化引擎
            if report.training_loss is not None:
                engine.record_sleep_training(
                    loss=report.training_loss,
                    samples=report.training_samples_used,
                )

            if engine.metrics.tasks_completed > 0:
                report.evolution_events = engine.metrics.evolution_cycles

                # 检查是否需要触发进化
                total = engine.metrics.tasks_completed + engine.metrics.tasks_failed
                if total > 0 and total % 50 == 0:
                    engine._trigger_evolution("sleep_cycle")
                    report.evolution_events += 1

        except ImportError:
            logger.info("  EvolutionEngine not available, skipping")
        
        # 用户画像（向后兼容）
        try:
            from taiji.infra.user_profile import get_user_profile
            logger.warning("taiji.infra.user_profile 模块用于旧路径")
            profile = get_user_profile()
            
            suggestions = profile.get_task_pattern_suggestions()
            report.user_patterns_updated = len(suggestions)
            
            if suggestions:
                report.recommendations.extend(suggestions)
                
        except ImportError:
            logger.info("  UserProfile not available, skipping")

    def _sleep_phase_knowledge_distillation(self, report: SleepReport):
        """Phase 3.5: 知识蒸馏 - 累积知识 -> 训练数据"""
        try:
            from taiji.agent_ext.knowledge_learner import get_knowledge_learner
            logger.warning("taiji.agent_ext.knowledge_learner 模块用于旧路径（神经元架构使用 feed_engine 替代）")
            learner = get_knowledge_learner()
            model = self._get_model()
            tokenizer = self._get_tokenizer()
            result = learner.maybe_distill_to_training_data(model=model, tokenizer=tokenizer)
            if result.get("distilled"):
                count = result.get("samples", 0)
                report.recommendations.append(f"[蒸馏] 知识→训练数据: 生成 {count} 条样本")
                logger.info(f"  Knowledge distilled: {count} training samples generated")
            else:
                logger.debug(f"  Knowledge distillation skipped: {result.get('reason', '')}")
        except ImportError:
            logger.info("  KnowledgeLearner not available, skipping distillation")
        except Exception as e:
            logger.warning(f"  Knowledge distillation failed: {e}")
    
    def _sleep_phase_evaluation(self, report: SleepReport) -> dict:
        """Phase 4: 自我评估 - 用 QualityFilter 评估神经元质量 + 调用 apoptosis.check_activation"""
        # 神经元架构：用 QualityFilter 评估 + apoptosis 检查
        if self.cortex is not None and hasattr(self.cortex, 'neurons') and self.cortex.neurons:
            health = self._evaluate_cortex_quality(report)
            # 保存健康报告
            health_path = os.path.join(self.data_dir, "health_report.json")
            try:
                with open(health_path, "w", encoding="utf-8") as f:
                    json.dump(health, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"  保存健康报告失败: {e}")
            return health

        # 回退路径：旧自我评估器（向后兼容）
        try:
            from taiji.infra.self_evaluator import get_self_evaluator
            from taiji.life.evolution_engine import get_evolution_engine
            logger.warning("taiji.infra.self_evaluator 模块用于旧路径")

            evaluator = get_self_evaluator()
            engine = get_evolution_engine()

            stats = evaluator.get_stats()
            trends = evaluator.get_improvement_trends()

            health = {
                "phase": engine.metrics.current_phase,
                "tasks_completed": engine.metrics.tasks_completed,
                "evaluation_count": stats.get("total_evaluations", 0),
                "avg_score": stats.get("avg_score", 0),
                "trends": trends,
                "status": "healthy",
            }

            # 保存健康报告
            health_path = os.path.join(self.data_dir, "health_report.json")
            with open(health_path, "w", encoding="utf-8") as f:
                json.dump(health, f, indent=2, ensure_ascii=False)

            return health

        except ImportError:
            logger.info("  SelfEvaluator not available, skipping")
            return {"status": "unknown"}

    def _evaluate_cortex_quality(self, report: SleepReport) -> dict:
        """
        神经元架构：评估 Cortex 神经元质量。

        使用 QualityFilter 评估各神经元的 PPL，
        调用 lifecycle.apoptosis.check_activation 检查激活率。
        """
        health = {
            "n_neurons": len(self.cortex.neurons),
            "neurons": {},
            "status": "healthy",
        }

        # 从凋亡追踪器获取 PPL 记录（Phase 2 已记录）
        ppl_records = {}
        if self._lifecycle is not None:
            try:
                ppl_records = self._lifecycle.apoptosis._failure_counts  # 有失败计数的神经元
            except Exception:
                pass

        # 检查每个神经元的激活率
        apoptosed = []
        if self._lifecycle is not None and self._coaction is not None:
            try:
                total_rounds = self._current_step
                for nid in self.cortex.neurons.keys():
                    activation_count = self._coaction._activation_count.get(nid, 0)
                    triggered = self._lifecycle.apoptosis.check_activation(
                        nid, activation_count, total_rounds
                    )
                    health["neurons"][nid] = {
                        "activation_count": activation_count,
                        "apoptosis_triggered": triggered,
                    }
                    if triggered:
                        apoptosed.append(nid)

                if apoptosed:
                    health["status"] = "degraded"
                    report.recommendations.append(
                        f"[凋亡] {len(apoptosed)} 个神经元触发凋亡: {apoptosed[:5]}"
                    )
                    logger.warning(f"  凋亡触发: {apoptosed}")
            except Exception as e:
                logger.warning(f"  凋亡检查失败: {e}")

        # 更新 QualityFilter 的 PPL 记录
        if self._quality_filter is not None and self._lifecycle is not None:
            try:
                # 从凋亡追踪器的失败计数推断 PPL 状态
                neuron_ppls = {}
                for nid in self.cortex.neurons.keys():
                    fail_count = self._lifecycle.apoptosis._failure_counts.get(nid, 0)
                    # 粗略映射：失败计数越高，PPL 越高
                    neuron_ppls[nid] = 50.0 + fail_count * 50.0
                self._quality_filter.set_ppls(neuron_ppls)
            except Exception as e:
                logger.debug(f"  QualityFilter 更新失败: {e}")

        return health

    def _sleep_phase_recursive_improvement(self, report: SleepReport):
        """
        Phase 5: 递归改进 — 策略优化 + 进化语料生成

        基于 Gödel Agent (ACL 2025) 的思想：
        态极在睡眠时分析自己的行为策略，找出可以改进的地方。
        同时生成下一代训练数据（递归蒸馏素材）。
        """
        try:
            # B4 修复：使用全局单例，保留历史策略记录
            from taiji.life.recursive_improver import get_recursive_improver
            improver = get_recursive_improver()

            # B3 修复：将 Phase 4 的评估结果注入到改进分析中
            health = report.health_status if hasattr(report, 'health_status') else None
            if health and health != "unknown":
                logger.debug(f"  基于评估结果执行改进分析 (health: {health})")

            # 1. 分析策略并生成改进提案
            proposals = improver.analyze_and_improve()
            if proposals:
                logger.info(f"  Generated {len(proposals)} improvement proposals")
                for p in proposals:
                    if p.confidence >= 0.7:
                        report.recommendations.append(f"[改进] {p.description}")
                        # Deep Coupling: 发布改进事件到 EventBus
                        try:
                            from taiji.infra.events import get_event_bus
                            bus = get_event_bus()
                            bus.publish("improvement_proposal", {
                                "proposal": {
                                    "type": p.proposal_type,
                                    "description": p.description,
                                    "new_value": p.new_value,
                                    "confidence": p.confidence,
                                }
                            }, source="sleep_engine")
                        except Exception:
                            pass

            # 2. 检查是否准备好进化（设计下一代）
            try:
                from taiji.life.evolution_engine import get_evolution_engine
                engine = get_evolution_engine()
                evolution_status = engine.check_evolution_ready()

                if evolution_status["ready"]:
                    logger.info(f"  Evolution ready: {evolution_status['reason']}")

                    # 态极自主设计下一代（可能变大、变小、专业化、多模态）
                    current_info = {
                        "name": evolution_status["current_generation"],
                        "params": "0.5B",  # TODO: 从实际模型获取
                        "hidden_size": 896,
                        "num_layers": 24,
                        "num_attention_heads": 14,
                        "weaknesses": self._identify_weaknesses(),
                        "strengths": self._identify_strengths(),
                        "resource_constraints": self._get_resource_constraints(),
                    }
                    next_gen_design = improver.design_next_generation(current_info)

                    # 保存设计方案
                    design_path = os.path.join(self.data_dir, "next_gen_design.json")
                    with open(design_path, "w", encoding="utf-8") as f:
                        json.dump(next_gen_design, f, indent=2, ensure_ascii=False)

                    report.recommendations.append(
                        f"[进化] 已设计下一代: {next_gen_design['next_gen_name']} "
                        f"(方向: {next_gen_design['evolution_direction']})"
                    )
                    logger.info(f"  Next generation designed: {next_gen_design['next_gen_name']} "
                               f"(direction: {next_gen_design['evolution_direction']})")

                    # ── 第三层闭环：执行代际迁移 ──
                    if self.config.auto_generation_transition:
                        try:
                            logger.info("  启动代际迁移（知识蒸馏）...")
                            model = self._get_model()
                            tokenizer = self._get_tokenizer()

                            if model is not None and tokenizer is not None:
                                training_texts = self._collect_training_texts()
                                result = engine.execute_generation_transition(
                                    design=next_gen_design,
                                    current_model=model,
                                    current_tokenizer=tokenizer,
                                    training_texts=training_texts,
                                    device=self._get_device(),
                                )
                                if result["success"]:
                                    report.recommendations.append(
                                        f"[进化] 代际迁移成功: {result['new_model_name']} "
                                        f"(distill_loss={result['distillation_loss']:.4f})"
                                    )
                                    logger.info(f"  代际迁移成功: {result['new_model_name']}")
                                else:
                                    report.recommendations.append(
                                        f"[进化] 代际迁移失败: {result.get('error', 'unknown')}"
                                    )
                                    logger.warning(f"  代际迁移失败: {result.get('error')}")
                            else:
                                logger.warning("  代际迁移跳过: model 或 tokenizer 不可用")
                        except Exception as e:
                            logger.error(f"  代际迁移异常: {e}", exc_info=True)
                            report.recommendations.append(f"[进化] 代际迁移异常: {e}")
            except ImportError:
                logger.info("  EvolutionEngine not available for evolution check")

            # 3. 生成进化语料（态极行为轨迹）
            self._generate_evolution_corpus(report)

            # 4. 睡眠评估反馈 → 针对性训练数据
            self._generate_weakness_training_data(report)

        except ImportError:
            logger.info("  RecursiveImprover not available, skipping")

    def _generate_weakness_training_data(self, report: SleepReport):
        """
        将睡眠评估中发现的弱点转化为针对性训练数据，
        存入标准训练数据目录供下一次 _sleep_phase_model_training 使用。
        闭合「评估 → 训练」反馈回路。
        """
        weaknesses = self._identify_weaknesses()
        if not weaknesses:
            return
        try:
            import os, json, datetime as dt
            output_dir = os.path.join(self.data_dir, "weakness_training_data")
            os.makedirs(output_dir, exist_ok=True)
            # 生成弱项针对性练习样本
            samples = []
            for w in weaknesses:
                # 根据弱项类型生成模板化训练样本
                if "数学" in w or "math" in w.lower():
                    samples.extend(self._math_weakness_samples())
                elif "代码" in w or "code" in w.lower() or "python" in w.lower():
                    samples.extend(self._code_weakness_samples())
                elif "准确" in w or "accuracy" in w.lower() or "低" in w:
                    samples.extend(self._accuracy_weakness_samples())
                elif "工具" in w or "tool" in w.lower() or "ReAct" in w:
                    samples.extend(self._tool_weakness_samples())
                else:
                    samples.append({
                        "instruction": f"请针对以下弱项提供详细解答：{w}",
                        "output": f"（此为自动生成的弱项针对性训练样本，指向：{w}）",
                        "weakness": w,
                    })
            if samples:
                ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(output_dir, f"weakness_fix_{ts}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(samples, f, indent=2, ensure_ascii=False)
                report.recommendations.append(
                    f"[训练反馈] 从 {len(weaknesses)} 个弱项生成 {len(samples)} 条训练数据"
                )
                logger.info(f"  Weaknesses → training data: {len(samples)} samples saved to {path}")
        except Exception as e:
            logger.warning(f"  弱项训练数据生成失败: {e}")

    def _math_weakness_samples(self) -> list:
        return [
            {"instruction": "计算 128 × 37 的结果", "output": "128 × 37 = 128 × (40 - 3) = 5120 - 384 = 4736"},
            {"instruction": "什么是勾股定理？请用例子说明", "output": "勾股定理：直角三角形中 a² + b² = c²。例：a=3, b=4 → c=5"},
        ]

    def _code_weakness_samples(self) -> list:
        return [
            {"instruction": "用 Python 写一个二分查找函数", "output": "def binary_search(arr, target):\n    left, right = 0, len(arr)-1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1"},
        ]

    def _accuracy_weakness_samples(self) -> list:
        return [
            {"instruction": "请详细解释相对论的基本原理", "output": "相对论由爱因斯坦提出，包含狭义和广义两部分。狭义相对论基于光速不变原理和相对性原理……"},
        ]

    def _tool_weakness_samples(self) -> list:
        return [
            {"instruction": "搜索 Python 3.12 的新特性并总结", "output": '[TOOL:search] Python 3.12 新特性\nPython 3.12 引入了更详细的错误信息、类型参数语法改进、per-interpreter GIL 等特性……'},
        ]

    def _identify_weaknesses(self) -> List[str]:
        """识别当前模型的弱点"""
        weaknesses = []
        try:
            from taiji.infra.self_evaluator import get_self_evaluator
            evaluator = get_self_evaluator()
            stats = evaluator.get_stats()
            if stats.get("avg_score", 1.0) < 0.6:
                weaknesses.append("整体回答质量偏低")
        except ImportError:
            pass

        # 从进化引擎获取失败模式
        try:
            from taiji.life.evolution_engine import get_evolution_engine
            engine = get_evolution_engine()
            total = engine.metrics.tasks_completed + engine.metrics.tasks_failed
            if total > 10:
                fail_rate = engine.metrics.tasks_failed / total
                if fail_rate > 0.3:
                    weaknesses.append(f"任务失败率高 ({fail_rate:.0%})")
        except ImportError:
            pass

        return weaknesses or ["信息不足，需要更多交互数据"]

    def _identify_strengths(self) -> List[str]:
        """识别当前模型的优势"""
        return ["中文理解", "身份稳定", "本地运行"]

    def _get_resource_constraints(self) -> dict:
        """获取当前设备的资源约束"""
        constraints = {"max_memory_gb": 16, "max_params": "7B"}
        try:
            import torch
            if torch.cuda.is_available():
                mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
                constraints["max_memory_gb"] = round(mem * 0.8)  # 留 20% 余量
                if mem < 8:
                    constraints["max_params"] = "1.5B"
                elif mem < 16:
                    constraints["max_params"] = "3B"
                elif mem < 24:
                    constraints["max_params"] = "7B"
                else:
                    constraints["max_params"] = "14B"
        except Exception as e:
            logger.debug("sleep_engine: non-critical %s", e, exc_info=True)
        return constraints

    def _generate_evolution_corpus(self, report: SleepReport):
        """生成进化语料（态极行为轨迹）"""
        try:
            corpus_dir = os.path.join(self.data_dir, "evolution_corpus")
            os.makedirs(corpus_dir, exist_ok=True)

            # 从工作记忆中提取行为轨迹
            from taiji.agent.working_memory import get_working_memory
            wm = get_working_memory()
            entries = wm.get_all()

            if not entries:
                logger.info("  No working memory entries for corpus generation")
                return

            # 生成行为样本
            samples = []
            for key, content in entries.items():
                if isinstance(content, str) and len(content) > 20:
                    samples.append({
                        "type": "memory_consolidation",
                        "key": key,
                        "content": content,
                        "timestamp": datetime.now().isoformat(),
                    })

            # 保存语料
            if samples:
                corpus_path = os.path.join(corpus_dir, f"corpus_{int(time.time())}.jsonl")
                with open(corpus_path, "w", encoding="utf-8") as f:
                    for s in samples:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")
                logger.info(f"  Generated {len(samples)} evolution corpus samples")

        except Exception as e:
            logger.debug(f"  Evolution corpus generation failed: {e}")

    def _collect_training_texts(self) -> list:
        """收集蒸馏训练用的文本数据。

        从工作记忆、进化语料和最近交互中提取文本列表。
        """
        texts = []

        # 1. 从工作记忆收集
        try:
            from taiji.agent.working_memory import get_working_memory
            wm = get_working_memory()
            entries = wm.get_all()
            for key, content in entries.items():
                if isinstance(content, str) and len(content) > 20:
                    texts.append(content)
        except ImportError:
            pass

        # 2. 从进化语料目录读取
        corpus_dir = os.path.join(self.data_dir, "evolution_corpus")
        if os.path.isdir(corpus_dir):
            for fname in sorted(os.listdir(corpus_dir))[-5:]:  # 最近 5 个
                fpath = os.path.join(corpus_dir, fname)
                if fname.endswith(".jsonl"):
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            for line in f:
                                try:
                                    item = json.loads(line)
                                    content = item.get("content", "")
                                    if content and len(content) > 20:
                                        texts.append(content)
                                except json.JSONDecodeError:
                                    continue
                    except Exception:
                        pass

        # 3. 确保至少有基本数据
        if not texts:
            texts = ["态极正在通过递归蒸馏自我进化。"]

        logger.info(f"  收集了 {len(texts)} 条训练文本用于蒸馏")
        return texts

    def _get_device(self) -> str:
        """获取当前可用的训练设备。"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _get_model(self):
        """获取当前模型实例。"""
        if self._model_provider:
            return self._model_provider()
        try:
            from taiji.core.app_state import app_state
            return app_state.model
        except ImportError:
            return None

    def _get_tokenizer(self):
        """获取当前 tokenizer 实例。"""
        if self._tokenizer_provider:
            return self._tokenizer_provider()
        try:
            from taiji.core.app_state import app_state
            return app_state.tokenizer
        except ImportError:
            return None

    # ─── 持久化 ─────────────────────────────────────
    
    def _ensure_data_dir(self):
        """延迟创建数据目录（只在首次写入时创建）"""
        if not self._data_dir_ready:
            os.makedirs(self.data_dir, exist_ok=True)
            self._data_dir_ready = True

    def _save_history(self):
        """保存睡眠历史"""
        self._ensure_data_dir()
        path = os.path.join(self.data_dir, "sleep_history.json")
        try:
            data = []
            for report in self._sleep_history[-50:]:  # 只保留最近 50 次
                data.append({
                    "timestamp": report.timestamp,
                    "duration_seconds": report.duration_seconds,
                    "phases_completed": report.phases_completed,
                    "memory_entries_cleared": report.memory_entries_cleared,
                    "training_samples_used": report.training_samples_used,
                    "evolution_events": report.evolution_events,
                    "health_status": report.health_status,
                })
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save sleep history: {e}")
    
    def _load_history(self):
        """加载睡眠历史"""
        path = os.path.join(self.data_dir, "sleep_history.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                self._sleep_history.append(SleepReport(**item))
        except Exception as e:
            logger.warning(f"Failed to load sleep history: {e}")
    
    # ─── 状态查询 ───────────────────────────────────
    
    def get_status(self) -> dict:
        """获取睡眠引擎状态"""
        return {
            "is_sleeping": self._is_sleeping,
            "last_sleep": self._last_sleep_time.isoformat() if self._last_sleep_time else None,
            "last_activity": self._last_activity_time.isoformat() if self._last_activity_time else None,
            "total_sleeps": len(self._sleep_history),
            "auto_sleep_enabled": self.config.auto_sleep_enabled,
        }
    
    def get_summary(self) -> str:
        """获取人类可读的状态摘要"""
        status = self.get_status()
        
        sleeping = "💤 睡眠中" if status["is_sleeping"] else "☀️ 清醒"
        last_sleep = status["last_sleep"] or "从未睡眠"
        
        lines = [
            "💤 睡眠引擎状态",
            "━━━━━━━━━━━━━━━━",
            f"当前状态: {sleeping}",
            f"上次睡眠: {last_sleep}",
            f"总睡眠次数: {status['total_sleeps']}",
            f"自动睡眠: {'✅ 开启' if status['auto_sleep_enabled'] else '❌ 关闭'}",
        ]
        
        if self._sleep_history:
            last = self._sleep_history[-1]
            lines.append(f"\n最近一次睡眠报告:")
            lines.append(f"  时长: {last.duration_seconds}s")
            lines.append(f"  阶段: {', '.join(last.phases_completed)}")
            lines.append(f"  健康状态: {last.health_status}")
        
        return "\n".join(lines)
    
    def get_sleep_trends(self) -> List[str]:
        """分析睡眠趋势"""
        if len(self._sleep_history) < 3:
            return ["数据不足，至少需要 3 次睡眠记录"]
        
        recent = self._sleep_history[-5:]
        avg_duration = sum(r.duration_seconds for r in recent) / len(recent)
        avg_phases = sum(len(r.phases_completed) for r in recent) / len(recent)
        
        trends = [
            f"最近 {len(recent)} 次睡眠平均时长: {avg_duration:.1f}s",
            f"平均完成阶段数: {avg_phases:.1f}/4",
        ]
        
        # 检查训练效果
        recent_training = [r.training_samples_used for r in recent if r.training_samples_used > 0]
        if recent_training:
            avg_samples = sum(recent_training) / len(recent_training)
            trends.append(f"平均训练样本数: {avg_samples:.0f}")
        
        return trends


# ─── 全局实例 ─────────────────────────────────────

_global_sleep: Optional[SleepEngine] = None


def get_sleep_engine(config: Optional[SleepConfig] = None) -> SleepEngine:
    """获取全局睡眠引擎实例"""
    global _global_sleep
    if _global_sleep is None:
        _global_sleep = SleepEngine(config)
    return _global_sleep
