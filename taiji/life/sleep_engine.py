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
from collections import deque

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
        self._feed_engine: Optional[Any] = None  # FeedEngine
        self._current_step: int = 0  # 全局步数计数器（供 SleepConsolidator）
        # P1-2: 神经调质状态（多巴胺/血清素/去甲肾上腺素）
        # 自主进化核心：双信号驱动调质，自动调节学习率
        if NeuromodulatorState is not None:
            try:
                self._neuromodulator = NeuromodulatorState()
            except Exception:
                self._neuromodulator = None
        else:
            self._neuromodulator = None

        # 自适应学习率：双信号驱动（loss 趋势 + 准确率）
        # 快速信号：loss 变化率每轮更新多巴胺目标值
        self._loss_history: deque = deque(maxlen=10)
        # 慢速信号：每 N 轮评估 next-token 准确率，校准血清素
        self._eval_interval: int = 5
        self._eval_counter: int = 0
        self._accuracy_history: deque = deque(maxlen=5)

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
        feed_engine: Optional[Any] = None,
        neuromodulator: Optional[Any] = None,
    ):
        """
        注入神经元架构组件（Cortex + ResonanceEnsemble 体系）。

        Args:
            cortex: Cortex 实例（含 neurons + ensemble）
            lifecycle: LifecycleManager（apoptosis/neurogenesis/maturity）
            sleep_consolidator: SleepConsolidator（睡眠巩固）
            stdp_tracker: STDPTracker（局部学习）
            feed_engine: FeedEngine（数据喂养）
            neuromodulator: NeuromodulatorState（P1-2，多巴胺/血清素/去甲肾上腺素）

        Note:
            RecursiveImprover 通过全局单例 get_recursive_improver() 访问，
            Phase 5 _sleep_phase_recursive_improvement 直接导入使用，无需注入。
        """
        if cortex is not None:
            self.cortex = cortex
        if lifecycle is not None:
            self._lifecycle = lifecycle
        if sleep_consolidator is not None:
            self._sleep_consolidator = sleep_consolidator
        if stdp_tracker is not None:
            self._stdp_tracker = stdp_tracker
        if feed_engine is not None:
            self._feed_engine = feed_engine

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

        # MaturityTracker: 注入 cortex.ensemble（驱动共振权重，幼稚态 0.1 → 成熟态 1.0）
        if self.cortex is not None and self._lifecycle is not None:
            try:
                self.cortex.set_maturity(self._lifecycle.maturity)
            except Exception as e:
                logger.debug(f"cortex.set_maturity 失败（非关键）: {e}")

        # SleepConsolidator: 注入 cortex（供 save_state/load_state 持久化 replay buffer）
        if self.cortex is not None and self._sleep_consolidator is not None:
            try:
                self.cortex.set_sleep_consolidator(self._sleep_consolidator)
            except Exception as e:
                logger.debug(f"cortex.set_sleep_consolidator 失败（非关键）: {e}")

        logger.info(
            f"Brain interfaces set: cortex={'✓' if self.cortex else '✗'}, "
            f"lifecycle={'✓' if self._lifecycle else '✗'}, "
            f"sleep_consolidator={'✓' if self._sleep_consolidator else '✗'}, "
            f"stdp={'✓' if self._stdp_tracker else '✗'}, "
            f"feed_engine={'✓' if self._feed_engine else '✗'}, "
            f"neuromodulator={'✓' if self._neuromodulator else '✗'}"
        )

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
        from datetime import datetime
        if self._is_sleeping:
            return
        self._is_sleeping = True
        report = SleepReport(
            timestamp=datetime.now().isoformat(),
            duration_seconds=0,
        )
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
        """Phase 2: 神经元训练 - 用收集的数据训练 Cortex 神经元（P7 独立 lm_head）。"""
        if self.cortex is not None and hasattr(self.cortex, 'neurons') and self.cortex.neurons:
            self._train_cortex_neurons(report)
            return

        logger.warning("Cortex 未注入，跳过睡眠训练")

    def _select_split_parent(self, domain: str) -> Optional[str]:
        """选择分裂父 neuron（LuminaNet splitting 融合）。

        策略：选同域中共振分数最高的 neuron 作为父本，
        高负载（高错误率）的 neuron 分裂出子 neuron 分担工作。

        Args:
            domain: 域名

        Returns:
            父 neuron ID，或 None（域内无 neuron 时从零新建）
        """
        if self.cortex is None:
            return None

        # 收集同域 neuron
        domain_nids = [
            nid for nid in self.cortex.neurons
            if nid == domain or nid.startswith(f"{domain}_")
        ]
        if not domain_nids:
            return None  # 域内无 neuron，从零新建

        if len(domain_nids) == 1:
            return domain_nids[0]

        # 多个同域 neuron 时选共振分数最高的
        try:
            scores = getattr(self.cortex.ensemble, 'round_scores', [])
            if scores:
                last_scores = scores[-1] if scores else {}
                best_nid = max(
                    domain_nids,
                    key=lambda n: last_scores.get(n, 0.0),
                )
                return best_nid
        except Exception:
            pass

        # fallback: 第一个同域 neuron
        return domain_nids[0]

    def _train_cortex_neurons(self, report: SleepReport):
        """
        神经元架构：训练 Cortex 中每个域的神经元（P7 模式）。

        核心流程：
        0. 调用 metabolism.update_neuromodulator() 评估硬件状态 → 更新 NE（field_write 强度）
        1. 从 feed_engine.get_pending_samples_by_domain() 获取按域分类的样本
        2. 用域 tokenizer + per-neuron embedding 训练独立 lm_head
        3. 记录 PPL 到 lifecycle.apoptosis.record_ppl
        4. 调用 stdp_tracker.apply_all_updates(cortex.neurons) 应用 STDP
        5. 检查 lifecycle.neurogenesis 触发条件
        """
        # Step 0: 硬件状态评估 → 更新去甲肾上腺素（field_write 强度）
        # 自主进化：训练前感知硬件负载，高负载时降低 NE → 减少 field_write → 节能
        try:
            from taiji.body import metabolism
            metabolism.set_neuromodulator(self._neuromodulator)
            metabolism.update_neuromodulator()
        except Exception as e:
            logger.debug(f"metabolism 调质更新失败（非关键）: {e}")

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

        tokenizer_hub = getattr(self.cortex, '_tokenizer_hub', None)
        if tokenizer_hub is None:
            logger.warning("  Cortex 未设置 tokenizer_hub，跳过训练")
            return

        logger.info("  P7 模式：使用 per-neuron embedding + 域 tokenizer 训练")

        # 获取训练锁，防止与 generate/其他训练并发修改 neuron 权重
        # 非阻塞：锁被占用时跳过本次训练，不阻塞睡眠流程
        from taiji.core.app_state import app_state
        if not app_state.try_start_training():
            logger.warning("  训练锁被占用，跳过本次 Cortex 训练")
            return

        try:
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

                # 分离文本样本和多模态样本
                text_samples = [s for s in samples if s.get("type") != "multimodal"]
                mm_samples = [s for s in samples if s.get("type") == "multimodal"]

                # 文本样本训练（经验驱动：shared_embedding + lm_head 协同学习）
                if text_samples:
                    avg_loss, ppl = self._train_single_neuron(
                        neuron, domain, text_samples, cortex=self.cortex
                    )

                    if avg_loss is not None:
                        total_loss = total_loss + avg_loss
                        trained_count = trained_count + 1
                        ppl_results[domain] = ppl
                        logger.info(f"  域 '{domain}' 文本训练完成: loss={avg_loss:.4f}, PPL={ppl:.1f}")

                # 多模态样本训练（新逻辑）— 所有 neuron 参与共振
                for mm_sample in mm_samples:
                    modality = mm_sample.get("modality")
                    if modality:
                        mm_loss, mm_ppl = self._train_multimodal_ensemble(
                            modality, mm_sample, tokenizer_hub=tokenizer_hub
                        )
                        if mm_loss is not None:
                            total_loss = total_loss + mm_loss
                            trained_count = trained_count + 1
                            logger.info(f"  模态 '{modality}' ensemble 训练完成: loss={mm_loss:.4f}, PPL={mm_ppl:.1f}")

                # 记录 PPL 到凋亡追踪器
                if self._lifecycle is not None:
                    try:
                        if domain in ppl_results:
                            self._lifecycle.apoptosis.record_ppl(domain, ppl_results[domain])
                    except Exception as e:
                        logger.debug(f"  apoptosis.record_ppl 失败: {e}")

            # 应用 STDP 更新（局部学习规则，也修改权重，需在锁内）
            if self._stdp_tracker is not None:
                try:
                    updates = self._stdp_tracker.apply_all_updates(self.cortex.neurons)
                    if updates:
                        logger.info(f"  STDP 更新: {len(updates)} 个神经元")
                except Exception as e:
                    logger.warning(f"  STDP 更新失败: {e}")

            # Contrastive phase: 增强 neuron 间场向量差异化
            # 机制借鉴 MoCo Top-k/Bottom-k Contrastive Loss
            # 在所有 neuron 单独训练 + STDP 后执行，推开跨域场向量
            try:
                contrastive_loss = self._train_contrastive_phase(self.cortex)
                if contrastive_loss is not None:
                    report.recommendations.append(
                        f"[对比学习] 场向量差异化 loss={contrastive_loss:.4f}"
                    )
            except Exception as e:
                logger.warning(f"  contrastive phase 失败（非关键）: {e}")
        finally:
            app_state.finish_training()

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
                        # 运行时创建新神经元并加入 ensemble
                        if self.cortex is not None:
                            try:
                                # LuminaNet splitting 融合：
                                # 同域已有 neuron 时优先分裂最强者（继承权重 + 噪声分化），
                                # 新 neuron 起点高于随机初始化；域首 neuron 从零新建
                                split_parent = self._select_split_parent(domain)
                                new_nid = self.cortex.add_neuron(
                                    domain, lifecycle=self._lifecycle,
                                    from_split=split_parent,
                                )
                                split_info = f" (split from {split_parent})" if split_parent else " (from scratch)"
                                logger.info(f"  🌱 neurogenesis 完成: {new_nid}{split_info}")
                                report.recommendations.append(
                                    f"[神经新生] 新神经元 {new_nid} 已创建{split_info}"
                                )
                            except Exception as ne:
                                logger.warning(f"  neurogenesis 创建失败: {ne}")
            except Exception as e:
                logger.debug(f"  neurogenesis 检查失败: {e}")

        # 检查孤立激活模式（CoactivationTracker 第二触发源）
        # 传入 maturity_tracker 过滤幼稚态 neuron：新 neuron 天然无共激活历史，
        # 100% pair 是低频，会形成"检测孤立→创建新 neuron→新 neuron 又孤立"的正反馈
        if self._lifecycle is not None and self.cortex is not None:
            try:
                coaction = getattr(self.cortex, "coaction", None)
                if coaction is not None:
                    maturity = getattr(self._lifecycle, "maturity", None)
                    isolated_nids = self._lifecycle.neurogenesis.detect_isolated_patterns(
                        coaction, min_isolation_ratio=0.8,
                        maturity_tracker=maturity, min_maturity_ratio=0.1,
                    )
                    if isolated_nids and self.cortex is not None:
                        logger.info(f"  孤立神经元检测: {isolated_nids}")
                        for nid in isolated_nids:
                            # 从 nid 推断 domain（格式: domain 或 domain_N）
                            domain = nid.split("_")[0] if "_" in nid else nid
                            try:
                                # LuminaNet splitting: 孤立 neuron 分裂出协同 neuron
                                # 孤立 neuron 自身作为分裂父本，子 neuron 继承权重后分化
                                split_parent = nid if nid in self.cortex.neurons else self._select_split_parent(domain)
                                new_nid = self.cortex.add_neuron(
                                    domain, lifecycle=self._lifecycle,
                                    from_split=split_parent,
                                )
                                split_info = f" (split from {split_parent})" if split_parent else ""
                                logger.info(f"  🌱 孤立协同神经元创建: {new_nid}{split_info} (为 {nid})")
                                report.recommendations.append(
                                    f"[神经新生] 孤立神经元 {nid} → 创建协同神经元 {new_nid}{split_info}"
                                )
                            except Exception as ne:
                                logger.warning(f"  孤立协同神经元创建失败: {ne}")
            except Exception as e:
                logger.debug(f"  孤立模式检测失败: {e}")

        # 递增成熟度
        if self._lifecycle is not None:
            try:
                self._lifecycle.maturity.tick_all()
            except Exception as e:
                logger.debug(f"  maturity.tick_all 失败: {e}")

        # 重置域错误率计数器（每个 sleep 周期独立统计）
        # 避免终身累积错误率导致每轮触发 neurogenesis
        if self._feed_engine is not None:
            try:
                self._feed_engine.reset_domain_counts()
            except Exception as e:
                logger.debug(f"  reset_domain_counts 失败: {e}")

        # 记录训练损失
        if trained_count > 0:
            report.training_loss = total_loss / trained_count

        # ── 自适应学习率：双信号驱动神经调质 ──
        if trained_count > 0 and self._neuromodulator is not None:
            self._update_neuromodulators(report.training_loss)

        # 训练后自动保存经验积累状态（shared_embedding + lm_head 权重）
        # 使下次启动 Cortex 时从当前状态继续，而非从随机初始化重新开始
        # 测试模式下（TAJIJI_TEST_MODE=1）跳过保存，确保测试可复现
        if trained_count > 0 and not os.environ.get('TAJIJI_TEST_MODE'):
            # 更新 fingerprint：embed_adapter 在训练中已更新，重新计算 fingerprint
            for nid, neuron in self.cortex.neurons.items():
                if hasattr(neuron, "freeze_fingerprint"):
                    neuron.freeze_fingerprint()
            try:
                neurons_dir = getattr(self.cortex, 'neurons_dir', 'data/neurons')
                self.cortex.save_state(neurons_dir)
                logger.info(f"  经验积累状态已保存到 {neurons_dir}/cortex_state.pt")
            except Exception as e:
                logger.warning(f"  保存经验积累状态失败（非致命）: {e}")

        # 步数递增
        self._current_step += 1

    def _update_neuromodulators(self, current_loss: float) -> None:
        """双信号驱动神经调质更新（自主进化核心）。

        快速信号（每轮）：loss 变化率 → 多巴胺 → 学习率倍数
        慢速信号（每 N 轮）：next-token 准确率 → 血清素 → 满足度

        人脑启发：
        - 多巴胺 = 奖励预测误差。loss 快速下降 = 学习有效 = 正奖励 → dopamine↑ → lr↑
        - 血清素 = 满足感。准确率长期改善 = 能力提升 = 满足 → serotonin↑
        """
        # ── 快速信号：loss 趋势 → 多巴胺 ──
        self._loss_history.append(current_loss)

        if len(self._loss_history) >= 2:
            prev_loss = self._loss_history[-2]
            if prev_loss > 0:
                # loss 变化率：负值表示下降（学习有效）
                delta = (current_loss - prev_loss) / prev_loss

                if delta < -0.2:
                    # 快速下降 → 强奖励
                    dopamine_target = 0.85
                elif delta < -0.05:
                    # 正常下降 → 适度奖励
                    dopamine_target = 0.6
                elif delta < 0.05:
                    # 停滞 → 降低
                    dopamine_target = 0.3
                else:
                    # loss 上升 → 惩罚
                    dopamine_target = 0.15

                self._neuromodulator.set_targets(dopamine=dopamine_target)

                logger.info(
                    f"  调质更新: loss={current_loss:.4f} (Δ={delta:+.1%}) → "
                    f"dopamine_target={dopamine_target} → lr_mult={self._neuromodulator.get_lr_multiplier():.2f}"
                )

        # ── 慢速信号：每 N 轮评估准确率 → 血清素 ──
        self._eval_counter += 1
        if self._eval_counter >= self._eval_interval:
            self._eval_counter = 0
            try:
                accuracy = self._evaluate_next_token_accuracy()
                if accuracy is not None:
                    self._accuracy_history.append(accuracy)

                    if len(self._accuracy_history) >= 2:
                        prev_acc = self._accuracy_history[-2]
                        acc_delta = accuracy - prev_acc

                        if acc_delta > 0.02:
                            # 准确率提升 → 满足
                            serotonin_target = 0.7
                        elif acc_delta > -0.02:
                            # 持平 → 中性
                            serotonin_target = 0.5
                        else:
                            # 下降 → 不满足
                            serotonin_target = 0.3

                        self._neuromodulator.set_targets(serotonin=serotonin_target)
                        logger.info(
                            f"  慢速校准: acc={accuracy:.1%} (Δ={acc_delta:+.1%}) → "
                            f"serotonin_target={serotonin_target}"
                        )
            except Exception as e:
                logger.debug(f"  准确率评估失败: {e}")

        # EMA 趋近目标值（调质不会突变，而是缓慢调整）
        self._neuromodulator.step()

    def _evaluate_next_token_accuracy(self) -> Optional[float]:
        """评估 next-token 预测准确率（慢速信号）。

        用 feed_engine 中最近的样本做评估：
        - 对每个样本，用前缀预测下一个 token
        - 统计 top-1 准确率
        """
        if self.cortex is None or self._feed_engine is None:
            return None

        tokenizer_hub = getattr(self.cortex, '_tokenizer_hub', None)
        shared_embedding = getattr(self.cortex, '_shared_embedding', None)
        general_sp = getattr(self.cortex, '_general_sp', None)

        if tokenizer_hub is None or shared_embedding is None or general_sp is None:
            return None

        # 获取最近样本
        domain_samples = self._feed_engine.get_pending_samples_by_domain()
        if not domain_samples:
            return None

        correct = 0
        total = 0

        import torch
        with torch.no_grad():
            for domain, samples in domain_samples.items():
                neuron = self.cortex.neurons.get(domain)
                if neuron is None:
                    continue

                domain_sp = tokenizer_hub.get_tokenizer(domain)
                if domain_sp is None:
                    continue

                for sample in samples[:5]:  # 每域最多评估 5 条
                    text = sample.get("text", "") if isinstance(sample, dict) else str(sample)
                    if not text or len(text) < 5:
                        continue

                    domain_ids = tokenizer_hub.encode(text, domain=domain)
                    if len(domain_ids) < 4:
                        continue

                    # 逐 token 映射构造输入（与训练路径一致）
                    general_ids = []
                    for did in domain_ids:
                        piece = domain_sp.id_to_piece(did)
                        gen_ids = general_sp.EncodeAsIds(piece)
                        if gen_ids:
                            general_ids.append(gen_ids[0])
                        else:
                            general_ids.append(0)

                    # 对每个位置预测下一个 token
                    for i in range(1, min(len(general_ids) - 1, 8)):
                        prefix = general_ids[:i + 1]
                        if len(prefix) < 2:
                            continue

                        ids_tensor = torch.tensor([prefix], dtype=torch.long,
                                                   device=shared_embedding.weight.device)
                        emb = shared_embedding(ids_tensor)
                        result = neuron.forward(emb, field_state=None, round_num=1,
                                                return_logits=True)
                        logits = result.get("logits")
                        if logits is None:
                            continue

                        pred = torch.argmax(logits[0, -1, :]).item()
                        true = domain_ids[i + 1] if i + 1 < len(domain_ids) else domain_ids[-1]

                        total += 1
                        if pred == true:
                            correct += 1

        if total == 0:
            return None
        return correct / total

    def _train_single_neuron(
        self, neuron, domain: str, samples: list, cortex
    ) -> tuple:
        """P7: 训练单个神经元的独立 lm_head + shared_embedding 协同学习。

        经验驱动学习（非蒸馏）：
        - 输入：general tokenizer encode → general_ids → cortex._shared_embedding 查表 → embeddings
        - 目标：domain tokenizer encode → domain_ids（lm_head 输出在 domain vocab）
        - 可训练参数：neuron.lm_head + neuron.embed_adapter + cortex._shared_embedding
        - 训练后 shared_embedding 的更新保留在 cortex 中（经验积累）

        Args:
            neuron: ResonanceNeuron 实例
            domain: 域标签
            samples: 训练样本列表（dict with text content）
            cortex: Cortex 实例（提供 shared_embedding + general_sp + tokenizer_hub）

        Returns:
            (avg_loss, ppl) or (None, None) on failure
        """
        import torch
        import torch.nn.functional as F

        # 从 cortex 获取 P7 组件
        shared_embedding = getattr(cortex, '_shared_embedding', None)
        general_sp = getattr(cortex, '_general_sp', None)
        tokenizer_hub = getattr(cortex, '_tokenizer_hub', None)

        if shared_embedding is None:
            logger.warning(f"  [{domain}] cortex._shared_embedding 未设置，跳过")
            return None, None
        if general_sp is None:
            logger.warning(f"  [{domain}] cortex._general_sp 未设置，跳过")
            return None, None
        if tokenizer_hub is None:
            logger.warning(f"  [{domain}] cortex._tokenizer_hub 未设置，跳过")
            return None, None

        device = next(neuron.parameters()).device

        # 收集可训练参数：lm_head + embed_adapter + shared_embedding
        # shared_embedding 是感官层，与神经元协同学习（经验驱动，非蒸馏）
        trainable_params = list(neuron.lm_head.parameters())
        if hasattr(neuron, 'embed_adapter'):
            trainable_params.extend(neuron.embed_adapter.parameters())
        trainable_params.extend(shared_embedding.parameters())

        if not trainable_params:
            logger.warning(f"  [{domain}] 无可训练参数，跳过")
            return None, None

        # 自适应学习率：神经调质（多巴胺）驱动 lr 倍数
        # 自主进化时，多巴胺由 loss 趋势 + 准确率双信号自动调节
        base_lr = 1e-3
        lr_mult = 1.0
        if self._neuromodulator is not None:
            lr_mult = self._neuromodulator.get_lr_multiplier()
        # MaturityTracker: 幼稚态神经元 lr 倍数（×3.0），成熟态衰减到 ×1.0
        # 新生神经元学习加速，追赶成熟神经元的能力
        if self._lifecycle is not None:
            try:
                maturity_lr_mult = self._lifecycle.maturity.get_lr_multiplier(domain)
                lr_mult *= maturity_lr_mult
            except Exception:
                pass
        adaptive_lr = base_lr * lr_mult
        optimizer = torch.optim.AdamW(trainable_params, lr=adaptive_lr)

        # 提取训练文本
        texts = []
        for sample in samples:
            if isinstance(sample, dict):
                text = sample.get("text", "") or sample.get("content", "") or \
                       sample.get("task", "") or sample.get("answer", "") or \
                       " ".join(str(v) for v in sample.values() if isinstance(v, str))
            else:
                text = str(sample)
            if len(text.strip()) > 10:
                texts.append(text)

        if not texts:
            logger.debug(f"  [{domain}] 无有效训练文本，跳过")
            return None, None

        # 限制样本数（CPU 模式下不宜太多）
        max_samples = min(len(texts), 64)
        # 随机采样：每轮训练不同的 64 条样本，释放大训练集的全部价值
        # 避免固定前 64 条导致数据利用率只有 64/N
        import random
        if len(texts) > max_samples:
            random.shuffle(texts)
        texts = texts[:max_samples]

        neuron.train()
        total_loss = 0.0
        trained_steps = 0

        # 多 epoch 训练：少量样本需要重复学习才能有效更新权重
        # 从随机初始化学习需要更多训练次数（非微调）
        NUM_EPOCHS = 3
        domain_sp = tokenizer_hub.get_tokenizer(domain)
        for epoch in range(NUM_EPOCHS):
            for text in texts:
                try:
                    # 目标：domain tokenizer encode → domain_ids（lm_head 输出空间）
                    domain_ids = tokenizer_hub.encode(text, domain=domain)
                    if not domain_ids or len(domain_ids) < 3:
                        continue
                    domain_ids = domain_ids[:256]
                    target_ids = torch.tensor([domain_ids], dtype=torch.long, device=device)

                    # 输入：逐 token 映射 domain_ids → general_ids
                    # 每个 domain token 的 piece 用 general tokenizer 重新编码，
                    # 取第一个 general token id 查找 shared_embedding。
                    # 这样 input 和 target 长度一致（都是 len(domain_ids)），
                    # 自回归 CE loss 的 shift 对齐正确。
                    general_ids = []
                    for did in domain_ids:
                        piece = domain_sp.id_to_piece(did)
                        gen_ids = general_sp.EncodeAsIds(piece)
                        if gen_ids:
                            general_ids.append(gen_ids[0])
                        else:
                            general_ids.append(0)

                    if len(general_ids) < 3:
                        continue
                    input_ids = torch.tensor([general_ids], dtype=torch.long, device=device)
                    embeddings = shared_embedding(input_ids)

                    # Forward + backward
                    optimizer.zero_grad()
                    result = neuron.forward(
                        embeddings, field_state=None, round_num=1,
                        return_logits=True,
                    )
                    logits = result["logits"]  # [1, L, domain_vocab]

                    # 自回归 CE loss: predict next domain token
                    # input 和 target 长度一致（都是 len(domain_ids)），shift 对齐正确
                    min_len = logits.size(1) - 1
                    if min_len < 1:
                        continue
                    shift_logits = logits[:, :min_len, :].contiguous()
                    shift_targets = target_ids[:, 1:1 + min_len].contiguous()

                    # clamp targets to neuron's vocab
                    vocab_size = logits.size(-1)
                    shift_targets = shift_targets.clamp(0, vocab_size - 1)

                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_targets.view(-1),
                        ignore_index=-100,
                    )
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    trained_steps += 1

                except Exception as e:
                    logger.debug(f"  [{domain}] 训练步失败: {e}")
                    continue

        neuron.eval()

        if trained_steps == 0:
            return None, None

        avg_loss = total_loss / trained_steps
        ppl = math.exp(min(avg_loss, 20))
        return avg_loss, ppl

    def _train_contrastive_phase(self, cortex) -> Optional[float]:
        """Contrastive phase: 增强 neuron 间场向量差异化。

        在所有 neuron 单独训练后执行：
        1. 取每个 domain 一个样本，用 general tokenizer encode → shared_embedding
        2. 所有 neuron forward 获取 field_vectors（保持计算图）
        3. 计算 inter loss（跨 neuron push apart）+ intra loss（同 neuron 内 push apart）
        4. 反传到 shared_embedding + embed_adapter（不动 lm_head，保护 LM 能力）

        机制借鉴：MoCo Top-k/Bottom-k Contrastive Loss。
        与 BioOSS 协同：差异化场向量使 inhibitory neuron 能精准定向抑制。

        Args:
            cortex: Cortex 实例

        Returns:
            contrastive loss 或 None（跳过时）
        """
        import torch
        import torch.nn.functional as F

        if len(cortex.neurons) < 2:
            return None  # 单 neuron 无 inter loss 意义

        shared_embedding = getattr(cortex, '_shared_embedding', None)
        general_sp = getattr(cortex, '_general_sp', None)
        tokenizer_hub = getattr(cortex, '_tokenizer_hub', None)
        if shared_embedding is None or general_sp is None or tokenizer_hub is None:
            return None

        device = next(shared_embedding.parameters()).device

        # 收集可训练参数：shared_embedding + 所有 neuron 的 embed_adapter
        # 不含 lm_head（保护刚学到的 LM 能力）
        trainable_params = list(shared_embedding.parameters())
        for neuron in cortex.neurons.values():
            if hasattr(neuron, 'embed_adapter'):
                trainable_params.extend(neuron.embed_adapter.parameters())

        if not trainable_params:
            return None

        optimizer = torch.optim.AdamW(trainable_params, lr=2e-4)  # 较小 lr

        # 每个域用域特异性文本，增强对比意义
        domain_samples = {
            "zh": "神经元共振场架构设计",
            "en": "neural resonance field architecture",
            "code": "def resonance(field): return field.sync()",
            "math": "integral of sin(x) over domain",
            "general": "system design pattern overview",
        }
        all_field_vectors: Dict[str, torch.Tensor] = {}
        for nid, neuron in cortex.neurons.items():
            # 从 domain 名推断（nid 可能是 "zh" 或 "zh_1"）
            domain = nid.split("_")[0] if "_" in nid else nid
            try:
                domain_sp = tokenizer_hub.get_tokenizer(domain)
            except Exception:
                continue

            try:
                # 域特异性文本（不同 domain 用不同输入，增强对比意义）
                sample_text = domain_samples.get(domain, "system overview")
                domain_ids = tokenizer_hub.encode(sample_text, domain=domain)
                if not domain_ids or len(domain_ids) < 3:
                    continue
                domain_ids = domain_ids[:32]

                # 逐 token 映射到 general tokenizer
                general_ids = []
                for did in domain_ids:
                    piece = domain_sp.id_to_piece(did)
                    gen_ids = general_sp.EncodeAsIds(piece)
                    if gen_ids:
                        general_ids.append(gen_ids[0])
                    else:
                        general_ids.append(0)

                if len(general_ids) < 3:
                    continue

                input_ids = torch.tensor([general_ids], dtype=torch.long, device=device)
                embeddings = shared_embedding(input_ids)

                neuron.train()
                result = neuron.forward(
                    embeddings, field_state=None, round_num=1,
                    return_logits=False,
                )
                field_vec = result["field_vector"]  # [1, D]
                all_field_vectors[nid] = field_vec
            except Exception as e:
                logger.debug(f"  contrastive phase: neuron {nid} forward 失败: {e}")
                continue

        if len(all_field_vectors) < 2:
            return None

        # 计算 contrastive loss
        # 高维空间（field_dim=2048）中，随机初始化的场向量天然近正交
        # 用 cosine² 而非 margin-based loss，任何正的 cosine 都会产生梯度
        nids = list(all_field_vectors.keys())
        N = len(nids)

        # 归一化
        normed = {}
        for nid, vecs in all_field_vectors.items():
            v = vecs if vecs.dim() == 2 else vecs.unsqueeze(0)
            normed[nid] = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

        # Inter loss: 跨域 neuron push apart（cosine²，强推开）
        # Intra-group diversity (Hi-MoE 融合): 同域 neuron margin-based loss
        # 同域 neuron 处理相似输入，允许一定相似（margin=0.3）但不完全相同，
        # 防止 neurogenesis 生成的新 neuron 学成老 neuron 的副本
        inter_loss = torch.tensor(0.0, device=device)
        inter_count = 0
        intra_group_loss = torch.tensor(0.0, device=device)
        intra_group_count = 0
        intra_margin = 0.3  # 同域允许的相似度上限

        # 按域分组（nid 格式 "domain" 或 "domain_n"）
        from collections import defaultdict
        domain_groups: Dict[str, List[str]] = defaultdict(list)
        for nid in nids:
            domain = nid.split("_")[0] if "_" in nid else nid
            domain_groups[domain].append(nid)

        for i in range(N):
            for j in range(i + 1, N):
                v_i = normed[nids[i]]
                v_j = normed[nids[j]]
                sim = v_i @ v_j.T
                domain_i = nids[i].split("_")[0] if "_" in nids[i] else nids[i]
                domain_j = nids[j].split("_")[0] if "_" in nids[j] else nids[j]

                if domain_i == domain_j and len(domain_groups[domain_i]) > 1:
                    # 同域 neuron：margin-based loss（允许相似度 < margin，超过则惩罚）
                    # 防止完全相同，但允许共享域特征
                    sim_mean = sim.mean()
                    intra_group_loss = intra_group_loss + F.relu(sim_mean - intra_margin) ** 2
                    intra_group_count += 1
                else:
                    # 跨域 neuron：cosine² 强推开
                    inter_loss = inter_loss + (sim ** 2).mean()
                    inter_count += 1

        inter_loss = inter_loss / max(inter_count, 1)
        intra_group_loss = intra_group_loss / max(intra_group_count, 1)

        # Intra loss (legacy): 同 neuron 内 batch 维度（batch=1 时为 0）
        intra_loss = torch.tensor(0.0, device=device)
        intra_count = 0
        for nid in nids:
            v = normed[nid]
            B = v.shape[0]
            if B < 2:
                continue
            sim = v @ v.T
            mask = 1.0 - torch.eye(B, device=v.device)
            intra_loss = intra_loss + (sim * mask).abs().mean()
            intra_count += 1
        intra_loss = intra_loss / max(intra_count, 1)

        total_contrastive = inter_loss + intra_group_loss + intra_loss

        # 反传（小权重，不主导训练）
        optimizer.zero_grad()
        total_contrastive.backward()
        optimizer.step()

        # 恢复 neuron eval 模式
        for neuron in cortex.neurons.values():
            neuron.eval()

        logger.info(
            f"  contrastive phase: inter={inter_loss.item():.4f}, "
            f"intra_group={intra_group_loss.item():.4f}, "
            f"intra={intra_loss.item():.4f}, neurons={N}"
        )
        print(f"  [contrastive] inter={inter_loss.item():.4f}, "
              f"intra_group={intra_group_loss.item():.4f}, "
              f"intra={intra_loss.item():.4f}, neurons={N}")
        return total_contrastive.item()

    def _train_multimodal_ensemble(
        self, modality: str, sample: dict, tokenizer_hub
    ) -> tuple:
        """P8: 多模态 ensemble 共振训练。

        与推理路径一致：所有注册了该模态的 neuron 参与共振，
        weighted_logits 作为最终输出计算 loss，反传到所有参与 neuron。

        Args:
            modality: 模态类型（image/audio/video）
            sample: 多模态训练样本（含 input_ids, target_ids）
            tokenizer_hub: P7 TokenizerHub

        Returns:
            (loss, ppl) or (None, None) on failure
        """
        import torch
        import torch.nn.functional as F

        cortex = self.cortex
        if cortex is None:
            logger.debug(f"  [{modality}] cortex 未初始化")
            return None, None

        # 找出所有支持该模态的 neuron
        mm_nids = [
            nid for nid, neuron in cortex.neurons.items()
            if modality in neuron.mm_projections and modality in neuron.mm_lm_heads
        ]
        if not mm_nids:
            logger.debug(f"  [{modality}] 无 neuron 支持该模态")
            return None, None

        # 收集所有可训练参数
        trainable_params = []
        for nid in mm_nids:
            neuron = cortex.neurons[nid]
            if modality in neuron.mm_projections:
                trainable_params.extend(neuron.mm_projections[modality].parameters())
            if modality in neuron.mm_lm_heads:
                trainable_params.extend(neuron.mm_lm_heads[modality].parameters())

        if not trainable_params:
            logger.debug(f"  [{modality}] 无可训练参数")
            return None, None

        optimizer = torch.optim.AdamW(trainable_params, lr=5e-5)

        input_ids = sample.get("input_ids", [])
        target_ids = sample.get("target_ids", [])
        if not input_ids or not target_ids:
            logger.debug(f"  [{modality}] 无有效训练数据")
            return None, None

        encoder = tokenizer_hub.modal_encoders.get(modality)
        if encoder is None or not hasattr(encoder, "model") or not hasattr(encoder.model, "quantizer"):
            logger.debug(f"  [{modality}] codec 不可用")
            return None, None

        codebook = encoder.model.quantizer.codebook.to(next(cortex.neurons[mm_nids[0]].parameters()).device)
        device = next(cortex.neurons[mm_nids[0]].parameters()).device

        # 构建输入 embedding（每个 neuron 独立投影）
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        input_feat = codebook(input_tensor)

        target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device).unsqueeze(0)
        target_feat = codebook(target_tensor)

        neuron_embeddings: Dict[str, torch.Tensor] = {}
        for nid in mm_nids:
            neuron = cortex.neurons[nid]
            input_emb = neuron.encode_multimodal_input(input_feat, modality)
            target_emb = neuron.encode_multimodal_input(target_feat, modality)
            full_emb = torch.cat([input_emb, target_emb[:, :-1, :]], dim=1)
            neuron_embeddings[nid] = full_emb

        # 训练模式
        for nid in mm_nids:
            cortex.neurons[nid].train()
        optimizer.zero_grad()

        # ensemble forward（共振）—— 与推理路径完全一致
        result = cortex.ensemble.forward(
            neuron_embeddings=neuron_embeddings,
            return_logits=True,
            active_filter=True,
            active_nids=mm_nids,
            mm_logits_modality=modality,
        )

        # 取加权 logits 计算 loss
        if "weighted_logits" not in result:
            logger.debug(f"  [{modality}] ensemble 未返回 weighted_logits")
            return None, None

        logits = result["weighted_logits"]
        shift_logits = logits[:, -len(target_ids):, :].contiguous()
        shift_targets = target_tensor.contiguous()

        vocab_size = logits.size(-1)
        shift_targets = shift_targets.clamp(0, vocab_size - 1)

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            ignore_index=-100,
        )
        loss.backward()
        optimizer.step()

        # 恢复 eval 模式
        for nid in mm_nids:
            cortex.neurons[nid].eval()

        ppl = math.exp(min(loss.item(), 20))
        return loss.item(), ppl

    def _sleep_phase_knowledge_integration(self, report: SleepReport) -> dict:
        """Phase 3: REM — 知识整合。

        调用 SleepConsolidator 执行：
        - 重放高共振场状态（replay buffer）
        - 强化强 side_channels（共激活高的连接 ×1.1）
        - 修剪弱 side_channels（weight < 0.01）
        - 更新 fingerprint
        - 遗忘弱共激活 pair
        """
        if self._sleep_consolidator is None or self.cortex is None:
            logger.info("  Phase 3: sleep_consolidator 或 cortex 未注入，跳过")
            return {"status": "skipped"}

        coaction = getattr(self.cortex, "coaction", None)
        result = self._sleep_consolidator.consolidate(
            neurons=self.cortex.neurons,
            coactivation_tracker=coaction,
            current_step=self._current_step,
        )

        logger.info(
            f"  Phase 3: 知识整合完成 — "
            f"重放 {result.get('replayed_states', 0)} 状态, "
            f"强化 {result.get('channels_reinforced', 0)} 连接, "
            f"修剪 {result.get('channels_pruned', 0)} 连接, "
            f"更新 {result.get('fingerprints_updated', 0)} fingerprint"
        )

        report.evolution_events += result.get("channels_reinforced", 0)
        return result

    def _sleep_phase_knowledge_distillation(self, report: SleepReport) -> dict:
        """Phase 3.5: 知识蒸馏 — 将累积知识转化为训练数据。

        当前：最小实现，记录待处理样本数。
        未来：从 SleepConsolidator 的高共振状态生成合成训练数据。
        """
        if self._feed_engine is None:
            return {"status": "skipped"}

        pending = 0
        if hasattr(self._feed_engine, "get_pending_count"):
            try:
                pending = self._feed_engine.get_pending_count()
            except Exception:
                pass
        if pending > 0:
            logger.info(f"  Phase 3.5: {pending} 个待处理样本将在下次训练中使用")
        return {"status": "ok", "pending_samples": pending}

    def _sleep_phase_evaluation(self, report: SleepReport) -> dict:
        """Phase 4: 清醒准备 — 自我评估。

        评估 Cortex 神经元质量，检测凋亡候选并执行清理。
        """
        logger.info("  Phase 4: 评估 Cortex 神经元质量...")
        health = self._evaluate_cortex_quality(report)

        n_neurons = health.get("n_neurons", 0)
        status = health.get("status", "unknown")
        logger.info(f"  Phase 4: {n_neurons} neurons, status={status}")

        return health

    def _evaluate_cortex_quality(self, report: SleepReport) -> dict:
        """
        P7: 评估 Cortex 神经元质量。

        使用 lifecycle.apoptosis.check_activation 检查激活率。
        激活计数从 coaction（CoactivationTracker）获取，而非硬编码 0。
        """
        health = {
            "n_neurons": len(self.cortex.neurons),
            "neurons": {},
            "status": "healthy",
        }

        # 检查每个神经元的激活率
        apoptosed = []
        if self._lifecycle is not None:
            try:
                total_rounds = max(1, self._current_step)
                # 从 coaction 获取真实激活计数（修复硬编码 0 的 bug）
                coaction = getattr(self.cortex, "coaction", None)
                activation_counts = {}
                if coaction is not None:
                    activation_counts = getattr(coaction, "_activation_counts", {})

                for nid in list(self.cortex.neurons.keys()):
                    activation_count = activation_counts.get(nid, 0)
                    triggered = self._lifecycle.apoptosis.check_activation(
                        nid, activation_count, total_rounds
                    )
                    health["neurons"][nid] = {
                        "activation_count": activation_count,
                        "total_rounds": total_rounds,
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

                    # 执行清理：从 cortex/ensemble 移除 + 删除 ckpt
                    if self.cortex is not None:
                        for nid in apoptosed:
                            try:
                                self.cortex.remove_neuron(nid, delete_ckpt=True)
                                report.recommendations.append(
                                    f"[凋亡] 神经元 {nid} 已清理"
                                )
                            except Exception as ce:
                                logger.warning(f"  凋亡清理失败 {nid}: {ce}")
            except Exception as e:
                logger.warning(f"  凋亡检查失败: {e}")

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
