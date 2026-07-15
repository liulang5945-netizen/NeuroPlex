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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger("SleepEngine")


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
        
        self._data_dir_ready = False
        self._load_history()
        
        logger.info(f"SleepEngine initialized: auto={self.config.auto_sleep_enabled}, interval={self.config.sleep_interval_hours}h")
    
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
        """Phase 2: 模型训练 — 用收集的数据在线微调态极模型"""
        try:
            from taiji.agent_ext.data_collector import get_collector
            from taiji.data.data_generator import generate_bulk_react_data, generate_bulk_conversation_data
            
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
            logger.info("  DataCollector not available, skipping")
    
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

            # 判断是否为态极模型
            from taiji.architecture import ModelSelf
            if not isinstance(model, ModelSelf):
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
        """Phase 3: 知识整合 — 进化引擎 + 用户画像"""
        # 进化引擎
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
        
        # 用户画像
        try:
            from taiji.infra.user_profile import get_user_profile
            profile = get_user_profile()
            
            suggestions = profile.get_task_pattern_suggestions()
            report.user_patterns_updated = len(suggestions)
            
            if suggestions:
                report.recommendations.extend(suggestions)
                
        except ImportError:
            logger.info("  UserProfile not available, skipping")

    def _sleep_phase_knowledge_distillation(self, report: SleepReport):
        """Phase 3.5: 知识蒸馏 — 累积知识 → 训练数据"""
        try:
            from taiji.agent_ext.knowledge_learner import get_knowledge_learner
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
        """Phase 4: 自我评估"""
        try:
            from taiji.infra.self_evaluator import get_self_evaluator
            from taiji.life.evolution_engine import get_evolution_engine
            
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
            return {"status": "unknown"}

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
