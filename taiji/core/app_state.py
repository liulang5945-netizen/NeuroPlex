
"""
集中式应用状态管理
使用细粒度锁替代全局单锁，推理请求不阻塞训练

每个关键资源拥有独立的锁：
- model_lock: 模型/推理引擎的读写锁
- train_lock: 训练状态锁
- publish_lock: 发布状态锁
"""
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Optional

from taiji.core.config import get_external_path
from taiji.tools.rag import RAGKnowledgeBase

logger = logging.getLogger("AppState")


@dataclass
class AppState:
    """
    集中式应用状态管理，使用细粒度锁替代全局单锁
    推理请求不会阻塞训练状态查询，反之亦然。
    """

    # 模型相关（高频访问）
    trainer: Optional[object] = None
    model: Optional[object] = None
    tokenizer: Optional[object] = None
    _loaded_model_name: Optional[str] = field(default=None, repr=False)

    # 态极多模态引擎（当加载 ModelSelf 模型时自动启用）
    taiji_engine: Optional[object] = None
    _is_taiji: bool = field(default=False, repr=False)
    _taiji_bridge: Optional[object] = field(default=None, repr=False)

    # 训练/微调状态
    is_training: bool = False
    stop_training_requested: bool = False
    pause_training_requested: bool = False
    _trainer_ref: Optional[object] = field(default=None, repr=False)

    # 模型发布状态
    publishing: bool = False
    stop_publishing_requested: bool = False

    # RAG 知识库
    rag_kb: object = None

    # 启动状态（线程安全保护）
    startup_error: Optional[str] = None
    startup_complete: bool = False

    # 模型切换状态（异步切换用）
    switch_status: str = "idle"  # idle / switching / success / error
    switch_message: str = ""
    switch_error: str = ""

    # 锁（实例级，避免类变量共享问题）
    _model_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    train_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    publish_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _startup_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _switch_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # 后台任务注册（用于优雅关闭）
    background_tasks: list = field(default_factory=list)

    # ======================== 模型切换操作 ========================

    def update_switch_status(self, status: str, message: str = "", error: str = ""):
        """更新模型切换状态（线程安全）"""
        with self._switch_lock:
            self.switch_status = status
            self.switch_message = message
            self.switch_error = error

    def get_switch_status(self) -> dict:
        """获取当前模型切换状态"""
        with self._switch_lock:
            return {
                "status": self.switch_status,
                "message": self.switch_message,
                "error": self.switch_error,
            }

    def reset_switch_status(self):
        """重置切换状态为 idle"""
        with self._switch_lock:
            self.switch_status = "idle"
            self.switch_message = ""
            self.switch_error = ""

    def update_rag_kb(self, kb):
        self.rag_kb = kb

    # ======================== 模型操作 ========================

    def get_model_info(self):
        """快速获取模型信息"""
        return {
            "loaded": self.model is not None,
            "model_name": self._loaded_model_name,
            "ready": self.startup_complete and not self.startup_error,
        }

    def unload_model(self):
        """卸载当前模型，释放所有显存和内存（用于热切换前清理）"""
        with self._model_lock:
            import gc

            # 0. 卸载态极引擎（如果存在）
            if self.taiji_engine is not None:
                try:
                    if hasattr(self.taiji_engine, 'cancel'):
                        self.taiji_engine.cancel()
                except Exception as e:
                    logger.warning(f"卸载态极引擎时出错（可忽略）: {e}")
                self.taiji_engine = None
                self._is_taiji = False

            if self._taiji_bridge is not None:
                try:
                    self._taiji_bridge.cleanup()
                except Exception as e:
                    logger.warning(f"Cleanup taiji bridge failed: {e}")
                self._taiji_bridge = None

            # 1. 卸载 GGUF 引擎（如果存在）
            old_trainer = self.trainer
            if old_trainer is not None:
                try:
                    if hasattr(old_trainer, 'unload'):
                        old_trainer.unload()
                except Exception as e:
                    logger.warning(f"卸载 GGUF 引擎时出错（可忽略）: {e}")

            # 2. 释放 PyTorch 模型
            if self.model is not None:
                old_model = self.model
                self.model = None
                try:
                    self.tokenizer = None
                    import torch
                    if hasattr(old_model, 'to'):
                        try:
                            old_model.to('cpu')
                        except Exception:
                            pass
                    del old_model
                except Exception as e:
                    logger.warning(f"释放 PyTorch 模型时出错: {e}")

            self.tokenizer = None
            self.trainer = None
            self._loaded_model_name = None
            self.startup_complete = False
            self.startup_error = None

            # 3. 强制垃圾回收
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass

            logger.info("旧模型已完全卸载，显存/内存已释放")

    def update_model(self, model, tokenizer, trainer, model_name: str):
        """更新模型（写操作）。调用前应先执行 unload_model() 清理旧模型。"""
        with self._model_lock:
            if self.model is not None and self.model is not model:
                self.trainer = None
                self.model = None
                self.tokenizer = None
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.model = model
            self.tokenizer = tokenizer
            self.trainer = trainer
            self._loaded_model_name = model_name

    def set_taiji_engine(self, taiji_engine):
        """设置态极多模态引擎（当加载 ModelSelf 模型时调用）"""
        self.taiji_engine = taiji_engine
        self._is_taiji = taiji_engine is not None
        if taiji_engine:
            logger.info("态极多模态引擎已启用")

    def set_taiji_bridge(self, bridge):
        """设置态极桥接层（新架构）"""
        self._taiji_bridge = bridge
        self._is_taiji = bridge is not None and bridge.is_initialized

    def is_taiji(self) -> bool:
        """当前是否使用态极引擎"""
        # 优先检查桥接层
        if self._taiji_bridge is not None and self._taiji_bridge.is_initialized:
            return True
        return self._is_taiji and self.taiji_engine is not None

    def get_taiji_engine(self):
        """获取态极多模态引擎实例"""
        return self.taiji_engine

    def get_taiji_bridge(self):
        """获取态极桥接层实例"""
        return self._taiji_bridge

    def get_trainer(self):
        return self.trainer

    def get_tokenizer(self):
        return self.tokenizer

    # ======================== 训练操作 ========================

    def try_start_training(self) -> bool:
        """尝试获取训练锁，成功返回 True"""
        acquired = self.train_lock.acquire(blocking=False)
        if not acquired:
            return False
        self.is_training = True
        self.stop_training_requested = False
        self.pause_training_requested = False
        return True

    def finish_training(self):
        """释放训练锁"""
        self.is_training = False
        self.stop_training_requested = False
        self.pause_training_requested = False
        self._trainer_ref = None
        try:
            self.train_lock.release()
        except RuntimeError:
            pass

    def training_context(self, logger_name: str = "Training"):
        """
        训练锁上下文管理器 — 确保同一时刻只有一个训练进程操作模型。

        用法:
            with app_state.training_context("SleepEngine"):
                # 训练代码
                ...
        如果获取锁失败（已有训练在运行），抛出 RuntimeError。
        """
        return _TrainingContext(self, logger_name)

    # ======================== 发布操作 ========================

    def try_start_publishing(self) -> bool:
        """尝试获取发布锁"""
        acquired = self.publish_lock.acquire(blocking=False)
        if not acquired:
            return False
        self.publishing = True
        self.stop_publishing_requested = False
        return True

    def finish_publishing(self):
        """释放发布锁"""
        self.publishing = False
        self.stop_publishing_requested = False
        try:
            self.publish_lock.release()
        except RuntimeError:
            pass

    def force_reset_publishing(self) -> dict:
        """紧急强制重置发布状态（修复死锁）"""
        was_publishing = self.publishing
        locked = self.publish_lock.locked()
        self.stop_publishing_requested = True
        self.publishing = False
        self.stop_publishing_requested = False
        if locked:
            try:
                self.publish_lock.release()
            except RuntimeError:
                pass
        logger.warning(f"发布状态已强制重置 (was_publishing={was_publishing}, lock_was_held={locked})")
        return {
            "was_publishing": was_publishing,
            "lock_was_held": locked,
        }

    # ======================== 启动状态 ========================

    def mark_starting(self):
        with self._startup_lock:
            self.startup_complete = False
            self.startup_error = None

    def mark_started(self):
        with self._startup_lock:
            self.startup_complete = True
            self.startup_error = None
        logger.info("模型 API 服务已就绪！")

    def mark_startup_failed(self, error_msg: str):
        with self._startup_lock:
            self.startup_error = error_msg
            self.startup_complete = True

    def register_background_task(self, thread: threading.Thread):
        self.background_tasks.append(thread)



class _TrainingContext:
    """训练锁上下文管理器 — 确保同一时刻只有一个训练进程操作模型"""

    def __init__(self, state: "AppState", logger_name: str):
        self._state = state
        self._logger_name = logger_name

    def __enter__(self):
        if not self._state.try_start_training():
            raise RuntimeError(
                f"[{self._logger_name}] 无法获取训练锁，"
                f"另一个训练正在进行 (is_training={self._state.is_training})"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._state.finish_training()
        return False  # 不吞异常


# 全局状态实例
app_state = AppState()
app_state.rag_kb = RAGKnowledgeBase(persist_dir=get_external_path("rag_data"))
