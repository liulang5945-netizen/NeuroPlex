"""
态极躯干 (BodyCore)
====================

态极的骨骼系统 — 管理态极的所有资源引用。

态极原生实现，专门为态极服务。
所有引擎通过 BodyCore 获取模型和分词器，不直接导入外部模块。

包含三个身体模块：
- limbs（行动系统）：工具调用、文件操作
- metabolism（代谢系统）：硬件感知、资源管理
- senses（感知系统）：API 输入、终端、前端
"""
import os
import logging
import threading
from typing import Optional, Any, Callable

logger = logging.getLogger("Taiji.Body")


class BodyCore:
    """
    态极的躯干 — 资源管理器

    管理态极的所有外部资源引用：
    - 模型（大脑）
    - 分词器（语言中枢）
    - 设备（肌肉）
    - 动作提供者（手脚）
    - 感知器（感官）
    - 代谢器（代谢）

    核心原则：依赖注入，不依赖全局单例。
    """

    def __init__(self):
        self._model = None
        self._cortex = None  # 大脑（Cortex）- 认知主体
        self._tokenizer = None
        self._device = "cpu"
        self._action_provider = None
        self._data_collector = None
        self._lock = threading.Lock()
        self._callbacks = {
            "model_change": [],
        }

        # 身体模块
        self._limbs = None  # 行动系统
        self._metabolism = None  # 代谢系统
        self._senses = None  # 感知系统

    # ── 模型管理 ──

    @property
    def model(self):
        """获取当前模型"""
        return self._model

    # ── 大脑（Cortex）管理 ──

    @property
    def cortex(self):
        """获取大脑（Cortex）"""
        return self._cortex

    def set_cortex(self, cortex):
        """设置大脑（取代旧的 set_model/self._model）"""
        with self._lock:
            old = self._cortex
            self._cortex = cortex
            logger.info(f"Cortex set: {type(cortex).__name__}")
            # 通知感知系统
            if self._senses is not None and hasattr(self._senses, 'set_cortex'):
                self._senses.set_cortex(cortex)

    def set_model(self, model):
        """设置模型（已废弃，请使用 set_cortex）"""
        logger.warning("set_model() is deprecated, use set_cortex() instead")
        # 如果传入的是 Cortex 实例，自动路由到 set_cortex
        if type(model).__name__ == 'Cortex':
            self.set_cortex(model)
        else:
            with self._lock:
                old_model = self._model
                self._model = model
                logger.info(f"Model switched: {type(old_model).__name__} -> {type(model).__name__}")
                # 通知订阅者
                for cb in self._callbacks.get("model_change", []):
                    try:
                        cb(old_model, model)
                    except Exception as e:
                        logger.error(f"Model change callback error: {e}")

    def on_model_change(self, callback: Callable):
        """注册模型切换回调"""
        self._callbacks.setdefault("model_change", []).append(callback)

    # ── 分词器管理 ──

    @property
    def tokenizer(self):
        """获取当前分词器"""
        return self._tokenizer

    def set_tokenizer(self, tokenizer):
        """设置分词器"""
        with self._lock:
            self._tokenizer = tokenizer

    # ── 设备管理 ──

    @property
    def device(self) -> str:
        """获取当前计算设备"""
        return self._device

    def set_device(self, device: str):
        """设置计算设备"""
        self._device = device
        logger.info(f"Device set to: {device}")

    def auto_detect_device(self) -> str:
        """自动检测最佳设备"""
        try:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
        except ImportError:
            self._device = "cpu"
        logger.info(f"Auto-detected device: {self._device}")
        return self._device

    # ── 动作提供者（手脚）──

    @property
    def action_provider(self):
        """获取动作提供者"""
        return self._action_provider

    def set_action_provider(self, provider):
        """设置动作提供者（态极的手脚）"""
        self._action_provider = provider
        logger.info(f"ActionProvider set: {type(provider).__name__}")

    # ── 数据收集器 ──

    @property
    def data_collector(self):
        """获取数据收集器"""
        return self._data_collector

    def set_data_collector(self, collector):
        """设置数据收集器"""
        self._data_collector = collector

    # ── 身体模块管理 ──

    @property
    def limbs(self):
        """获取行动系统（手脚）"""
        if self._limbs is None:
            from taiji.body import limbs
            self._limbs = limbs
        return self._limbs

    @property
    def metabolism(self):
        """获取代谢系统"""
        if self._metabolism is None:
            from taiji.body import metabolism
            self._metabolism = metabolism
        return self._metabolism

    @property
    def senses(self):
        """获取感知系统（感官）"""
        if self._senses is None:
            from taiji.body import senses
            self._senses = senses
        return self._senses

    # ── 资源检查 ──

    def check_resources(self) -> dict:
        """检查系统资源状态"""
        import psutil
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            result = {
                "cpu_percent": cpu_percent,
                "memory_total_gb": round(memory.total / (1024**3), 1),
                "memory_used_gb": round(memory.used / (1024**3), 1),
                "memory_percent": memory.percent,
                "device": self._device,
                "has_model": self._model is not None,
                "has_tokenizer": self._tokenizer is not None,
                "has_action_provider": self._action_provider is not None,
            }
        except ImportError:
            result = {
                "device": self._device,
                "has_model": self._model is not None,
                "has_tokenizer": self._tokenizer is not None,
                "has_action_provider": self._action_provider is not None,
            }

        # GPU 信息
        try:
            import torch
            if torch.cuda.is_available():
                result["gpu_name"] = torch.cuda.get_device_name(0)
                result["gpu_memory_total_gb"] = round(
                    torch.cuda.get_device_properties(0).total_mem / (1024**3), 1
                )
                result["gpu_memory_used_gb"] = round(
                    torch.cuda.memory_allocated(0) / (1024**3), 1
                )
        except Exception as e:
            logger.debug("core: non-critical %s", e, exc_info=True)

        return result

    def is_healthy(self) -> bool:
        """检查态极是否健康"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 95:
                return False
            if psutil.cpu_percent(interval=0.1) > 95:
                return False
        except ImportError:
            pass
        # 大脑必须加载
        if self._cortex is not None and hasattr(self._cortex, 'is_loaded'):
            if not self._cortex.is_loaded:
                return False
        return True

    def cleanup(self):
        """资源清理"""
        with self._lock:
            if self._model is not None:
                try:
                    import gc
                    import torch
                    if hasattr(self._model, 'to'):
                        self._model.to('cpu')
                    del self._model
                    self._model = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info("Model cleaned up")
                except Exception as e:
                    logger.warning(f"Cleanup error: {e}")

            # Cortex 清理（在 model 清理之后）
            if self._cortex is not None:
                try:
                    import gc
                    import torch
                    del self._cortex
                    self._cortex = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info("Cortex cleaned up")
                except Exception as e:
                    logger.warning(f"Cortex cleanup error: {e}")

            self._tokenizer = None
            self._action_provider = None

    def get_status(self) -> dict:
        """获取躯干状态"""
        # 使用懒加载属性，确保模块被加载
        limbs_ok = self.limbs is not None
        metabolism_ok = self.metabolism is not None
        senses_ok = self.senses is not None

        return {
            "device": self._device,
            "has_model": self._model is not None,
            "model_type": type(self._model).__name__ if self._model else None,
            "has_cortex": self._cortex is not None,
            "cortex_type": type(self._cortex).__name__ if self._cortex else None,
            "has_tokenizer": self._tokenizer is not None,
            "has_action_provider": self._action_provider is not None,
            "healthy": self.is_healthy(),
            "limbs_available": limbs_ok,
            "metabolism_available": metabolism_ok,
            "senses_available": senses_ok,
        }