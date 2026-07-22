"""Cortex 启动加载器 — 神经元架构的唯一模型加载入口。

启动时调用 assemble_cortex 装配 Cortex 神经元架构，
将 Cortex 实例注入 app_state.model，作为运行时认知主体。

用法：
    from taiji.core.model_loader import load_model_on_startup
    load_model_on_startup()  # 在 API lifespan 中调用
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("ModelLoader")


def load_model_on_startup() -> None:
    """启动时加载 Cortex 神经元架构到 app_state。

    流程：
    1. 调用 assemble_cortex 装配 Cortex + TokenizerHub + bio 模块
    2. 注入 app_state.model / tokenizer
    3. 构造 SleepEngine 并注入 cortex + modules
    4. 标记启动完成

    失败时标记 startup_error，不抛出异常（让 API 继续运行，端点返回 503）。
    """
    from taiji.core.app_state import app_state

    try:
        logger.info("[ModelLoader] 开始装配 Cortex 神经元架构...")

        # 解析设备
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass

        # 装配 Cortex
        from taiji.loader import assemble_cortex
        neurons_dir = os.environ.get("TAIJI_NEURONS_DIR", "data/neurons")
        cortex, tokenizer, modules = assemble_cortex(
            neurons_dir=neurons_dir,
            device=device,
            max_rounds=3,
            wire_bio_modules=True,
        )

        # 注入 app_state（直接赋值，不调用 update_model 避免 gc 旧模型的副作用）
        app_state.model = cortex
        app_state.tokenizer = tokenizer
        app_state.trainer = None
        app_state._loaded_model_name = "cortex"

        # 构造 SleepEngine 并接线 bio 模块
        try:
            from taiji.life.sleep_engine import SleepEngine, get_sleep_engine
            sleep = get_sleep_engine()
            sleep.set_brain_interfaces(
                cortex=cortex,
                lifecycle=modules.get("lifecycle"),
                sleep_consolidator=modules.get("sleep_consolidator"),
                stdp_tracker=modules.get("stdp_tracker"),
                feed_engine=modules.get("feed_engine"),
                neuromodulator=modules.get("neuromodulator"),
            )
            logger.info("[ModelLoader] SleepEngine 已接线")
        except Exception as e:
            logger.warning(f"[ModelLoader] SleepEngine 接线失败（非致命）: {e}")

        # 构造 FeedEngine（API 端点会显式传 tokenizer_hub，无需注入 cortex）
        try:
            from taiji.life.feed_engine import get_feed_engine
            get_feed_engine()  # 预初始化全局实例
            logger.info("[ModelLoader] FeedEngine 已预初始化")
        except Exception as e:
            logger.warning(f"[ModelLoader] FeedEngine 预初始化失败（非致命）: {e}")

        app_state.mark_started()
        n_neurons = len(cortex.neurons)
        logger.info(
            f"[ModelLoader] Cortex 加载完成: {n_neurons} neurons, device={device}"
        )

    except Exception as e:
        logger.error(f"[ModelLoader] Cortex 加载失败: {e}", exc_info=True)
        app_state.mark_startup_failed(str(e))


def startup_download_progress() -> dict:
    """兼容旧接口：返回空进度（Cortex 不需要下载）。"""
    return {"progress": 100, "status": "done", "message": "Cortex loaded"}
