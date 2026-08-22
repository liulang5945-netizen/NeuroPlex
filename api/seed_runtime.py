"""Seed 原生运行时：加载 / 卸载 / 热切换 / 字节级对话。

``neuroplex/core/app_state.py`` 的对等物：Cortex（neuroplex）保留为可切换的
冻结对照，Seed 是独立可切换的原生运行时。两者互斥：任一时刻聊天主路由
只走其中一个。本模块不导入 ``neuroplex``，切换语义由调用方编排。

对话口径与阶段 1 训练管线一致（``scripts/training/train_seed_corpus.py``）：
对话结构用 ``问：/答：`` 文本标记序列化，会话边界由基底的
``boundary_symbol`` 承担，全程不引入 tokenizer。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("ApiServer.SeedRuntime")

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent.parent / "checkpoints" / "seed_corpus.pt"
)

_TURN_MARKERS = ("\n问：", "问：")


class SeedRuntime:
    """单个 Seed 有机体 + 字节级对话接口（线程安全）。"""

    RUNTIME_TYPE = "seed"

    def __init__(self, model: Any, checkpoint_path: Optional[Path] = None) -> None:
        self.model = model
        self.checkpoint_path = checkpoint_path
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        if self.checkpoint_path is not None:
            return f"seed:{self.checkpoint_path.name}"
        return "seed:scratch"

    @classmethod
    def load(cls, checkpoint_path: Optional[Path | str] = None) -> "SeedRuntime":
        """从 seed-native-v1 检查点装配 Seed（与训练管线同一信封）。"""
        import torch

        from seed import Seed

        path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = Seed.from_checkpoint(checkpoint)
        logger.info("Seed runtime loaded from %s", path)
        return cls(model, path)

    @staticmethod
    def _serialize(prompt: str, history: Sequence[Tuple[str, str]] | None) -> str:
        """沿用训练语料的 问：/答： 标记把多轮对话铺成一段文本。"""
        parts: List[str] = []
        for user, assistant in history or []:
            if user and assistant:
                parts.append(f"问：{user}\n答：{assistant}")
        parts.append(f"问：{prompt}\n答：")
        return "\n".join(parts)

    def chat(
        self,
        prompt: str,
        *,
        history: Sequence[Tuple[str, str]] | None = None,
        max_length: int = 256,
        learn: bool = True,
    ) -> str:
        """生成回复；可选把整段对话作为清醒持续学习写回基底。"""
        text = self._serialize(prompt, history)
        with self._lock:
            raw = self.model.generate(
                text.encode("utf-8"),
                max_length,
                stop_at_boundary=True,
                sample=False,
            )
            answer = raw.decode("utf-8", errors="replace")
            if learn:
                # 多轮上下文由基底持久状态天然承担：整段会话文本一次写回，
                # 与 learn_bytes 的训练语义完全一致。
                self.model.learn_bytes(
                    (text + answer).encode("utf-8"),
                    include_boundary=True,
                )
        for marker in _TURN_MARKERS:
            index = answer.find(marker)
            if index >= 0:
                answer = answer[:index]
        return answer.strip()

    def save(self, path: Optional[Path | str] = None) -> Path:
        """落盘当前状态（默认写回来源检查点）。"""
        import torch

        target = Path(path or self.checkpoint_path or DEFAULT_CHECKPOINT)
        with self._lock:
            torch.save(self.model.checkpoint(), target)
        logger.info("Seed runtime saved to %s", target)
        return target

    def status(self) -> Dict[str, Any]:
        return {
            "runtime_type": self.RUNTIME_TYPE,
            "name": self.name,
            "tick": int(self.model.tick),
            "parameters": int(self.model.parameter_count()),
        }


# ---------------- 进程级单例与热切换 ----------------

_runtime: Optional[SeedRuntime] = None
_runtime_lock = threading.Lock()


def is_seed_active() -> bool:
    return _runtime is not None


def get_seed_runtime() -> Optional[SeedRuntime]:
    return _runtime


def activate_seed(
    checkpoint_path: Optional[Path | str] = None,
) -> SeedRuntime:
    """加载并激活 Seed 运行时（替换既有实例）。"""
    global _runtime
    with _runtime_lock:
        runtime = SeedRuntime.load(checkpoint_path)
        _runtime = runtime
        return runtime


def deactivate_seed() -> None:
    """卸载 Seed 运行时（切回 Cortex 主路径时调用）。"""
    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            logger.info("Seed runtime deactivated (%s)", _runtime.name)
        _runtime = None


def seed_status() -> Dict[str, Any]:
    runtime = _runtime
    if runtime is None:
        return {"runtime_type": "seed", "active": False}
    payload = runtime.status()
    payload["active"] = True
    return payload
