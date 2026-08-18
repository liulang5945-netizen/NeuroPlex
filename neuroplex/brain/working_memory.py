"""WorkingMemory — 前额叶工作记忆（P6-4）。

人脑参考：
  前额叶皮层（PFC）维持当前任务的上下文信息，跨多轮决策保持连贯。
  - 短时记忆：维持最近 N 秒的感知/动作（电活动依赖，易失）
  - 注意力回放：对重要内容加强维持
  - 容量限制：米勒数 7±2，超出后旧内容被遗忘

态极实现：
  WorkingMemory 维护 token-id 滑动窗口（默认最近 512 token）。
  - 每次 generate 后，把 prompt + generated 追加到 memory
  - 下次 generate 时，memory 内容作为前缀拼到 input_ids 前面
  - 超出 max_tokens 时，FIFO 丢弃最老 token
  - 可选：重要性加权（用户标记的"重要"轮次衰减更慢）

接入方式：
  Cortex.set_working_memory(memory) 注册后，generate() 自动维护和注入 memory。
  未注册时（默认）完全无状态，向后兼容。

Usage:
    mem = WorkingMemory(max_tokens=512)
    cortex.set_working_memory(mem)
    # 第一次调用
    cortex.generate("你好")  # memory 记录 prompt + 响应
    # 第二次调用，memory 自动作为前缀
    cortex.generate("你叫什么")  # 模型能看到上一轮内容
"""
from __future__ import annotations

from collections import deque
from typing import Optional, List
import torch


class WorkingMemory:
    """前额叶工作记忆 — token-id 滑动窗口维持上下文。

    Attributes:
        max_tokens: 窗口容量（默认 512，约 2-3 轮对话）
        buffer: deque(maxlen=max_tokens) 存储 token IDs
        round_marks: 记录每轮对话的起止位置（便于衰减/回放）
    """

    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens
        self.buffer: deque = deque(maxlen=max_tokens)
        # round_marks[i] = (start_idx, end_idx, importance) 第 i 轮的 token 范围
        # 注意：buffer 是 deque，旧 token 会被丢弃；round_marks 用相对当前 buffer 的索引
        self.round_marks: List[tuple] = []
        self.current_round_start: int = 0

    def reset(self) -> None:
        """清空工作记忆（新会话开始时调用）。"""
        self.buffer.clear()
        self.round_marks.clear()
        self.current_round_start = 0

    def append_round(
        self,
        prompt_ids: List[int],
        generated_ids: List[int],
        importance: float = 1.0,
    ) -> None:
        """追加一轮对话到工作记忆。

        Args:
            prompt_ids: 本轮的 prompt token IDs
            generated_ids: 本轮生成的 token IDs
            importance: 重要性权重（1.0=普通，>1.0=重要，<1.0=可遗忘）
        """
        # 记录本轮的起止位置（在追加前）
        round_start = len(self.buffer)
        # 追加 prompt 和 generated
        self.buffer.extend(prompt_ids)
        self.buffer.extend(generated_ids)
        round_end = len(self.buffer)
        # 调整起始位置（若 deque 满了，旧 token 已被丢弃，索引会偏移）
        dropped = max(0, round_start - len(self.buffer))
        round_start -= dropped
        round_end = min(round_end, len(self.buffer))
        # 清理无效的旧 round_marks（已被 deque 丢弃的部分）
        self._cleanup_stale_marks(round_end - round_start)
        self.round_marks.append((round_start, round_end, importance))

    def _cleanup_stale_marks(self, current_round_size: int) -> None:
        """清理已被 deque 丢弃的旧 round_marks。"""
        if not self.round_marks:
            return
        # 简单策略：若 round_marks 数量过多（>20），保留最近 20 轮
        if len(self.round_marks) > 20:
            self.round_marks = self.round_marks[-20:]

    def get_context_ids(self) -> List[int]:
        """获取当前工作记忆的全部 token IDs（用作下次 generate 的前缀）。"""
        return list(self.buffer)

    def get_context_tensor(self, device: torch.device) -> torch.Tensor:
        """获取工作记忆的 tensor 形式（[1, N]）。"""
        ids = self.get_context_ids()
        if not ids:
            return None
        return torch.tensor([ids], dtype=torch.long, device=device)

    def __len__(self) -> int:
        return len(self.buffer)

    def is_empty(self) -> bool:
        return len(self.buffer) == 0

    def get_summary(self) -> dict:
        """返回工作记忆摘要（诊断用）。"""
        return {
            "total_tokens": len(self.buffer),
            "max_tokens": self.max_tokens,
            "n_rounds": len(self.round_marks),
            "rounds": [
                {"start": s, "end": e, "importance": imp, "size": e - s}
                for s, e, imp in self.round_marks[-5:]  # 最近 5 轮
            ],
        }

    def save(self, path: str) -> None:
        """保存工作记忆到磁盘（会话暂停时调用）。"""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "max_tokens": self.max_tokens,
            "buffer": list(self.buffer),
            "round_marks": self.round_marks,
        }, path)

    @classmethod
    def load(cls, path: str) -> "WorkingMemory":
        """从磁盘加载工作记忆（会话恢复时调用）。"""
        data = torch.load(path, weights_only=False)
        mem = cls(max_tokens=data["max_tokens"])
        mem.buffer.extend(data["buffer"])
        mem.round_marks = data["round_marks"]
        return mem
