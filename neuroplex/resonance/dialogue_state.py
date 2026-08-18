"""S12: 多轮对话状态管理。

替代 cortex.py 的前缀拼接方案，用 field_state 持久化 + 对话轮次 token
实现真正的多轮对话状态管理。

核心机制：
1. 每轮对话结束后保存 field_state 快照（save_round_state）
2. 下一轮开始时加载上一轮的 field_state（load_round_state）
3. 对话轮次 token 标记轮次边界，让模型识别新轮次
4. 滑动窗口保留最近 N 轮的 field_state（避免无限累积）

人脑启发：海马体在对话间保持工作记忆，每轮对话更新海马状态，
而非把所有历史对话文本重新读一遍（前缀拼接的低效做法）。
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

import torch


class DialogueState:
    """多轮对话状态管理器。

    管理 field_state 的多轮持久化，替代文本前缀拼接。

    Usage:
        dialogue = DialogueState(max_rounds=5)
        # 第 1 轮
        dialogue.start_round()
        result = cortex.generate("你好")
        dialogue.end_round(field)  # 保存 field_state

        # 第 2 轮
        dialogue.start_round()  # 加载上一轮 field_state
        result = cortex.generate("刚才我说了什么？")  # 模型通过 field_state 记忆
        dialogue.end_round(field)
    """

    # 对话轮次 token（在 prompt 前插入，标记新轮次开始）
    # 这些是 general tokenizer 的特殊 token id（由 cortex 初始化时注册）
    ROUND_START_TOKEN = "<|round_start|>"
    ROUND_END_TOKEN = "<|round_end|>"

    def __init__(
        self,
        max_rounds: int = 5,
        round_token_id: Optional[int] = None,
    ):
        """初始化对话状态管理器。

        Args:
            max_rounds: 保留最近 N 轮的 field_state（滑动窗口）
                        N=5 时，第 6 轮开始时丢弃第 1 轮的 state
                        0 = 不持久化（每轮独立，向后兼容）
            round_token_id: 对话轮次 token 的 general vocab id
                            None 时不在 prompt 前插入轮次标记
        """
        self.max_rounds = max_rounds
        self.round_token_id = round_token_id

        # 滑动窗口：最近 N 轮的 field_state 快照
        self._round_states: deque = deque(maxlen=max_rounds if max_rounds > 0 else 0)

        # 当前轮次编号（0-indexed）
        self._current_round: int = -1

        # 对话历史（文本，用于 debug/日志，不参与推理）
        self._dialogue_history: List[Dict[str, str]] = []

    @property
    def current_round(self) -> int:
        """当前轮次编号（0-indexed，-1 表示未开始）。"""
        return self._current_round

    @property
    def n_rounds(self) -> int:
        """已完成的轮次数。"""
        return len(self._round_states)

    def start_round(self, field: Optional[Any] = None) -> Optional[Dict[str, torch.Tensor]]:
        """开始新一轮对话。

        如果有保存的 field_state，加载最近一轮的 state 到 field。
        这让模型通过 field_state 隐式记忆上一轮的上下文，
        而非把历史文本拼到 prompt 前面。

        Args:
            field: ResonanceField 实例（用于加载 state）
                   None 时只返回 state_dict，不加载到 field

        Returns:
            加载的 state_dict（None 表示无历史或 max_rounds=0）
        """
        self._current_round += 1

        if self.max_rounds <= 0 or not self._round_states:
            return None

        # 加载最近一轮的 field_state
        last_state = self._round_states[-1]
        if field is not None:
            field.load_round_state(last_state)
        return last_state

    def end_round(self, field: Any) -> None:
        """结束当前轮次，保存 field_state 快照。

        Args:
            field: ResonanceField 实例（保存其 state）
        """
        if self.max_rounds <= 0:
            return

        state_snapshot = field.save_round_state()
        self._round_states.append(state_snapshot)

    def add_dialogue_entry(self, role: str, content: str) -> None:
        """添加对话历史条目（仅用于日志/debug，不参与推理）。

        Args:
            role: "user" 或 "assistant"
            content: 对话内容
        """
        self._dialogue_history.append({"role": role, "content": content})
        # 限制历史长度，避免无限增长
        if len(self._dialogue_history) > self.max_rounds * 2 + 2:
            self._dialogue_history = self._dialogue_history[-(self.max_rounds * 2 + 2):]

    def get_dialogue_history(self) -> List[Dict[str, str]]:
        """获取对话历史（用于 debug/日志）。"""
        return list(self._dialogue_history)

    def should_prepend_round_token(self) -> bool:
        """是否应该在 prompt 前插入轮次标记 token。

        只在第 2 轮及以后插入（第 1 轮无需标记）。
        """
        return self.round_token_id is not None and self._current_round > 0

    def prepend_round_token(self, general_ids: List[int]) -> List[int]:
        """在 prompt general_ids 前插入轮次标记 token。

        Args:
            general_ids: 原始 prompt 的 general token ids

        Returns:
            带 round_start token 前缀的 general_ids
        """
        if not self.should_prepend_round_token():
            return general_ids
        return [self.round_token_id] + general_ids

    def reset(self) -> None:
        """重置对话状态（新会话开始时调用）。"""
        self._round_states.clear()
        self._dialogue_history.clear()
        self._current_round = -1

    def get_state_dict(self) -> dict:
        """序列化为可持久化的 dict（保存到 checkpoint）。"""
        return {
            "max_rounds": self.max_rounds,
            "round_token_id": self.round_token_id,
            "current_round": self._current_round,
            "round_states": list(self._round_states),
            "dialogue_history": self._dialogue_history,
        }

    def load_state_dict(self, state: dict) -> None:
        """从 dict 恢复状态。"""
        self.max_rounds = state.get("max_rounds", self.max_rounds)
        self.round_token_id = state.get("round_token_id", self.round_token_id)
        self._current_round = state.get("current_round", -1)
        self._round_states = deque(
            state.get("round_states", []),
            maxlen=self.max_rounds if self.max_rounds > 0 else 0,
        )
        self._dialogue_history = state.get("dialogue_history", [])
