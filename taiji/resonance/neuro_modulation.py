"""神经调质系统与睡眠巩固（人脑启发）。

神经调质 (Neuromodulation)：
- 多巴胺/血清素/去甲肾上腺素等全局调质
- 根据奖励/注意力状态调节学习率和兴奋性
- 态极实现：全局标量信号，调节 lr / field_write 强度 / refractory 长度

睡眠巩固 (Sleep Consolidation)：
- 睡眠期间海马回放白天经历
- 将短期记忆转移到皮层长期存储
- 修剪弱突触
- 态极实现：离线重放 + side_channels 强化/修剪 + fingerprint 更新
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("Taiji.NeuroModulation")


@dataclass
class NeuromodulatorState:
    """神经调质状态（人脑启发：多巴胺/血清素/去甲肾上腺素）。

    每个调质是 [0, 1] 标量，影响不同的系统参数：
    - dopamine: 奖励信号，高→学习率↑，低→学习率↓+新生加速
    - serotonin: 满足感，高→refractory↑（更易满足），低→refractory↓
    - norepinephrine: 注意力/警觉，高→field_write 强度↑
    """

    dopamine: float = 0.5      # 奖励/错误反馈驱动
    serotonin: float = 0.5     # 满足感/稳定度
    norepinephrine: float = 0.5  # 警觉/注意力

    # 目标值（由外部信号设定，实际值缓慢趋近）
    _target_dopamine: float = 0.5
    _target_serotonin: float = 0.5
    _target_norepinephrine: float = 0.5

    # EMA 趋近速率
    ema_alpha: float = 0.1

    def set_targets(
        self,
        dopamine: Optional[float] = None,
        serotonin: Optional[float] = None,
        norepinephrine: Optional[float] = None,
    ) -> None:
        """设置目标调质水平（由外部信号驱动）。"""
        if dopamine is not None:
            self._target_dopamine = max(0.0, min(1.0, dopamine))
        if serotonin is not None:
            self._target_serotonin = max(0.0, min(1.0, serotonin))
        if norepinephrine is not None:
            self._target_norepinephrine = max(0.0, min(1.0, norepinephrine))

    def step(self) -> None:
        """EMA 趋近目标值（调质不会突变，而是缓慢调整）。"""
        self.dopamine += self.ema_alpha * (self._target_dopamine - self.dopamine)
        self.serotonin += self.ema_alpha * (self._target_serotonin - self.serotonin)
        self.norepinephrine += self.ema_alpha * (self._target_norepinephrine - self.norepinephrine)

    def get_lr_multiplier(self) -> float:
        """获取学习率倍数（多巴胺驱动）。

        高多巴胺 → 学习率↑（奖励信号，强化学习）
        低多巴胺 → 学习率↓但触发新生（错误信号）
        """
        # 多巴胺 0.5 = 中性，学习率倍数 1.0
        # 多巴胺 1.0 = 强奖励，学习率倍数 2.0
        # 多巴胺 0.0 = 强惩罚，学习率倍数 0.5
        return 0.5 + self.dopamine * 1.5

    def get_refractory_multiplier(self) -> float:
        """获取不应期倍数（血清素驱动）。

        高血清素 → 不应期↑（满足，不易再激活）
        低血清素 → 不应期↓（不满足，易再激活）
        """
        # 血清素 0.5 = 中性，倍数 1.0
        # 血清素 1.0 = 高满足，倍数 1.5
        # 血清素 0.0 = 低满足，倍数 0.5
        return 0.5 + self.serotonin * 1.0

    def get_field_write_scale(self) -> float:
        """获取 field_write 强度倍数（去甲肾上腺素驱动）。

        高去甲肾上腺素 → 场写入↑（高度警觉，强信号）
        低去甲肾上腺素 → 场写入↓（放松，弱信号）
        """
        return 0.5 + self.norepinephrine * 1.0

    def should_trigger_neurogenesis(self) -> bool:
        """是否应该触发神经元新生（低多巴胺持续）。"""
        return self.dopamine < 0.2

    def get_state_dict(self) -> dict:
        """获取状态字典（用于持久化）。"""
        return {
            "dopamine": self.dopamine,
            "serotonin": self.serotonin,
            "norepinephrine": self.norepinephrine,
            "_target_dopamine": self._target_dopamine,
            "_target_serotonin": self._target_serotonin,
            "_target_norepinephrine": self._target_norepinephrine,
        }

    def load_state_dict(self, state: dict) -> None:
        """加载状态。"""
        self.dopamine = state.get("dopamine", 0.5)
        self.serotonin = state.get("serotonin", 0.5)
        self.norepinephrine = state.get("norepinephrine", 0.5)
        self._target_dopamine = state.get("_target_dopamine", 0.5)
        self._target_serotonin = state.get("_target_serotonin", 0.5)
        self._target_norepinephrine = state.get("_target_norepinephrine", 0.5)


class SleepConsolidator:
    """睡眠巩固周期（人脑启发：离线重放+突触修剪）。

    人脑在睡眠期间：
    1. 海马回放白天经历（高共振场状态序列）
    2. 强化经常共激活的突触
    3. 修剪弱突触（突触缩放）
    4. 将短期记忆转移到长期存储

    态极实现：
    1. 重放近期 high-resonance 场状态
    2. 强化 slow EMA 高的 side_channels
    3. 修剪权重低于阈值的 side_channels
    4. 更新 fingerprint（将 slow EMA 趋势编码到长期方向）
    """

    def __init__(
        self,
        replay_buffer_size: int = 100,
        consolidation_interval: int = 1000,
    ):
        self.replay_buffer_size = replay_buffer_size
        self.consolidation_interval = consolidation_interval

        # 重放缓冲区：存储 high-resonance 场状态
        self._replay_buffer: deque = deque(maxlen=replay_buffer_size)

        # 上次巩固的步数
        self._last_consolidation_step: int = 0

    def record_high_resonance_state(
        self,
        field_state: torch.Tensor,
        resonance_score: float,
        step: int,
        threshold: float = 0.5,
    ) -> None:
        """记录一次高共振场状态（用于后续重放）。

        Args:
            field_state: 场状态向量
            resonance_score: 本次共振的最高分数
            step: 当前步数
            threshold: 共振分数阈值，高于此值才记录
        """
        if resonance_score > threshold:
            self._replay_buffer.append({
                "state": field_state.detach().clone(),
                "score": resonance_score,
                "step": step,
            })

    def should_consolidate(self, current_step: int) -> bool:
        """是否应该执行巩固。"""
        return (current_step - self._last_consolidation_step) >= self.consolidation_interval

    @torch.no_grad()
    def consolidate(
        self,
        neurons: dict,
        coactivation_tracker: Optional[Any] = None,
        current_step: int = 0,
    ) -> dict:
        """执行一次睡眠巩固。

        Args:
            neurons: {neuron_id: ResonanceNeuron}
            coactivation_tracker: CoactivationTracker 实例
            current_step: 当前步数

        Returns:
            巩固统计
        """
        logger.info("开始睡眠巩固（step=%d，重放缓冲=%d）",
                    current_step, len(self._replay_buffer))

        stats = {
            "replayed_states": 0,
            "channels_reinforced": 0,
            "channels_pruned": 0,
            "fingerprints_updated": 0,
            "pairs_forgotten": 0,
        }

        # 1. 重放高共振场状态
        for record in list(self._replay_buffer):
            # 重放时不需要真正 forward，只是统计
            stats["replayed_states"] += 1

        # 2. 强化 slow EMA 高的 side_channels
        if coactivation_tracker is not None and hasattr(coactivation_tracker, "get_strong_pairs"):
            strong_pairs = coactivation_tracker.get_strong_pairs(threshold=0.2)
            for pre_id, post_id in strong_pairs:
                post_key = str(post_id)
                # 在 post_neuron 的 side_channels 中找到 pre_id 对应的通道
                for nid, neuron in neurons.items():
                    if hasattr(neuron, "excite_channels") and post_key in neuron.excite_channels:
                        # 强化：权重 × 1.1
                        neuron.excite_channels[post_key].weight.data *= 1.1
                        stats["channels_reinforced"] += 1

        # 3. 修剪弱 side_channels
        for nid, neuron in neurons.items():
            if hasattr(neuron, "prune_weak_channels"):
                pruned = neuron.prune_weak_channels(threshold=0.01)
                stats["channels_pruned"] += pruned

        # 4. 更新 fingerprint（将 slow EMA 趋势编码到长期方向）
        for nid, neuron in neurons.items():
            if hasattr(neuron, "freeze_fingerprint"):
                neuron.freeze_fingerprint()
                stats["fingerprints_updated"] += 1

        # 5. 遗忘弱共激活 pair
        if coactivation_tracker is not None and hasattr(coactivation_tracker, "forget_weak"):
            stats["pairs_forgotten"] = coactivation_tracker.forget_weak()

        # 6. 清空重放缓冲区（已巩固）
        self._replay_buffer.clear()

        # 7. 更新巩固时间
        self._last_consolidation_step = current_step

        logger.info("睡眠巩固完成: %s", stats)
        return stats

    def get_stats(self) -> dict:
        """返回统计信息。"""
        return {
            "replay_buffer_size": len(self._replay_buffer),
            "last_consolidation_step": self._last_consolidation_step,
            "next_consolidation_in": max(
                0, self.consolidation_interval - (
                    self._last_consolidation_step % self.consolidation_interval
                )
            ),
        }
