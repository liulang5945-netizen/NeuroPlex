"""STDP（脉冲时序依赖可塑性）局部学习规则。

人脑参考：
- 突触前神经元在突触后神经元之前发放 → LTP（长时程增强）
- 突触前神经元在突触后神经元之后发放 → LTD（长时程减弱）
- 这是局部学习规则，不需要全局误差信号

态极实现：
- 记录 peer 神经元的 field_vector 时序
- 若 A 在 B 写入前已指向相似方向 → 增强 A→B 通道（LTP）
- 若 A 在 B 之后才指向相似方向 → 减弱 A→B 通道（LTD）
- 形成"因果链"：A 领先 B 则 A 指导 B

与全局反向传播的区别：
- STDP 只更新 side_channels 权重，不影响 Transformer body
- 不需要 loss 信号，纯局部时序驱动
- 可以在推理时（无梯度）进行在线学习
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("Taiji.STDP")


@dataclass
class FiringRecord:
    """单次发放记录。"""
    neuron_id: str
    round_num: int
    field_vector: torch.Tensor  # [B, D] 或 [D]


class STDPRule:
    """STDP 学习规则（单次更新）。

    LTP: Δw = η⁺ · exp(-Δt / τ⁺), Δt = t_post - t_pre > 0 (pre 先于 post)
    LTD: Δw = -η⁻ · exp(Δt / τ⁻), Δt < 0 (post 先于 pre)

    在态极中：
    - "发放时间"= 写入场的轮次
    - "方向相似度"= field_vector 的 cosine 相似度
    - 只对相似度 > threshold 的 pair 应用 STDP（避免噪声）
    """

    def __init__(
        self,
        eta_plus: float = 0.01,   # LTP 学习率
        eta_minus: float = 0.005,  # LTD 学习率（通常小于 LTP）
        tau_plus: float = 2.0,     # LTP 时间常数（轮次）
        tau_minus: float = 2.0,    # LTD 时间常数
        similarity_threshold: float = 0.3,  # 相似度阈值
    ):
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.similarity_threshold = similarity_threshold

    def compute_weight_update(
        self,
        pre_firing: FiringRecord,
        post_firing: FiringRecord,
    ) -> float:
        """计算单次 STDP 权重更新量。

        Args:
            pre_firing: 突触前神经元（A）的发放记录
            post_firing: 突触后神经元（B）的发放记录

        Returns:
            权重更新量（正=LTP，负=LTD，0=不更新）
        """
        # 时间差：Δt = t_post - t_pre
        delta_t = post_firing.round_num - pre_firing.round_num

        # 方向相似度（cosine）
        v_pre = pre_firing.field_vector
        v_post = post_firing.field_vector
        if v_pre.dim() == 1:
            v_pre = v_pre.unsqueeze(0)
        if v_post.dim() == 1:
            v_post = v_post.unsqueeze(0)
        sim = ((v_pre * v_post).sum(dim=-1) /
               (v_pre.norm(dim=-1) * v_post.norm(dim=-1) + 1e-8)).mean().item()

        # 相似度不足，不更新
        if sim < self.similarity_threshold:
            return 0.0

        if delta_t > 0:
            # pre 先于 post → LTP
            return self.eta_plus * float(torch.exp(torch.tensor(-delta_t / self.tau_plus)).item()) * sim
        elif delta_t < 0:
            # post 先于 pre → LTD
            return -self.eta_minus * float(torch.exp(torch.tensor(delta_t / self.tau_minus)).item()) * sim
        else:
            # 同轮次，小幅 LTP（视为同时发放）
            return self.eta_plus * 0.5 * sim


class STDPTracker:
    """
    STDP 追踪器：记录发放历史，应用 STDP 更新到 side_channels。

    使用方式：
        tracker = STDPTracker()
        # 每轮记录发放
        tracker.record_firing(nid, round_num, field_vector)
        # 推理结束后应用更新
        tracker.apply_updates(neuron, peer_neuron)
    """

    def __init__(
        self,
        history_length: int = 5,
        stdp_rule: Optional[STDPRule] = None,
    ):
        self.history_length = history_length
        self.stdp_rule = stdp_rule or STDPRule()
        # neuron_id -> 发放历史 (deque, 最新在右)
        self._firing_history: dict = {}

    def record_firing(
        self,
        neuron_id: str,
        round_num: int,
        field_vector: torch.Tensor,
    ) -> None:
        """记录一次神经元发放。"""
        if neuron_id not in self._firing_history:
            self._firing_history[neuron_id] = deque(maxlen=self.history_length)
        self._firing_history[neuron_id].append(
            FiringRecord(neuron_id, round_num, field_vector.detach().clone())
        )

    def _get_history(self, neuron_id: str) -> list:
        """获取某神经元的发放历史。"""
        return list(self._firing_history.get(neuron_id, []))

    @torch.no_grad()
    def apply_updates(
        self,
        post_neuron: nn.Module,
        pre_neuron_id: str,
    ) -> dict:
        """对 post_neuron 的 side_channels 应用 STDP 更新。

        检查 post_neuron 的每个 side_channel（对应 pre_neuron_id），
        根据 pre 和 post 的发放时序应用 LTP/LTD。

        Args:
            post_neuron: 突触后神经元（拥有 side_channels）
            pre_neuron_id: 突触前神经元 ID（side_channel 的 key）

        Returns:
            更新统计 {channel_key: weight_delta}
        """
        post_history = self._get_history(post_neuron.config.neuron_id or "self")
        pre_history = self._firing_history.get(pre_neuron_id, deque())
        pre_history = list(pre_history)

        if not post_history or not pre_history:
            return {}

        key = str(pre_neuron_id)
        updates = {}

        # 对每对 (pre, post) 发放应用 STDP
        for pre_fire in pre_history:
            for post_fire in post_history:
                delta = self.stdp_rule.compute_weight_update(pre_fire, post_fire)
                if abs(delta) < 1e-6:
                    continue

                # 应用到 excite_channels（LTP 增强，LTD 减弱）
                if hasattr(post_neuron, "excite_channels") and key in post_neuron.excite_channels:
                    linear = post_neuron.excite_channels[key]
                    # 按比例缩放权重
                    scale = 1.0 + delta
                    scale = max(0.5, min(2.0, scale))  # 限制在 [0.5, 2.0]
                    linear.weight.data *= scale
                    updates[f"excite:{key}"] = delta

                # 反向应用到 inhibit_channels（LTD 增强 inhibitory，LTP 减弱）
                if hasattr(post_neuron, "inhibit_channels") and key in post_neuron.inhibit_channels:
                    linear = post_neuron.inhibit_channels[key]
                    # 反向：LTD 增强 inhibitory（因为 pre 落后 post，应该抑制 pre 的未来影响）
                    scale = 1.0 - delta
                    scale = max(0.5, min(2.0, scale))
                    linear.weight.data *= scale
                    updates[f"inhibit:{key}"] = -delta

        return updates

    @torch.no_grad()
    def apply_all_updates(self, neurons: dict) -> dict:
        """对所有神经元的 side_channels 批量应用 STDP 更新。

        Args:
            neurons: {neuron_id: ResonanceNeuron}

        Returns:
            {neuron_id: {channel_key: weight_delta}}
        """
        all_updates = {}
        for post_id, post_neuron in neurons.items():
            post_updates = {}
            # 检查该神经元的所有 side_channels
            for channel_dict in [getattr(post_neuron, "excite_channels", {}),
                                  getattr(post_neuron, "inhibit_channels", {})]:
                for key in channel_dict.keys():
                    pre_id = key
                    if pre_id in neurons or pre_id in self._firing_history:
                        updates = self.apply_updates(post_neuron, pre_id)
                        post_updates.update(updates)
            if post_updates:
                all_updates[post_id] = post_updates

        return all_updates

    def clear_history(self) -> None:
        """清空所有发放历史（推理结束后调用）。"""
        self._firing_history.clear()

    def get_stats(self) -> dict:
        """返回统计信息。"""
        return {
            "neurons_tracked": len(self._firing_history),
            "total_records": sum(len(h) for h in self._firing_history.values()),
        }
