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
        # 相似度门控。2026-08-14 验收实测（R11）：原默认 0.3 使 STDP 从未生效——
        # 各 neuron 的场写入方向（含跨规格投影后的统一空间）cosine 实测 ±0.03，
        # 0.3 阈值下所有 pair 被门控，睡眠期 apply 恒空转（离散/连续路径皆然）。
        # 改为 0.0：方向"同向即强化、反向即门控"——sim 仍作为 delta 的乘数，
        # 方向一致的 pair 天然获得更大更新（语义不变，仅去掉过度约束）。
        similarity_threshold: float = 0.0,
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

        # C25-B：持久化共激活统计（突触结构演化的驱动数据，跨会话）
        # (pre_id, post_id) -> {"count": 累积共激活次数, "total_sim": sim 总和,
        #                       "last_update": 最近一次统计的 round}
        # 语义与 apply_updates 一致：pre 先于 post 发放 → (pre, post) 有向对
        self._coactivation_stats: dict = {}
        # 结构演化阈值（修剪/生长）
        self.grow_count_threshold = 5      # 共激活次数 ≥ 此值 → 视为高共激活
        self.grow_sim_threshold = 0.3      # 平均 sim ≥ 此值 → 视为方向一致
        self.prune_count_threshold = 2     # 共激活次数 < 此值 → 视为低共激活
        self.structure_last_updated: int = 0

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

    def accumulate_coactivation(self) -> int:
        """把本轮 firing_history 中的 (pre, post) 有向对累积进持久化统计。

        语义与 STDP 规则一致：A 先于 B 发放（A.round < B.round）→ (A, B) 是
        有向 pre→post 对，累积 count + 方向 sim（供结构生长/修剪判断）。

        幂等：累积后 firing_history 由调用方 clear_history 清空，同一批发放
        不会被重复累积。

        Returns:
            新增/更新的 (pre, post) 对数量
        """
        # 按 neuron 聚合发放记录，按 round 排序
        by_neuron: dict = {}
        for nid, hist in self._firing_history.items():
            records = sorted(hist, key=lambda r: r.round_num)
            by_neuron[nid] = records

        added = 0
        nids = list(by_neuron.keys())
        for i, pre_id in enumerate(nids):
            for post_id in nids[i:]:
                if pre_id == post_id:
                    continue
                for pre_fire in by_neuron[pre_id]:
                    for post_fire in by_neuron[post_id]:
                        # 有向：pre 先于 post 发放才累积（LTP 方向）
                        if pre_fire.round_num >= post_fire.round_num:
                            continue
                        # 方向 sim（cosine，与 STDPRule.compute_weight_update 一致）
                        v_pre = pre_fire.field_vector
                        v_post = post_fire.field_vector
                        if v_pre.dim() == 1:
                            v_pre = v_pre.unsqueeze(0)
                        if v_post.dim() == 1:
                            v_post = v_post.unsqueeze(0)
                        sim = float(((v_pre * v_post).sum(dim=-1) /
                                     (v_pre.norm(dim=-1) * v_post.norm(dim=-1) + 1e-8)).mean().item())
                        key = (str(pre_id), str(post_id))
                        entry = self._coactivation_stats.setdefault(
                            key, {"count": 0, "total_sim": 0.0, "last_update": post_fire.round_num})
                        entry["count"] += 1
                        entry["total_sim"] += sim
                        entry["last_update"] = max(entry["last_update"], post_fire.round_num)
                        added += 1
        return added

    def get_coactivation_stats(self, pre_id: str, post_id: str) -> dict:
        """查询 (pre, post) 的累积共激活统计。"""
        key = (str(pre_id), str(post_id))
        return self._coactivation_stats.get(key, {"count": 0, "total_sim": 0.0, "last_update": 0})

    @torch.no_grad()
    def apply_structure_updates(
        self,
        neurons: dict,
        min_weight_prune: float = 0.01,
    ) -> dict:
        """突触结构演化：通道条目修剪 + 生长（C25-B，突触可塑性从权重缩放升级为结构演化）。

        生物参考：睡眠期突触规模调节——长期高共激活的连接强化（已在 STDP
        权重缩放 + SleepConsolidator 强化覆盖），长期低共激活的连接被修剪，
        高共激活但缺失的连接长出新的（神经发生/突触生长）。

        规则（保守、可逆性优先）：
        - 修剪：通道存在但 (post_id, pre_id) 共激活 count < prune_count_threshold
          **且** 通道权重 L1 均值 < min_weight_prune → 删除条目（弱连接 + 无
          共激活证据 → 清除；权重大的通道即使统计低也保留，防误删已学习连接）
        - 生长：peer 在 neurons 中、count ≥ grow_count_threshold 且
          avg_sim ≥ grow_sim_threshold，但通道缺失 → 建立新通道（邻居相似
          初始化：与目标 peer 共激活最相似的已有通道权重 + 小噪声）

        Args:
            neurons: {neuron_id: ResonanceNeuron}
            min_weight_prune: 修剪的权重下限（L1 均值低于此值才可修剪）

        Returns:
            {"pruned": 修剪通道数, "grown": 生长通道数, "pruned_keys": [...], "grown_keys": [...]}
        """
        stats = {"pruned": 0, "grown": 0, "pruned_keys": [], "grown_keys": []}

        # ── 1. 修剪：低共激活 + 弱权重通道条目删除 ──
        for post_id, post_neuron in neurons.items():
            for ch_dict, ctype in ((getattr(post_neuron, "excite_channels", {}), "excite"),
                                   (getattr(post_neuron, "inhibit_channels", {}), "inhibit")):
                for pre_id in list(ch_dict.keys()):
                    cstat = self.get_coactivation_stats(pre_id, post_id)
                    if cstat["count"] >= self.prune_count_threshold:
                        continue
                    linear = ch_dict[pre_id]
                    w_mean = float(linear.weight.data.abs().mean().item())
                    if w_mean >= min_weight_prune:
                        continue  # 权重仍强 → 保留（防误删已学习连接）
                    # 删除通道条目 + 关联 scale param / bias buffer
                    del ch_dict[pre_id]
                    for suffix in (f"{ctype}_scale_{pre_id}", f"{ctype}_bias_{pre_id}"):
                        for reg_name in list(post_neuron._parameters.keys()):
                            if reg_name == suffix:
                                del post_neuron._parameters[reg_name]
                        for reg_name in list(post_neuron._buffers.keys()):
                            if reg_name == suffix:
                                del post_neuron._buffers[reg_name]
                    stats["pruned"] += 1
                    stats["pruned_keys"].append(f"{post_id}<-{pre_id}({ctype})")

        # ── 2. 生长：高共激活缺失通道建立（邻居相似初始化）──
        # 预计算每个 post 已有通道 peer 列表（供"邻居"相似性匹配）
        for post_id, post_neuron in neurons.items():
            # 该 post 已有的所有通道 peer（excite + inhibit）
            existing_peers = sorted(
                set(getattr(post_neuron, "excite_channels", {}).keys())
                | set(getattr(post_neuron, "inhibit_channels", {}).keys()))
            # 已覆盖的 (pre, post) 对（两个方向都算，避免重复建通道）
            covered = set()
            for ch_dict in (getattr(post_neuron, "excite_channels", {}),
                            getattr(post_neuron, "inhibit_channels", {})):
                covered.update(ch_dict.keys())

            for pre_id, cstat in list(self._coactivation_stats.items()):
                pre, post = pre_id
                if post != post_id or pre not in neurons:
                    continue
                if pre in covered:
                    continue
                if cstat["count"] < self.grow_count_threshold:
                    continue
                avg_sim = cstat["total_sim"] / max(cstat["count"], 1)
                if avg_sim < self.grow_sim_threshold:
                    continue
                # 生长通道：邻居相似初始化
                self._grow_channel(post_neuron, pre, neurons[pre], existing_peers)
                covered.add(pre)
                stats["grown"] += 1
                stats["grown_keys"].append(f"{post_id}<-{pre}(excite)")

        return stats

    def _grow_channel(
        self,
        post_neuron: nn.Module,
        pre_id: str,
        pre_neuron: nn.Module,
        existing_peers: list,
    ) -> None:
        """生长一条新通道（邻居相似初始化）。

        若 post 已有指向其他 peer 的 excite 通道，找与目标 pre 共激活统计
        最相似的已有 peer（共同 pre 的共激活模式），以其通道权重为初始值
        + 小噪声；无邻居则标准 init_std 初始化。init_scale 沿用
        establish_side_channel 默认（50.0）。
        """
        init_weight = None
        if existing_peers:
            # 邻居 = 与 pre 共享最多共激活统计的已有通道 peer
            best_peer, best_score = None, -1.0
            for peer in existing_peers:
                if peer not in getattr(post_neuron, "excite_channels", {}):
                    continue
                # 相似度 = (peer, pre) 与 (peer, 目标) 的共激活重叠
                overlap = 0.0
                for (a, b), s in self._coactivation_stats.items():
                    if a == peer:
                        if b == pre_id:
                            overlap += s["count"]
                        else:
                            overlap += s["count"] * 0.05  # 其他 peer 弱贡献
                if overlap > best_score:
                    best_score, best_peer = overlap, peer
            if best_peer is not None:
                init_weight = post_neuron.excite_channels[best_peer].weight.data.clone()

        # 直接构造 Linear（维度：pre.field_dim → post.hidden_size）
        src_dim = pre_neuron.config.field_dim
        dst_dim = post_neuron.config.hidden_size
        channel = nn.Linear(src_dim, dst_dim, bias=False)
        if init_weight is not None and init_weight.shape == channel.weight.shape:
            channel.weight.data = init_weight + torch.randn_like(init_weight) * 0.005
        else:
            nn.init.normal_(channel.weight, std=0.01)
        post_neuron.excite_channels[pre_id] = channel
        scale_param = nn.Parameter(torch.tensor(50.0))
        post_neuron.register_parameter(f"excite_scale_{pre_id}", scale_param)
        post_neuron.register_buffer(f"excite_bias_{pre_id}", torch.zeros(1))

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
            "coactivation_pairs": len(self._coactivation_stats),
            "coactivation_events": sum(s["count"] for s in self._coactivation_stats.values()),
        }

    def get_state_dict(self) -> dict:
        """C25-B：持久化共激活统计（突触结构演化的长期数据，跨会话）。

        与 C25-D replay buffer 持久化同一模式：coactivation_stats 是结构
        修剪/生长的依据，必须跨会话保留；firing_history 是短期运行时数据，
        重启后重新积累。
        """
        return {
            "coactivation_stats": {
                f"{pre}|{post}": dict(v)
                for (pre, post), v in self._coactivation_stats.items()
            },
            "grow_count_threshold": self.grow_count_threshold,
            "grow_sim_threshold": self.grow_sim_threshold,
            "prune_count_threshold": self.prune_count_threshold,
            "structure_last_updated": self.structure_last_updated,
        }

    def load_state_dict(self, state: dict) -> None:
        """从 dict 恢复状态（C25-B：跨会话共激活统计恢复）。"""
        self._coactivation_stats = {}
        for k, v in (state.get("coactivation_stats") or {}).items():
            pre, post = k.split("|", 1)
            self._coactivation_stats[(pre, post)] = dict(v)
        self.grow_count_threshold = state.get("grow_count_threshold", self.grow_count_threshold)
        self.grow_sim_threshold = state.get("grow_sim_threshold", self.grow_sim_threshold)
        self.prune_count_threshold = state.get("prune_count_threshold", self.prune_count_threshold)
        self.structure_last_updated = state.get("structure_last_updated", 0)
        # firing_history 不恢复（短期运行时数据）
