"""神经元生命周期管理（人脑启发：凋亡与新生）。

人脑参考：
- 凋亡 (Apoptosis): 弱连接神经元被清除，保持系统健康
- 新生 (Neurogenesis): 海马齿状回成年后仍有新生，填补知识盲区
- 幼稚态: 新生神经元初始高可塑性，逐步成熟

态极实现：
- ApoptosisTracker: 追踪连续高 PPL 神经元，触发凋亡
- NeurogenesisTrigger: 检测知识盲区，触发新生
- MaturityTracker: 管理新生神经元的成熟度，控制学习率和共振权重
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("Taiji.Lifecycle")


@dataclass
class ApoptosisTracker:
    """
    神经元凋亡追踪器（人脑启发：弱连接神经元被清除）。

    追踪连续 K 次评估 PPL > threshold 的神经元，
    标记为"凋亡"后从 ensemble 移除并清理资源。

    触发条件：
    1. 连续 failure_threshold 次 PPL > ppl_threshold（宽限期 grace_evals 之后）
    2. 或长期（>min_rounds_observed）激活率 < activation_ratio

    宽限期保护：前 grace_evals 次评估不触发凋亡。
    随机初始化或 lm_head 未训练的神经元 PPL 天然很高（幼稚态），
    宽限期给它们时间通过 feed+sleep 积累经验后再评估凋亡。
    """

    ppl_threshold: float = 200.0
    failure_threshold: int = 3
    activation_ratio: float = 0.05
    min_rounds_observed: int = 20
    grace_evals: int = 10  # 宽限期：前 N 次评估不触发凋亡

    # nid -> 连续失败计数
    _failure_counts: dict = field(default_factory=dict)
    # nid -> True 表示已凋亡
    _apoptosed: dict = field(default_factory=dict)
    # nid -> 累计评估次数
    _eval_counts: dict = field(default_factory=dict)

    def record_ppl(self, neuron_id: str, ppl: float) -> bool:
        """记录一次 PPL 评估，返回是否触发凋亡。

        Args:
            neuron_id: 神经元 ID
            ppl: 本次评估的 PPL

        Returns:
            True 如果该神经元触发凋亡
        """
        if self._apoptosed.get(neuron_id, False):
            return True  # 已凋亡

        # 累计评估次数，宽限期内不触发凋亡
        self._eval_counts[neuron_id] = self._eval_counts.get(neuron_id, 0) + 1
        if self._eval_counts[neuron_id] <= self.grace_evals:
            return False

        if ppl > self.ppl_threshold:
            self._failure_counts[neuron_id] = self._failure_counts.get(neuron_id, 0) + 1
            if self._failure_counts[neuron_id] >= self.failure_threshold:
                self._apoptosed[neuron_id] = True
                logger.warning(
                    "神经元 %s 连续 %d 次 PPL > %.1f（当前 %.1f），触发凋亡",
                    neuron_id,
                    self._failure_counts[neuron_id],
                    self.ppl_threshold,
                    ppl,
                )
                return True
        else:
            # 评估通过，重置失败计数
            self._failure_counts[neuron_id] = 0

        return False

    def check_activation(self, neuron_id: str, activation_count: int, total_rounds: int) -> bool:
        """检查激活率是否过低，触发凋亡。

        Args:
            neuron_id: 神经元 ID
            activation_count: 神经元被激活的总次数
            total_rounds: 总轮次

        Returns:
            True 如果该神经元触发凋亡
        """
        if self._apoptosed.get(neuron_id, False):
            return True
        if total_rounds < self.min_rounds_observed:
            return False

        ratio = activation_count / total_rounds if total_rounds > 0 else 0
        if ratio < self.activation_ratio:
            self._apoptosed[neuron_id] = True
            logger.warning(
                "神经元 %s 激活率 %.3f < %.3f（%d/%d 轮），触发凋亡",
                neuron_id,
                ratio,
                self.activation_ratio,
                activation_count,
                total_rounds,
            )
            return True
        return False

    def is_apoptosed(self, neuron_id: str) -> bool:
        return self._apoptosed.get(neuron_id, False)

    def get_apoptosis_candidates(self) -> list:
        """获取所有已凋亡的神经元 ID。"""
        return [nid for nid, ap in self._apoptosed.items() if ap]

    def cleanup_neuron(
        self,
        neuron_id: str,
        ckpt_path: Optional[str] = None,
        ensemble: Optional[Any] = None,
    ) -> bool:
        """清理凋亡神经元的资源。

        Args:
            neuron_id: 神经元 ID
            ckpt_path: ckpt 文件路径，若提供则删除
            ensemble: ResonanceEnsemble 实例，若提供则从 neurons 移除

        Returns:
            True 如果清理成功
        """
        if not self.is_apoptosed(neuron_id):
            return False

        # 从 ensemble 移除
        if ensemble is not None and hasattr(ensemble, "neurons"):
            if neuron_id in ensemble.neurons:
                del ensemble.neurons[neuron_id]
                logger.info("已从 ensemble 移除凋亡神经元 %s", neuron_id)

        # 删除 ckpt 文件
        if ckpt_path is not None and os.path.exists(ckpt_path):
            try:
                os.remove(ckpt_path)
                logger.info("已删除凋亡神经元 ckpt: %s", ckpt_path)
            except OSError as e:
                logger.error("删除 ckpt %s 失败: %s", ckpt_path, e)
                return False

        # 清理其他神经元的 side_channels
        if ensemble is not None and hasattr(ensemble, "neurons"):
            key = str(neuron_id)
            for other_nid, other_neuron in ensemble.neurons.items():
                if hasattr(other_neuron, "excite_channels") and key in other_neuron.excite_channels:
                    del other_neuron.excite_channels[key]
                if hasattr(other_neuron, "inhibit_channels") and key in other_neuron.inhibit_channels:
                    del other_neuron.inhibit_channels[key]

        return True

    def reset(self, neuron_id: str) -> None:
        """重置某神经元的失败计数和评估计数（不复活已凋亡的）。"""
        self._failure_counts.pop(neuron_id, None)
        self._eval_counts.pop(neuron_id, None)


@dataclass
class MaturityTracker:
    """
    神经元成熟度追踪器（人脑启发：新生神经元幼稚态）。

    新生神经元初始为"幼稚态"：
    - 高学习率（base_lr × maturity_lr_multiplier）
    - 低共振权重（maturity_min_resonance_weight）
    - 逐步成熟：学习率衰减，共振权重提升

    成熟过程：
    maturity_counter 从 0 递增到 maturity_rounds
    - maturity_counter=0: 完全幼稚（lr×3, weight=0.1）
    - maturity_counter=maturity_rounds: 完全成熟（lr×1, weight=1.0）
    """

    maturity_rounds: int = 100  # 成熟所需轮数
    maturity_lr_multiplier: float = 3.0  # 幼稚态学习率倍数
    maturity_min_resonance_weight: float = 0.1  # 幼稚态最小共振权重

    # nid -> 成熟度计数器
    _maturity: dict = field(default_factory=dict)

    def register_new(self, neuron_id: str) -> None:
        """注册新生神经元（初始 maturity=0）。"""
        self._maturity[neuron_id] = 0
        logger.info("注册新生神经元 %s（幼稚态开始）", neuron_id)

    def tick(self, neuron_id: str) -> None:
        """递增神经元的成熟度计数器。"""
        if neuron_id in self._maturity:
            self._maturity[neuron_id] += 1

    def get_maturity_ratio(self, neuron_id: str) -> float:
        """获取成熟度比例 [0, 1]。

        0 = 完全幼稚
        1 = 完全成熟
        """
        if neuron_id not in self._maturity:
            return 1.0  # 未注册视为已成熟
        return min(1.0, self._maturity[neuron_id] / self.maturity_rounds)

    def get_lr_multiplier(self, neuron_id: str) -> float:
        """获取学习率倍数（幼稚态高，成熟态低）。"""
        ratio = self.get_maturity_ratio(neuron_id)
        # 线性衰减：幼稚态 maturity_lr_multiplier，成熟态 1.0
        return self.maturity_lr_multiplier * (1 - ratio) + 1.0 * ratio

    def get_resonance_weight(self, neuron_id: str) -> float:
        """获取共振权重（幼稚态低，成熟态高）。"""
        ratio = self.get_maturity_ratio(neuron_id)
        # 线性增长：幼稚态 maturity_min_resonance_weight，成熟态 1.0
        return self.maturity_min_resonance_weight + (1.0 - self.maturity_min_resonance_weight) * ratio

    def is_mature(self, neuron_id: str) -> bool:
        """是否已完全成熟。"""
        return self.get_maturity_ratio(neuron_id) >= 1.0

    def tick_all(self) -> None:
        """递增所有注册神经元的成熟度。"""
        for nid in list(self._maturity.keys()):
            self._maturity[nid] += 1
            if self._maturity[nid] >= self.maturity_rounds and not self.is_mature(nid):
                logger.info("神经元 %s 已完全成熟", nid)


@dataclass
class NeurogenesisTrigger:
    """
    神经元新生触发器（人脑启发：海马齿状回新生神经元）。

    检测"知识盲区"：
    1. 某 domain 持续高错误率
    2. CoactivationTracker 检测到孤立激活模式

    触发新生流程：
    1. 从 teacher 蒸馏新神经元 ckpt
    2. 初始化为"幼稚态"
    3. 加入 ensemble
    """

    # domain -> 连续高错误率计数
    _domain_error_counts: dict = field(default_factory=dict)
    error_rate_threshold: float = 0.5
    error_count_for_trigger: int = 8  # 需要连续 8 次高错误率才触发（避免过快扩张）

    def record_domain_error(self, domain: str, error_rate: float) -> bool:
        """记录某 domain 的错误率，返回是否触发新生。

        Args:
            domain: 域名
            error_rate: 错误率 [0, 1]

        Returns:
            True 如果该 domain 需要新生神经元
        """
        if error_rate > self.error_rate_threshold:
            self._domain_error_counts[domain] = self._domain_error_counts.get(domain, 0) + 1
            if self._domain_error_counts[domain] >= self.error_count_for_trigger:
                logger.warning(
                    "domain %s 连续 %d 次错误率 > %.1f（当前 %.3f），触发新生",
                    domain,
                    self._domain_error_counts[domain],
                    self.error_rate_threshold,
                    error_rate,
                )
                # 重置计数，避免重复触发
                self._domain_error_counts[domain] = 0
                return True
        else:
            self._domain_error_counts[domain] = 0

        return False

    def detect_isolated_patterns(
        self,
        coactivation_tracker: Any,
        min_isolation_ratio: float = 0.8,
        maturity_tracker: Any = None,
        min_maturity_ratio: float = 0.1,
        min_total_pairs: int = 5,
    ) -> list:
        """检测孤立激活模式（可能需要新生神经元填补）。

        如果某神经元的共激活 pair 中 >80% 都低于阈值，
        说明它"孤立"，可能需要新生一个相关神经元来协同。

        Args:
            coactivation_tracker: CoactivationTracker 实例
            min_isolation_ratio: 低频 pair 占比阈值（默认 0.8）
            maturity_tracker: MaturityTracker 实例（可选，跳过幼稚态 neuron）
            min_maturity_ratio: 最小成熟度比例（默认 0.1，即至少经过 10 轮 tick）
            min_total_pairs: 最小共激活 pair 数（默认 5）。
                低于此数的 neuron 跳过——共激活矩阵还未充分填充时，
                所有 pair 频率都低，会导致假阳性"孤立"判断。

        Returns:
            孤立神经元 ID 列表
        """
        if not hasattr(coactivation_tracker, "_slow_matrix"):
            return []

        isolated = []
        # 统计每个神经元的总 pair 数和低频 pair 数
        pair_stats: dict = {}  # nid -> [total, low_freq]
        for (i, j), freq in coactivation_tracker._slow_matrix.items():
            for nid in [i, j]:
                # 跳过幼稚态神经元（maturity < min_maturity_ratio）：
                # 新 neuron 天然没有共激活历史，100% 的 pair 都是低频，
                # 会形成"检测孤立 → 创建新 neuron → 新 neuron 又孤立"的正反馈
                if maturity_tracker is not None:
                    if maturity_tracker.get_maturity_ratio(nid) < min_maturity_ratio:
                        continue
                if nid not in pair_stats:
                    pair_stats[nid] = [0, 0]
                pair_stats[nid][0] += 1
                if freq < coactivation_tracker.forget_threshold * 10:
                    pair_stats[nid][1] += 1

        for nid, (total, low_freq) in pair_stats.items():
            # min_total_pairs: 共激活数据不足时跳过低频判断，
            # 避免假阳性 "孤立" 导致神经元爆炸
            if total >= min_total_pairs and low_freq / total > min_isolation_ratio:
                isolated.append(nid)

        return isolated


@dataclass
class LifecycleManager:
    """
    生命周期管理器：统一管理凋亡、新生、成熟度。

    使用方式：
        lifecycle = LifecycleManager()
        # 每轮评估
        for nid, ppl in ppl_results.items():
            lifecycle.apoptosis.record_ppl(nid, ppl)
        # 清理凋亡神经元
        for nid in lifecycle.apoptosis.get_apoptosis_candidates():
            lifecycle.apoptosis.cleanup_neuron(nid, ckpt_path, ensemble)
        # 检测新生需求
        if lifecycle.neurogenesis.record_domain_error("math", 0.6):
            # 创建新神经元...
            lifecycle.maturity.register_new(new_nid)
        # 每轮递增成熟度
        lifecycle.maturity.tick_all()
    """

    apoptosis: ApoptosisTracker = field(default_factory=ApoptosisTracker)
    neurogenesis: NeurogenesisTrigger = field(default_factory=NeurogenesisTrigger)
    maturity: MaturityTracker = field(default_factory=MaturityTracker)

    def step(
        self,
        ppl_results: dict,
        ensemble: Any,
        ckpt_dir: Optional[str] = None,
    ) -> dict:
        """执行一次生命周期步进。

        Args:
            ppl_results: {neuron_id: ppl}
            ensemble: ResonanceEnsemble
            ckpt_dir: ckpt 目录路径

        Returns:
            dict with:
            - apoptosed: 凋亡的神经元列表
            - new_needs: 需要新生的 domain 列表
        """
        apoptosed = []
        for nid, ppl in ppl_results.items():
            if self.apoptosis.record_ppl(nid, ppl):
                ckpt_path = (
                    os.path.join(ckpt_dir, f"neuron_{nid}.pt")
                    if ckpt_dir else None
                )
                self.apoptosis.cleanup_neuron(nid, ckpt_path, ensemble)
                apoptosed.append(nid)

        # 递增所有注册神经元的成熟度
        self.maturity.tick_all()

        return {
            "apoptosed": apoptosed,
        }
