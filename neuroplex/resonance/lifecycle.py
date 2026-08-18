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
    人脑启发的分层神经元凋亡追踪器（v2，2026-08-06 重构）。

    人脑对应：
    - 突触修剪先行：弱突触（side_channels）先被修剪，神经元本体保留（见 prune_synapses）
    - 活动依赖存活（use it or lose it）：存活依赖持续活动，种群内相对竞争（非绝对阈值）
    - 神经营养因子竞争：营养 = 网络贡献（协作边际贡献 + 网络中心度），营养不足者凋亡
    - 凋亡级联：启动（低生存分）→ 隔离观察 → 执行，非一步到位；隔离期可复活
    - 成熟度保护：幼稚态神经元受保护（MaturityTracker 联动）
    - 抑制性神经元保护：皮层抑制性比例稳定，凋亡阈值放宽

    状态机：active → (连续低分) → candidate → isolated → (观察期无反转) → dead
            isolated → (分数回升) → active（可复活）

    PPL 不用固定绝对阈值（如 200）——general 256K 空间与域空间 PPL 口径完全不同，
    固定阈值会误杀全部 general 空间神经元。改为多维生存评分 + 种群相对分位。
    """

    # 相对阈值（种群竞争）
    low_score_threshold: float = 0.4    # 生存分 < 此值视为"低分"
    failure_threshold: int = 3          # 连续 3 轮低分 → candidate（防单轮抖动）
    observe_rounds: int = 10            # 隔离观察期（级联过程，可复活）

    # 状态机（nid -> state）
    _states: dict = field(default_factory=dict)          # active|candidate|isolated|dead
    _failure_counts: dict = field(default_factory=dict)  # nid -> 连续低分计数
    _isolate_since: dict = field(default_factory=dict)   # nid -> 进入隔离的轮次
    _scores: dict = field(default_factory=dict)          # nid -> 最近生存分

    # 兼容旧字段（旧调用方仍可调用 record_ppl / check_activation）
    _apoptosed: dict = field(default_factory=dict)
    _eval_counts: dict = field(default_factory=dict)
    # 旧固定阈值保留为"兜底"（仅当未注入 ppl_percentile 时用于 record_ppl 兼容）
    ppl_threshold: float = 200.0
    failure_threshold_legacy: int = 3
    grace_evals: int = 10
    activation_ratio: float = 0.05
    min_rounds_observed: int = 20

    # ── 多维生存评分 ───────────────────────────────────

    def compute_survival_score(self, metrics: dict) -> float:
        """多维生存评分（0-1，越高越健康）。

        metrics（由 sleep 侧采集注入）：
          activity: 0-1 激活率（归一化到种群最大值）
          ppl_percentile: 0-1 PPL 在种群中的百分位（1=最优）——空间自适应，
                          general 256K 与域空间各自计算分位，不直接比绝对值
          contribution: 0-1 协作边际贡献（A/B 剔除实验：剔除后协作不变差=1；可选）
          connectivity: 0-1 网络中心度（side channel 出入度归一化）
          redundancy: 0-1 与最强同质 neuron 的 field_vector 相似度（1=完全冗余，惩罚项）
          maturity_ratio: 0-1 成熟度（幼稚态 <0.5 → 直接高保护，不判死）
          is_inhibitory: bool 抑制性保护（皮层抑制性比例稳定，放宽贡献/连接要求）
        信号缺失 → 该维度不参与，权重重归一化。
        """
        # 幼稚态保护：未成熟神经元不参与凋亡竞争（人脑：新生神经元有存活窗口）
        maturity = metrics.get("maturity_ratio", 1.0)
        if maturity < 0.5:
            return 1.0

        # 信号 → 权重（缺失信号自动剔除并重归一化）
        dims = {
            "activity": (metrics.get("activity"), 0.25),
            "ppl": (metrics.get("ppl_percentile"), 0.25),
            "contribution": (metrics.get("contribution"), 0.25),
            "connectivity": (metrics.get("connectivity"), 0.15),
            "redundancy": (metrics.get("redundancy"), 0.10),  # 惩罚项
        }
        # 抑制性保护：皮层抑制性比例稳定——"网络贡献"不是其存活要求
        # （抑制性神经元靠功能活性存活，贡献/连接差 → 不惩罚；权重转移给活动+能力）
        if metrics.get("is_inhibitory"):
            dims["contribution"] = (None, 0.25)
            dims["connectivity"] = (None, 0.15)
            dims["activity"] = (metrics.get("activity"), 0.35)
            dims["ppl"] = (metrics.get("ppl_percentile"), 0.35)

        total_w, score = 0.0, 0.0
        for name, (val, w) in dims.items():
            if val is None:
                continue
            total_w += w
            if name == "redundancy":
                score -= w * val  # 冗余惩罚
            else:
                score += w * val
        if total_w <= 0:
            return 0.5
        return max(0.0, min(1.0, score / total_w))

    # ── 种群级状态机步进（睡眠 Phase 4 主入口）──────────

    def step_population(self, metrics_map: Dict[str, dict], step_round: int) -> Dict[str, str]:
        """种群级凋亡状态机步进，返回 {nid: new_state}。

        metrics_map: {nid: metrics}，先计算种群 ppl 百分位（若注入 ppl），再逐 nid 流转。
        """
        # 1. 种群 ppl 百分位（空间自适应：分位在同一空间内计算）
        ppl_vals = {
            nid: m["ppl"] for nid, m in metrics_map.items()
            if m.get("ppl") is not None
        }
        if ppl_vals:
            sorted_ids = sorted(ppl_vals, key=lambda k: ppl_vals[k])
            rank = {nid: i for i, nid in enumerate(sorted_ids)}
            n = len(sorted_ids)
            for nid in metrics_map:
                if nid in ppl_vals and n > 0:
                    # 分位 0-1：PPL 最低（最优）→ 1.0
                    metrics_map[nid]["ppl_percentile"] = 1.0 - rank[nid] / (n - 1) if n > 1 else 1.0

        # 2. 逐 nid 状态流转
        for nid, metrics in metrics_map.items():
            state = self._states.get(nid, "active")
            if state == "dead":
                continue
            score = self.compute_survival_score(metrics)
            self._scores[nid] = score
            low = score < self.low_score_threshold

            if state == "active":
                if low:
                    self._failure_counts[nid] = self._failure_counts.get(nid, 0) + 1
                    if self._failure_counts[nid] >= self.failure_threshold:
                        self._states[nid] = "candidate"  # 凋亡级联启动
                        logger.warning("神经元 %s 生存分 %.2f 连续 %d 轮偏低，进入 candidate",
                                       nid, score, self._failure_counts[nid])
                else:
                    self._failure_counts[nid] = 0
            elif state == "candidate":
                if not low:
                    # 分数回升 → 级联取消
                    self._states[nid] = "active"
                    self._failure_counts[nid] = 0
                    logger.info("神经元 %s 生存分回升 %.2f，取消凋亡级联", nid, score)
                else:
                    # 进入隔离（摘除路由，保留权重，可复活）
                    self._states[nid] = "isolated"
                    self._isolate_since[nid] = step_round
                    logger.warning("神经元 %s 进入隔离（观察 %d 轮）", nid, self.observe_rounds)
            elif state == "isolated":
                if not low:
                    # 激活/能力回升 → 复活
                    self._states[nid] = "active"
                    self._failure_counts[nid] = 0
                    self._isolate_since.pop(nid, None)
                    logger.info("神经元 %s 复活（生存分回升 %.2f）", nid, score)
                elif step_round - self._isolate_since.get(nid, step_round) >= self.observe_rounds:
                    # 观察期满 → 试复活（重新加入路由，下一轮决定生死）
                    # 人脑：凋亡级联的最后确认，给神经元最后一次证明自己的机会
                    self._states[nid] = "trial"
                    logger.warning("神经元 %s 隔离观察 %d 轮，进入 trial（试复活）", nid, self.observe_rounds)
            elif state == "trial":
                if not low:
                    # 试复活成功：分数恢复 → 真正复活
                    self._states[nid] = "active"
                    self._failure_counts[nid] = 0
                    self._isolate_since.pop(nid, None)
                    logger.info("神经元 %s 试复活成功（生存分 %.2f 恢复）", nid, score)
                else:
                    # 试复活失败 → 执行凋亡（不可逆）
                    self._states[nid] = "dead"
                    self._apoptosed[nid] = True
                    logger.warning("神经元 %s 试复活失败（生存分 %.2f），执行凋亡", nid, score)

        return dict(self._states)

    # ── 状态查询 ───────────────────────────────────────

    def get_state(self, neuron_id: str) -> str:
        return self._states.get(neuron_id, "active")

    def get_states(self) -> dict:
        return dict(self._states)

    def get_scores(self) -> dict:
        return dict(self._scores)

    def get_isolated(self) -> list:
        return [nid for nid, s in self._states.items() if s == "isolated"]

    def get_trial(self) -> list:
        """试复活中的神经元（隔离观察期满，需重新加入路由做最后确认）。"""
        return [nid for nid, s in self._states.items() if s == "trial"]

    def get_dead(self) -> list:
        return [nid for nid, s in self._states.items() if s == "dead"]

    def is_apoptosed(self, neuron_id: str) -> bool:
        return self._apoptosed.get(neuron_id, False) or self._states.get(neuron_id) == "dead"

    def get_apoptosis_candidates(self) -> list:
        """已凋亡（dead）的神经元 ID（兼容旧调用）。"""
        return self.get_dead()

    def revive(self, neuron_id: str) -> bool:
        """复活（隔离/试复活期恢复，或手动干预）。"""
        if self._states.get(neuron_id) in ("isolated", "candidate", "trial"):
            self._states[neuron_id] = "active"
            self._failure_counts.pop(neuron_id, None)
            self._isolate_since.pop(neuron_id, None)
            self._apoptosed.pop(neuron_id, None)
            logger.info("神经元 %s 手动复活", neuron_id)
            return True
        return False

    # ── 资源清理（dead 后执行）──────────────────────────

    def cleanup_neuron(
        self,
        neuron_id: str,
        ckpt_path: Optional[str] = None,
        ensemble: Optional[Any] = None,
    ) -> bool:
        """清理凋亡神经元的资源。

        Args:
            neuron_id: 神经元 ID
            ckpt_path: ckpt 文件路径，若提供则移入回收站目录（不直接删除，防误判丢失）
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

        # 移动 ckpt 到回收站目录（人脑：凋亡清除不是销毁信息，而是移出工作集）
        if ckpt_path is not None and os.path.exists(ckpt_path):
            try:
                recycle_dir = os.path.join(os.path.dirname(ckpt_path), "_recycle_bin")
                os.makedirs(recycle_dir, exist_ok=True)
                dst = os.path.join(recycle_dir, os.path.basename(ckpt_path))
                os.replace(ckpt_path, dst)
                logger.info("凋亡神经元 ckpt 移入回收站: %s", dst)
            except OSError as e:
                logger.error("移动 ckpt %s 失败: %s", ckpt_path, e)
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

    # ── 突触修剪（层级 0：先修剪连接，不动神经元本体）────

    def prune_synapses(self, neurons: Dict[str, Any],
                       min_usage: float = 0.01, stale_rounds: int = 10) -> int:
        """修剪弱突触（side_channels）——人脑突触修剪（Synaptic Pruning）。

        长期未被利用的侧通道（|proj*scale+bias| 均值低）被删除，
        神经元本体保留（用进废退，最温和的弱化层级）。

        Args:
            neurons: {nid: ResonanceNeuron}
            min_usage: 通道 usage 低于此值视为弱（neuron._channel_usage 统计）
            stale_rounds: 连续多少轮低利用才修剪（防单轮波动）

        Returns:
            修剪的通道数
        """
        pruned = 0
        for nid, neuron in neurons.items():
            usage = getattr(neuron, "_channel_usage", {}) or {}
            for key, u in usage.items():
                if u < min_usage:
                    ch_type, peer_id = key.split(":", 1)
                    ch_name = f"{ch_type}_channels"
                    channels = getattr(neuron, ch_name, None)
                    if channels and peer_id in channels:
                        del channels[peer_id]
                        # 清理关联 scale/bias 参数
                        for attr in (f"{ch_type}_scale_{peer_id}", f"{ch_type}_bias_{peer_id}"):
                            if hasattr(neuron, attr):
                                delattr(neuron, attr)
                        pruned += 1
                        logger.info("突触修剪: %s → %s 通道（usage=%.4f）", nid, peer_id, u)
        return pruned

    def reset(self, neuron_id: str) -> None:
        """重置某神经元的失败计数和评估计数（不复活已凋亡的）。"""
        self._failure_counts.pop(neuron_id, None)
        self._eval_counts.pop(neuron_id, None)

    # ── 兼容旧接口（旧调用方）──────────────────────────

    def record_ppl(self, neuron_id: str, ppl: float) -> bool:
        """【兼容】记录 PPL。v2 以种群评分判定，此处仅保留计数供诊断。

        保留旧固定阈值触发（仅当调用方未走 step_population 时兜底）。
        """
        if self._apoptosed.get(neuron_id, False):
            return True
        self._eval_counts[neuron_id] = self._eval_counts.get(neuron_id, 0) + 1
        if self._eval_counts[neuron_id] <= self.grace_evals:
            return False
        # 兼容兜底：只有尚未进入 v2 状态机的旧调用路径才用固定阈值
        if self._states.get(neuron_id, "active") == "active":
            if ppl > self.ppl_threshold:
                self._failure_counts[neuron_id] = self._failure_counts.get(neuron_id, 0) + 1
                if self._failure_counts[neuron_id] >= self.failure_threshold_legacy:
                    self._states[neuron_id] = "candidate"
                    self._apoptosed[neuron_id] = True
                    logger.warning(
                        "神经元 %s 连续 %d 次 PPL > %.1f（当前 %.1f），标记凋亡（兼容路径）",
                        neuron_id, self._failure_counts[neuron_id], self.ppl_threshold, ppl)
                    return True
            else:
                self._failure_counts[neuron_id] = 0
        return False

    def check_activation(self, neuron_id: str, activation_count: int, total_rounds: int) -> bool:
        """【兼容】检查激活率。v2 请用 step_population 注入 activity 信号。"""
        if self._apoptosed.get(neuron_id, False):
            return True
        if total_rounds < self.min_rounds_observed:
            return False
        ratio = activation_count / total_rounds if total_rounds > 0 else 0
        if ratio < self.activation_ratio and self._states.get(neuron_id, "active") == "active":
            self._apoptosed[neuron_id] = True
            self._states[neuron_id] = "candidate"
            logger.warning("神经元 %s 激活率 %.3f < %.3f（%d/%d 轮），标记凋亡（兼容路径）",
                           neuron_id, ratio, self.activation_ratio,
                           activation_count, total_rounds)
            return True
        return False


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
    1. 从经验和群体上下文初始化新神经元 checkpoint
    2. 初始化为"幼稚态"
    3. 加入 ensemble
    """

    # domain -> 连续高错误率计数
    _domain_error_counts: dict = field(default_factory=dict)
    # 错误率 = 1 - accuracy；> 0.5 意味着预测错误多于正确，明显能力不足
    error_rate_threshold: float = 0.5
    error_count_for_trigger: int = 8  # 需要连续 8 次高错误率才触发（避免过快扩张）

    # 错误率斜率判别器：区分"数据不足" vs "容量不足"
    # domain -> 最近错误率序列
    _domain_error_history: dict = field(default_factory=dict)
    slope_window: int = 5  # 斜率计算窗口（最近 N 次评估）
    # 斜率 ≥ 此值视为"平台/上升"（容量不足）；< 此值视为"仍在学习"（数据不足）
    # 负值越接近 0 表示下降越慢；正值表示错误率在上升（退化）
    plateau_slope_threshold: float = -0.02

    @staticmethod
    def _compute_slope(values: list) -> float:
        """对最近 N 个点做简单线性回归，返回斜率。

        斜率 < 0：错误率持续下降（神经元还在学习 → 数据不足）
        斜率 ≈ 0：平台期（神经元学到上限 → 容量不足）
        斜率 > 0：错误率上升（退化 → 容量不足 / 过拟合）
        """
        n = len(values)
        if n < 2:
            return 0.0
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(values) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
        den = sum((x - mean_x) ** 2 for x in xs)
        if den == 0:
            return 0.0
        return num / den

    def diagnose_domain(self, domain: str) -> str:
        """诊断某 domain 当前处于哪种状态（不触发任何动作）。

        Returns:
            "healthy"           —— 错误率低，无需进化
            "data_insufficient" —— 错误率高但仍在下降（斜率<阈值），应喂数据
            "capacity_limited"  —— 错误率高且平台/上升（斜率≥阈值），应加神经元
            "unknown"           —— 历史数据不足，无法判定
        """
        history = self._domain_error_history.get(domain, [])
        if not history:
            return "unknown"
        latest = history[-1]
        if latest <= self.error_rate_threshold:
            return "healthy"
        if len(history) < self.slope_window:
            return "unknown"
        slope = self._compute_slope(history[-self.slope_window:])
        if slope < self.plateau_slope_threshold:
            return "data_insufficient"
        return "capacity_limited"

    def record_domain_error(self, domain: str, error_rate: float) -> bool:
        """记录某 domain 的错误率，返回是否触发新生。

        斜率判别（2026-08-01 落地）：
        - 错误率高 + 斜率<0（持续下降）→ 神经元还在学习 → 数据不足，不触发新生
        - 错误率高 + 斜率≈0 或 >0（平台/上升）→ 容量上限 → 触发 neurogenesis
        - 历史不足 slope_window 次时，沿用原计数逻辑（保守触发）

        Args:
            domain: 域名
            error_rate: 错误率 [0, 1]

        Returns:
            True 如果该 domain 需要新生神经元
        """
        # 1. 记录错误率历史
        history = self._domain_error_history.setdefault(domain, [])
        history.append(error_rate)
        if len(history) > self.slope_window * 2:
            history.pop(0)

        # 2. 错误率低 → 重置计数，不触发
        if error_rate <= self.error_rate_threshold:
            self._domain_error_counts[domain] = 0
            return False

        # 3. 高错误率：累计计数
        self._domain_error_counts[domain] = self._domain_error_counts.get(domain, 0) + 1

        # 4. 斜率判别：历史足够长才启用
        if len(history) >= self.slope_window:
            slope = self._compute_slope(history[-self.slope_window:])
            if slope < self.plateau_slope_threshold:
                # 还在学习，数据不足——不触发新生，也不重置计数
                logger.info(
                    "domain %s 错误率高但持续下降（斜率 %.4f < %.2f），"
                    "判定为数据不足，继续喂数据而非加神经元",
                    domain, slope, self.plateau_slope_threshold,
                )
                return False

        # 5. 平台/上升 或 历史不足：达到计数阈值则触发新生
        if self._domain_error_counts[domain] >= self.error_count_for_trigger:
            slope_str = "N/A"
            if len(history) >= self.slope_window:
                slope_str = f"{self._compute_slope(history[-self.slope_window:]):.4f}"
            logger.warning(
                "domain %s 连续 %d 次错误率 > %.1f（当前 %.3f，斜率 %s），"
                "判定为容量不足，触发新生",
                domain,
                self._domain_error_counts[domain],
                self.error_rate_threshold,
                error_rate,
                slope_str,
            )
            # 重置计数，避免重复触发
            self._domain_error_counts[domain] = 0
            return True

        return False

    # ── SpecSelector：规格选择器（2026-08-01 落地）──
    # 触发 neurogenesis 后，根据任务难度选择新建神经元的规格
    # 生物学对应：海马新生神经元虽大小相似，但会根据环境信号分化为不同亚型
    #
    # 判别信号：
    #   斜率 → 决定"是否需要新神经元"（数据不足 vs 容量不足）
    #   错误率绝对值 → 决定"需要多大的神经元"（任务难度）
    #
    # 阈值设计（可调）：
    #   错误率 < 0.3  → 简单任务 → compact（36M，成本低）
    #   0.3 ≤ 错误率 < 0.6 → 中等任务 → standard（116M，中等容量）
    #   错误率 ≥ 0.6 → 复杂任务 → expert（285M，最大容量）
    simple_task_threshold: float = 0.3   # < 此值 → compact
    complex_task_threshold: float = 0.6  # ≥ 此值 → expert
    # 中间区间 → standard

    def select_spec(self, domain: str) -> str:
        """根据 domain 当前错误率水平选择新神经元的规格。

        必须在 record_domain_error 之后调用，基于该 domain 的历史错误率判定。
        如果该 domain 错误率低（healthy）或历史不足，不应当调用此方法
        （调用方应先用 diagnose_domain 确认 capacity_limited）。

        Returns:
            "compact" / "standard" / "expert"
        """
        history = self._domain_error_history.get(domain, [])
        if not history:
            # 无历史数据，保守用 compact（最小成本试错）
            logger.info(
                "spec 选择: domain %s 无历史数据，默认 compact（最小成本试错）",
                domain,
            )
            return "compact"

        latest_error = history[-1]
        if latest_error < self.simple_task_threshold:
            spec = "compact"
            reason = f"错误率 {latest_error:.3f} < {self.simple_task_threshold}（简单任务）"
        elif latest_error < self.complex_task_threshold:
            spec = "standard"
            reason = (f"错误率 {latest_error:.3f} ∈ "
                      f"[{self.simple_task_threshold}, {self.complex_task_threshold})（中等任务）")
        else:
            spec = "expert"
            reason = f"错误率 {latest_error:.3f} ≥ {self.complex_task_threshold}（复杂任务）"

        logger.info("spec 选择: domain %s → %s，原因: %s", domain, spec, reason)
        return spec

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
        metrics_map: dict,
        ensemble: Any,
        ckpt_dir: Optional[str] = None,
        step_round: int = 0,
        prune_neurons: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """执行一次生命周期步进（v2：多维生存评分 + 分层状态机）。

        Args:
            metrics_map: {neuron_id: metrics}（activity/ppl/contribution/connectivity/
                         redundancy/maturity_ratio/is_inhibitory，缺失信号自动降权）
            ensemble: ResonanceEnsemble
            ckpt_dir: ckpt 目录路径
            step_round: 当前轮次（隔离观察计时）
            prune_neurons: 提供 {nid: neuron} 时执行突触修剪（层级 0）

        Returns:
            dict with:
            - states: {nid: state}（active/candidate/isolated/dead）
            - isolated: 进入隔离的神经元列表（调用方应摘除路由，保留权重）
            - dead: 执行凋亡的神经元列表（调用方应清理 + 新生补偿）
            - pruned_synapses: 修剪的突触数
        """
        # 层级 0：突触修剪（弱连接先消失，神经元本体保留）
        pruned = 0
        if prune_neurons:
            pruned = self.apoptosis.prune_synapses(prune_neurons)

        # 状态机流转
        states = self.apoptosis.step_population(metrics_map, step_round)
        newly_isolated = [
            nid for nid, s in states.items()
            if s == "isolated" and self.apoptosis._isolate_since.get(nid, 0) == step_round
        ]
        # 隔离观察期满 → 试复活（sleep 侧需 revive_neuron 重新加入路由）
        newly_trial = [
            nid for nid, s in states.items() if s == "trial"
        ]
        dead = [
            nid for nid, s in states.items() if s == "dead"
        ]
        # dead 神经元清理（ckpt 移入回收站 + 从 ensemble 摘除）
        for nid in dead:
            ckpt_path = (
                os.path.join(ckpt_dir, f"neuron_{nid}.pt")
                if ckpt_dir else None
            )
            self.apoptosis.cleanup_neuron(nid, ckpt_path, ensemble)

        # 递增所有注册神经元的成熟度
        self.maturity.tick_all()

        return {
            "states": states,
            "isolated": newly_isolated,
            "trial": newly_trial,
            "dead": dead,
            "pruned_synapses": pruned,
        }
