"""Coactivation tracking for dynamic tribal grouping (P1-1).

双矩阵设计（人脑启发：快/慢突触可塑性）：
- _fast_matrix: 即时共激活计数（每次 update 累加），用于短期部落分组
- _slow_matrix: EMA 衰减的慢速矩阵（长期统计），供 detect_isolated_patterns 查询

调用时机：ensemble.forward() 中每个 round 结束后调用 update(active_ids)。
当多个神经元在同一 round 中共同激活，它们的 pair 计数增加。

孤立模式检测：如果某神经元的共激活 pair 中 >80% 都低于阈值，
说明它"孤立"，可能需要新生一个相关神经元来协同。
"""

import logging
from collections import defaultdict
from typing import Iterable, Tuple

logger = logging.getLogger("CoactivationTracker")


class CoactivationTracker:
    """Track coactivation patterns between neurons to form dynamic tribes.

    双矩阵：
    - _fast_matrix: 即时计数（每次 update +1），短期统计
    - _slow_matrix: EMA 衰减（alpha=0.05），长期统计，供孤立检测

    Attributes:
        forget_threshold: 低频判定阈值（pair 频次低于此值视为低频）
            detect_isolated_patterns 用 forget_threshold * 10 作为低频 cutoff
    """

    def __init__(self, ema_alpha: float = 0.05, forget_threshold: float = 0.01):
        # fast: (nid_i, nid_j) -> 即时计数
        self._fast_matrix: dict = defaultdict(float)
        # slow: (nid_i, nid_j) -> EMA 衰减值
        self._slow_matrix: dict = defaultdict(float)
        self.ema_alpha = ema_alpha
        self.forget_threshold = forget_threshold
        # 每个神经元参与过的总激活次数（用于归一化）
        self._activation_counts: dict = defaultdict(int)

        logger.info(
            f"CoactivationTracker initialized (ema_alpha={ema_alpha}, "
            f"forget_threshold={forget_threshold})"
        )

    def update(self, ids: Iterable[str], round_num: int = 1) -> None:
        """记录一次共激活事件。

        当多个神经元在同一 round 中共同激活时，所有 pair 的计数增加。
        slow 矩阵用 EMA 更新：slow = (1-alpha)*slow + alpha*fast_increment。

        Args:
            ids: 本 round 中激活的神经元 ID 列表
            round_num: round 编号（当前未使用，保留供未来扩展）
        """
        active_list = list(ids)
        if len(active_list) < 2:
            # 单个神经元激活，只记录 activation_count
            for nid in active_list:
                self._activation_counts[nid] += 1
            return

        # 记录所有 pair 的共激活
        for i in range(len(active_list)):
            for j in range(i + 1, len(active_list)):
                pair = tuple(sorted([active_list[i], active_list[j]]))
                self._fast_matrix[pair] += 1.0
                # EMA 更新 slow 矩阵
                self._slow_matrix[pair] = (
                    (1 - self.ema_alpha) * self._slow_matrix[pair]
                    + self.ema_alpha * 1.0
                )

        for nid in active_list:
            self._activation_counts[nid] += 1

    def get_coactivation(self, nid_i: str, nid_j: str) -> float:
        """获取两个神经元的共激活强度（slow 矩阵值）。"""
        pair = tuple(sorted([nid_i, nid_j]))
        return self._slow_matrix.get(pair, 0.0)

    def get_tribe(self, nid: str, min_strength: float = 0.1) -> list:
        """获取某神经元的部落成员（共激活强度 > min_strength 的神经元）。"""
        tribe = []
        for (i, j), strength in self._slow_matrix.items():
            if i == nid and strength > min_strength:
                tribe.append(j)
            elif j == nid and strength > min_strength:
                tribe.append(i)
        return tribe

    def get_all_tribes(self, min_strength: float = 0.1) -> dict:
        """获取所有部落分组（nid -> tribe_members）。"""
        tribes: dict = defaultdict(list)
        for (i, j), strength in self._slow_matrix.items():
            if strength > min_strength:
                tribes[i].append(j)
                tribes[j].append(i)
        return dict(tribes)

    def decay(self) -> None:
        """对 slow 矩阵进行一次衰减（可选，用于睡眠时遗忘）。

        slow = slow * (1 - ema_alpha)
        """
        for pair in list(self._slow_matrix.keys()):
            self._slow_matrix[pair] *= (1 - self.ema_alpha)

    def get_stats(self) -> dict:
        """获取统计信息。"""
        return {
            "total_pairs": len(self._slow_matrix),
            "total_activations": sum(self._activation_counts.values()),
            "fast_matrix_size": len(self._fast_matrix),
            "neurons_tracked": len(self._activation_counts),
        }
