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
    acetylcholine: float = 0.5   # C25-C：新颖性/注意聚焦（对比文档 171 行已列未实现）

    # 目标值（由外部信号设定，实际值缓慢趋近）
    _target_dopamine: float = 0.5
    _target_serotonin: float = 0.5
    _target_norepinephrine: float = 0.5
    _target_acetylcholine: float = 0.5

    # EMA 趋近速率
    ema_alpha: float = 0.1

    def set_targets(
        self,
        dopamine: Optional[float] = None,
        serotonin: Optional[float] = None,
        norepinephrine: Optional[float] = None,
        acetylcholine: Optional[float] = None,
    ) -> None:
        """设置目标调质水平（由外部信号驱动）。"""
        if dopamine is not None:
            self._target_dopamine = max(0.0, min(1.0, dopamine))
        if serotonin is not None:
            self._target_serotonin = max(0.0, min(1.0, serotonin))
        if norepinephrine is not None:
            self._target_norepinephrine = max(0.0, min(1.0, norepinephrine))
        if acetylcholine is not None:
            self._target_acetylcholine = max(0.0, min(1.0, acetylcholine))

    def step(self) -> None:
        """EMA 趋近目标值（调质不会突变，而是缓慢调整）。"""
        self.dopamine += self.ema_alpha * (self._target_dopamine - self.dopamine)
        self.serotonin += self.ema_alpha * (self._target_serotonin - self.serotonin)
        self.norepinephrine += self.ema_alpha * (self._target_norepinephrine - self.norepinephrine)
        self.acetylcholine += self.ema_alpha * (self._target_acetylcholine - self.acetylcholine)

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

    def get_attention_temp_gain(self) -> float:
        """S9: 获取注意力温度增益（norepinephrine 驱动）。

        作用于 Transformer 内部 attention 的 query 缩放：
        - 高 NE → temp_gain > 1 → logits 放大 → softmax 更尖锐（高警觉，聚焦）
        - 低 NE → temp_gain < 1 → logits 缩小 → softmax 更分散（低警觉，泛化）
        - NE = 0.5 → temp_gain = 1.0（中性，标准注意力）

        与 get_field_write_scale 使用相同映射，保持调质语义一致。
        """
        return 0.5 + self.norepinephrine * 1.0

    def get_attention_focus_gain(self) -> float:
        """C25-C：获取注意聚焦增益（乙酰胆碱驱动，与 NE 警觉互补）。

        人脑：乙酰胆碱（ACh）在注意新刺激时升高（新颖性→注意聚焦），
        习惯化时降低。作用于 attention 温度（与 NE 组合调制）：
        - 高 ACh → focus_gain > 1 → logits 放大 → softmax 更尖锐（聚焦新输入）
        - 低 ACh → focus_gain < 1 → logits 缩小 → softmax 更分散（习惯化）
        - ACh = 0.5 → focus_gain = 1.0（中性）

        映射范围 [0.6, 1.4]（比 NE 的 [0.5, 2.0] 温和——ACh 是精细调节，
        不覆盖警觉的主调制；0.5=中性 → 1.0，与 DA/NE/5-HT 约定一致）。
        ensemble forward 中与 NE temp_gain 相乘组合。
        """
        return 0.6 + self.acetylcholine * 0.8

    def get_ffn_gain(self) -> float:
        """S9: 获取 FFN 增益（dopamine 驱动）。

        作用于 Transformer 内部 SwiGLU FFN 的输出缩放：
        - 高 DA → ffn_gain > 1 → FFN 输出增强（奖励信号，强化重要特征通过）
        - 低 DA → ffn_gain < 1 → FFN 输出衰减（惩罚信号，弱化非重要特征）
        - DA = 0.5 → ffn_gain = 1.0（中性，标准 FFN）

        采用以 1.0 为中性的 [0.5, 1.5] 映射；训练/推理均以标准 FFN
        （ffn_gain=1.0）作为默认基线。
        """
        return 0.5 + self.dopamine

    def should_trigger_neurogenesis(self) -> bool:
        """是否应该触发神经元新生（低多巴胺持续）。"""
        return self.dopamine < 0.2

    def get_state_dict(self) -> dict:
        """获取状态字典（用于持久化）。"""
        return {
            "dopamine": self.dopamine,
            "serotonin": self.serotonin,
            "norepinephrine": self.norepinephrine,
            "acetylcholine": self.acetylcholine,
            "_target_dopamine": self._target_dopamine,
            "_target_serotonin": self._target_serotonin,
            "_target_norepinephrine": self._target_norepinephrine,
            "_target_acetylcholine": self._target_acetylcholine,
        }

    def load_state_dict(self, state: dict) -> None:
        """加载状态。"""
        self.dopamine = state.get("dopamine", 0.5)
        self.serotonin = state.get("serotonin", 0.5)
        self.norepinephrine = state.get("norepinephrine", 0.5)
        self.acetylcholine = state.get("acetylcholine", 0.5)  # 旧 ckpt 无 → 默认中性
        self._target_dopamine = state.get("_target_dopamine", 0.5)
        self._target_serotonin = state.get("_target_serotonin", 0.5)
        self._target_norepinephrine = state.get("_target_norepinephrine", 0.5)
        self._target_acetylcholine = state.get("_target_acetylcholine", 0.5)


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
        downscale_factor: float = 0.98,
    ):
        self.replay_buffer_size = replay_buffer_size
        self.consolidation_interval = consolidation_interval
        # 突触稳态下调（C25-D，人脑 NREM 慢波 downscaling）：睡眠期整体按比例
        # 缩小 side_channels 权重，突出强信号（对比文档 2.6 关键差异修复）
        self.downscale_factor = downscale_factor

        # 重放缓冲区：存储 high-resonance 场状态
        self._replay_buffer: deque = deque(maxlen=replay_buffer_size)

        # 上次巩固的步数
        self._last_consolidation_step: int = 0

    def record_high_resonance_state(
        self,
        field_state: torch.Tensor,
        resonance_score: float,
        step: int,
        active_nids: Optional[list] = None,
        threshold: float = 0.5,
        text: Optional[str] = None,
    ) -> None:
        """记录一次高共振场状态（用于后续重放）。

        C25-D：新增 active_nids（参与该共振的 neuron 集）——重放时据此
        "再激活"共激活统计，驱动突触巩固（人脑海马回放 → 皮层再激活），
        而非纯统计占位。

        C26 增量六（Phase 1.7 真正睡眠重放）：新增 text（触发本次共振的
        输入文本）——重放时作为"场状态条件化 forward"的语言目标，让神经元
        在记忆注意窗（该场状态作 field_state）下重放生成该文本，把"条件化
        读取"固化为可学习权重（读路径 field_read_layers + LoRA 双训）。

        Args:
            field_state: 场状态向量
            resonance_score: 本次共振的最高分数
            step: 当前步数
            active_nids: 参与本次共振的 neuron ID 列表（供重放再激活）
            threshold: 共振分数阈值，高于此值才记录
            text: 触发文本（可选；None = 旧记录无语言目标，仅共激活重放）
        """
        if resonance_score > threshold:
            self._replay_buffer.append({
                "state": field_state.detach().clone(),
                "score": resonance_score,
                "step": step,
                "active_nids": list(active_nids) if active_nids else None,
                "text": text,
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
        stdp_tracker: Optional[Any] = None,
    ) -> dict:
        """执行一次睡眠巩固。

        Args:
            neurons: {neuron_id: ResonanceNeuron}
            coactivation_tracker: CoactivationTracker 实例
            current_step: 当前步数
            stdp_tracker: STDPTracker 实例（C25-B：结构演化——共激活统计
                累积 + 通道修剪/生长，突触可塑性从权重缩放升级为结构演化）

        Returns:
            巩固统计
        """
        logger.info("开始睡眠巩固（step=%d，重放缓冲=%d）",
                    current_step, len(self._replay_buffer))

        stats = {
            "replayed_states": 0,
            "channels_reinforced": 0,
            "channels_pruned": 0,
            "channels_downscaled": 0,
            "channels_struct_pruned": 0,
            "channels_grown": 0,
            "fingerprints_updated": 0,
            "pairs_forgotten": 0,
        }

        # 1. 重放高共振场状态（C25-D 真重放）：重放时用记录的 active_nids
        #    再激活共激活统计（人脑海马回放 → 皮层再激活）——重放真正驱动
        #    突触巩固，而非纯统计占位。无 active_nids 的旧记录仅计数。
        for record in list(self._replay_buffer):
            stats["replayed_states"] += 1
            an = record.get("active_nids")
            if an and len(an) >= 2 and coactivation_tracker is not None:
                try:
                    coactivation_tracker.update(an)
                except Exception as e:
                    logger.warning("重放共激活更新失败（非关键）: %s", e)

        # 2. 强化 slow EMA 高的 side_channels（重放已更新共激活统计，
        #    此处强化的 strong_pairs 已包含重放贡献）
        if coactivation_tracker is not None and hasattr(coactivation_tracker, "get_strong_pairs"):
            strong_pairs = coactivation_tracker.get_strong_pairs(threshold=0.2)
            # 双向强化（2026-08-10 接口修复）：get_strong_pairs 返回 sorted pair
            # （无方向），原实现按 (pre, post) 单向解包依赖 ID 字典序的隐式约定，
            # 且"所有含 post_key 的 neuron"过宽（误强化无关 neuron 通道）。
            # 修正：共激活 pair (i,j) → 只强化 i→j 与 j→i 两条通道（精确双向）。
            for i, j in strong_pairs:
                for src, dst in ((i, j), (j, i)):
                    src_neuron = neurons.get(src)
                    if (src_neuron is not None and hasattr(src_neuron, "excite_channels")
                            and dst in src_neuron.excite_channels):
                        src_neuron.excite_channels[dst].weight.data *= 1.1
                        stats["channels_reinforced"] += 1

        # 2.5 突触稳态下调（C25-D，人脑 NREM 慢波 downscaling）：
        # 全局整体按比例缩小 side_channels 权重——强信号净保留（强化×1.1 后
        # 再 ×0.98 ≈ ×1.08），弱信号被进一步压低（×0.98），突出强连接。
        # 与"修剪弱通道"互补：downscaling 是连续调节，修剪是离散清除。
        downscaled = self._synaptic_downscaling(neurons, factor=self.downscale_factor)
        stats["channels_downscaled"] = downscaled

        # 2.6 突触结构演化（C25-B，人脑突触规模调节）：STDP 共激活统计累积
        # + 通道条目修剪/生长——连接层"突触可塑性"从权重缩放（STDP/
        # downscaling）升级为结构演化（长期低共激活弱通道清除 + 高共激活
        # 缺失通道生长，邻居相似初始化）。离线路径，不碰 forward_train 监督。
        if stdp_tracker is not None:
            try:
                stdp_tracker.accumulate_coactivation()
                struct_stats = stdp_tracker.apply_structure_updates(neurons)
                stats["channels_struct_pruned"] = struct_stats.get("pruned", 0)
                stats["channels_grown"] = struct_stats.get("grown", 0)
                if struct_stats.get("pruned") or struct_stats.get("grown"):
                    logger.info("  STDP 结构演化: 修剪 %d 通道, 生长 %d 通道",
                                struct_stats.get("pruned", 0), struct_stats.get("grown", 0))
            except Exception as e:
                logger.warning("STDP 结构演化失败（非关键）: %s", e)

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

    @torch.no_grad()
    def _synaptic_downscaling(self, neurons: dict, factor: float = 0.98) -> int:
        """突触稳态下调（人脑 NREM 慢波 downscaling）。

        睡眠期整体按比例缩小 side_channels 权重（excite + inhibit），
        突出强信号、压低弱噪声——对比文档 2.6 "态极 sleep 是'拿累积样本
        离线训练'，重放/下调是方向性借鉴，未实现生物意义上的'逐条回放 +
        全局缩放'"的"全局缩放"部分。

        Args:
            neurons: {neuron_id: ResonanceNeuron}
            factor: 缩放因子（<1 缩小；强通道因先前强化×1.1 净保留）

        Returns:
            被缩放的通道数
        """
        n = 0
        for neuron in neurons.values():
            for ch_dict in (getattr(neuron, "excite_channels", {}),
                            getattr(neuron, "inhibit_channels", {})):
                for linear in ch_dict.values():
                    if hasattr(linear, "weight"):
                        linear.weight.data *= factor
                        n += 1
        if n:
            logger.info("  突触稳态下调: %d 个通道 ×%.3f", n, factor)
        return n

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

    def get_state_dict(self) -> dict:
        """序列化为可持久化的 dict。

        replay_buffer 中的 field_state tensor 会被 torch.save 序列化，
        保留高共振经验供跨会话睡眠重放。
        """
        return {
            "replay_buffer": list(self._replay_buffer),
            "last_consolidation_step": self._last_consolidation_step,
            "replay_buffer_size": self.replay_buffer_size,
            "consolidation_interval": self.consolidation_interval,
            "downscale_factor": self.downscale_factor,
        }

    def load_state_dict(self, state: dict) -> None:
        """从 dict 恢复状态。"""
        self.replay_buffer_size = state.get("replay_buffer_size", self.replay_buffer_size)
        self.consolidation_interval = state.get("consolidation_interval", self.consolidation_interval)
        self.downscale_factor = state.get("downscale_factor", self.downscale_factor)
        self._replay_buffer = deque(
            state.get("replay_buffer", []),
            maxlen=self.replay_buffer_size,
        )
        self._last_consolidation_step = state.get("last_consolidation_step", 0)
        logger.info(
            f"SleepConsolidator restored: {len(self._replay_buffer)} replay states, "
            f"last_step={self._last_consolidation_step}"
        )
