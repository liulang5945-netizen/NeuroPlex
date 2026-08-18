"""Resonance field core - shared communication medium for all neurons.

The field is a D-dimensional vector space where neurons write
L2-normalised vectors and read the accumulated state.  It is the
"neural language" of the architecture - completely independent of
the tokenizer (Level 0) and the per-neuron concept spaces (Level 1).

Key properties:
- All writes are L2-normalised -> neuron size does not determine "loudness"
- Cosine similarity between a neuron's write and the field state is the
  scoring mechanism (resonance = alignment with the collective)
- W_cond is now ACTIVE: it projects the field state to a "conditioned"
  subspace before scoring, so it learns which cross-neuron patterns matter.

P0#3 — Divisive inhibition:
- Excitatory neurons accumulate additively: state += v_e
- Inhibitory neurons apply multiplicative decay: mask *= (1 - w * |v_i|)
- Effective field = state ⊙ mask (Hadamard product)
- GABA-like: inhibition divides/shunts, not subtracts

Fixes (this version):
  H2: state is [B, D], one independent field per sample (no cross-sample bleed)
  H5: score() uses leave-one-out (excludes the neuron's own contribution)
  H6: prediction_complementarity_score(): how much another neuron *corrects*
      this one's mistakes, measured on logits (not orthogonal geometry).
  H8: W_cond is now applied: conditioned = sigmoid(state @ W_cond) * state
      before scoring - the dead parameter becomes a learned gate on the field.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional

import torch
import torch.nn as nn


class ResonanceField(nn.Module):
    """Shared resonance field - the "neural language" of the architecture.

    State shape is now [B, D] (H2 fix): each sample in the batch gets its own
    independent field, so there is no cross-sample contamination.

    W_cond (H8 fix): a learnable [D, D] parameter applied as a multiplicative
    gate on the field state before scoring.  This was previously a dead
    parameter; it is now used in score() and complementarity().
    """

    # 每个神经元的写入历史最多保留这么多条（诊断用）
    # 上千神经元 × 多轮共振时，无界 list 会导致内存爆炸
    HISTORY_MAXLEN: int = 4

    def __init__(self, dim: int = 4096, device: Optional[torch.device] = None):
        super().__init__()
        self.dim = dim
        self._device = device or torch.device("cpu")

        self.register_buffer("state", torch.zeros(dim))
        # P0#3: inhibitory mask — accumulative multiplicative decay
        # Starts as all-ones (no inhibition). Updated by write_inhibit().
        self.register_buffer("inhibitory_mask", torch.ones(dim))
        # H2: per-sample field state, set lazily on first write with batch dim
        self._batch_size: int = 1

        self.W_cond = nn.Parameter(torch.randn(dim, dim) * 0.02)

        # deque(maxlen=...) 自动丢弃最老条目，防止 N×R 轮后内存爆炸
        self._write_history: Dict[str, Deque[torch.Tensor]] = {}
        self._contributions: Dict[str, torch.Tensor] = {}
        # BioOSS: 抑制性神经元贡献追踪（用于 leave-one-out 撤销）
        self._inhibit_contributions: Dict[str, torch.Tensor] = {}
        self.scores: Dict[str, float] = {}
        self.n_active: int = 0

    def reset(self, batch_size: int = 1) -> None:
        # Promote state to [B, D] up front for batch_size > 1 (H2): each sample
        # gets an independent field, so there is never cross-sample bleed.
        # R15（REMEDIATION_PLAN 2026-08-14）：零/一张量创建在字段所在设备
        # （W_cond 锚定设备），避免 reset 后 state/mask 悄悄回到 CPU。
        dev = self.W_cond.device
        if batch_size > 1:
            self.state = torch.zeros(batch_size, self.dim, device=dev)
            self.inhibitory_mask = torch.ones(batch_size, self.dim, device=dev)
        else:
            self.state = torch.zeros(self.dim, device=dev)
            self.inhibitory_mask = torch.ones(self.dim, device=dev)
        self._batch_size = batch_size
        self._contributions.clear()
        self._inhibit_contributions.clear()
        self.scores.clear()
        self._write_history.clear()
        self.n_active = 0

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def write(self, neuron_id: str, vector: torch.Tensor,
              scale=1.0) -> torch.Tensor:
        """写入神经元的场向量（L2 归一化后累加到 state）。

        P1-2: scale 由 NeuromodulatorState.get_field_write_scale 提供，
        高去甲肾上腺素 → 写入强度↑（警觉状态），低 → 写入强度↓（放松）。
        scale 作用于归一化后的向量，保证方向不变只调幅度。

        C8: scale 支持 per-sample tensor [B]，让 neuron 自身置信度调制写入幅度。
        - float：所有 sample 相同 scale（向后兼容）
        - [B] tensor：每个 sample 独立 scale（per-sample confidence）
        """
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        v_norm = vector / (vector.norm(dim=-1, keepdim=True) + 1e-8)
        # C8: 支持 per-sample scale（[B] tensor）
        if isinstance(scale, torch.Tensor) and scale.dim() == 1:
            v_scaled = v_norm * scale.unsqueeze(-1)  # [B, D] * [B, 1]
        else:
            v_scaled = v_norm * scale  # float 或已广播的 tensor
        B = v_scaled.shape[0]
        if B == 1 and self._batch_size == 1:
            self.state = self.state + v_scaled.squeeze(0)
        elif self.state.dim() == 1 and self._batch_size == 1:
            # auto-promote single [D] field to per-sample [B, D] (H2): robust when
            # neurons write batched vectors without an explicit reset(batch_size=...).
            self.state = self.state.unsqueeze(0).expand(B, -1).clone()
            self.state = self.state + v_scaled
            self._batch_size = B
        elif B == self._batch_size:
            if self.state.dim() == 1:
                self.state = self.state.unsqueeze(0).expand(self._batch_size, -1).clone()
            self.state = self.state + v_scaled
        else:
            raise ValueError(f"vector batch {B} != field batch {self._batch_size}")
        # _contributions 存储 scaled 版本，保证 leave-one-out 减法正确
        self._contributions[neuron_id] = v_scaled.detach()
        self.n_active += 1
        if neuron_id not in self._write_history:
            self._write_history[neuron_id] = deque(maxlen=self.HISTORY_MAXLEN)
        # _write_history 存储单位向量（语义：方向历史，与调质状态解耦）
        self._write_history[neuron_id].append(v_norm.detach())
        return v_scaled

    def update(self, neuron_id: str, vector: torch.Tensor,
               scale=1.0) -> torch.Tensor:
        """增量更新：减去该 neuron 的旧贡献，加上新贡献。

        用于多轮共振场景（如 TribeSuperNeuron.forward_tribe）：
        每轮成员重新写入时，需要替换而非累加，否则 state 无界增长。

        与 write 的区别：
        - write: 纯累加（state += v），适合 round 1 初始写入
        - update: 替换（state = state - old + new），适合 round 2+ 更新

        P1-2: scale 同 write()，由 NeuromodulatorState.get_field_write_scale 提供。
        C8: scale 支持 per-sample tensor [B]（同 write）。
        """
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        v_norm = vector / (vector.norm(dim=-1, keepdim=True) + 1e-8)
        # C8: 支持 per-sample scale（[B] tensor）
        if isinstance(scale, torch.Tensor) and scale.dim() == 1:
            v_scaled = v_norm * scale.unsqueeze(-1)  # [B, D] * [B, 1]
        else:
            v_scaled = v_norm * scale  # float 或已广播的 tensor

        # 减去旧贡献（_contributions 已存 scaled 版本，减法一致）
        old_contrib = self._contributions.get(neuron_id)
        if old_contrib is not None:
            if self.state.dim() == 1 and old_contrib.dim() == 1:
                self.state = self.state - old_contrib
            elif self.state.dim() == old_contrib.dim():
                self.state = self.state - old_contrib
            else:
                # 维度不匹配时广播减法
                self.state = self.state - old_contrib.squeeze(0) if old_contrib.dim() > self.state.dim() else self.state - old_contrib

        # 加上新贡献（复用 write 的累加逻辑）
        B = v_scaled.shape[0]
        if B == 1 and self._batch_size == 1:
            self.state = self.state + v_scaled.squeeze(0)
        elif B == self._batch_size:
            if self.state.dim() == 1:
                self.state = self.state.unsqueeze(0).expand(self._batch_size, -1).clone()
            self.state = self.state + v_scaled
        else:
            raise ValueError(f"vector batch {B} != field batch {self._batch_size}")

        self._contributions[neuron_id] = v_scaled.detach()
        if neuron_id not in self._write_history:
            self._write_history[neuron_id] = deque(maxlen=self.HISTORY_MAXLEN)
        self._write_history[neuron_id].append(v_norm.detach())
        return v_scaled

    def write_inhibit(self, neuron_id: str, vector: torch.Tensor,
                      weight: float = 1.0) -> torch.Tensor:
        """P0#3: 抑制性神经元写入——乘法衰减掩码。

        GABA-like divisive inhibition: mask *= (1 - weight * |v|)
        而非旧的 v=-v 符号翻转（在 L2 归一化字段中被抹掉）。

        Args:
            neuron_id: neuron identifier.
            vector: [D] or [B, D] field vector (positive direction).
            weight: inhibition strength (0-1, default 1.0).
                    由 neuron 的置信度/共振分驱动，弱神经元抑制效果弱。

        Returns:
            updated inhibitory_mask.
        """
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        # Use absolute value: inhibitory neuron's "preferred direction" becomes
        # the dimensions it attenuates. L2-normalized to keep scale bounded.
        v_abs = vector.abs()
        v_abs = v_abs / (v_abs.norm(dim=-1, keepdim=True) + 1e-8)

        # Divisive: mask_d = mask_d * (1 - w * |v_d|)
        # Multiple inhibitors accumulate multiplicatively
        # Clamp to [0, 1] to prevent mask from going negative
        decay = 1.0 - weight * v_abs
        decay = decay.clamp(min=0.0, max=1.0)

        B = decay.shape[0]
        if self.inhibitory_mask.dim() == 1 and B == 1:
            self.inhibitory_mask = self.inhibitory_mask * decay.squeeze(0)
        elif self.inhibitory_mask.dim() == 1 and B > 1:
            self.inhibitory_mask = self.inhibitory_mask.unsqueeze(0).expand(B, -1).clone()
            self.inhibitory_mask = self.inhibitory_mask * decay
        else:
            self.inhibitory_mask = self.inhibitory_mask * decay

        # Track contributions for potential undo (leave-one-out)
        self._inhibit_contributions[neuron_id] = decay.detach()

        return self.inhibitory_mask

    def apply_inhibitory_wta(self, top_k: int = 1) -> int:
        """Deviance detection 融合：inhibitory neuron 竞争性抑制（WTA）。

        人脑启发：inhibitory interneuron 之间相互竞争（winner-take-all），
        只有最强的抑制方向生效，避免全场过度衰减。
        对应 PNAS 2026 competitive inhibition motif：
        多个 inhibitory ensemble 竞争，只有胜出者的抑制施加到场。

        机制：
        1. 从 _inhibit_contributions 收集所有 inhibitory neuron 的 decay
        2. 按 decay 的抑制强度（1 - mean(decay)）排序
        3. 只保留 top-k 个最强抑制，重建 inhibitory_mask
        4. 撤销非 top-k 的 inhibitory 贡献

        Args:
            top_k: 保留的最强 inhibitory neuron 数量（默认 1）

        Returns:
            实际保留的 inhibitory neuron 数量
        """
        if not self._inhibit_contributions:
            return 0

        if len(self._inhibit_contributions) <= top_k:
            return len(self._inhibit_contributions)  # 无需竞争

        # 计算每个 inhibitory neuron 的抑制强度（decay 越小 = 抑制越强）
        inhibition_strengths = {}
        for nid, decay in self._inhibit_contributions.items():
            # 抑制强度 = 1 - mean(decay)，decay∈[0,1]，越小抑制越强
            if decay.dim() == 0:
                strength = 1.0 - decay.item()
            else:
                strength = 1.0 - decay.mean().item()
            inhibition_strengths[nid] = strength

        # 选 top-k 最强抑制
        ranked = sorted(inhibition_strengths.items(), key=lambda x: x[1], reverse=True)
        winners = {nid for nid, _ in ranked[:top_k]}

        # 重建 inhibitory_mask：只应用 winners 的 decay
        # 先 reset mask 到全 1（撤销所有 inhibitory）
        if self.inhibitory_mask.dim() == 1:
            self.inhibitory_mask = torch.ones_like(self.inhibitory_mask)
        else:
            self.inhibitory_mask = torch.ones_like(self.inhibitory_mask)

        for nid in winners:
            decay = self._inhibit_contributions[nid]
            self.inhibitory_mask = self.inhibitory_mask * decay

        # 移除非 winners 的贡献记录（它们已被撤销）
        non_winners = set(self._inhibit_contributions.keys()) - winners
        for nid in non_winners:
            del self._inhibit_contributions[nid]

        return len(winners)

    def lateral_inhibition_norm(self, eps: float = 1e-8) -> None:
        """NeuronSpark 融合：场状态 channel-wise L2 归一化。

        人脑启发：lateral inhibition 防止单个神经元主导皮层表征。
        多个 excitatory neuron 写入后，field state 沿他们共同方向累积，
        但 magnitude 无上限。L2 归一化后：
        - 方向保持共识（多个 neuron 共同方向）
        - 幅度 cap 为 1（防止单一 neuron 主导）
        - 与 WTA 互补：lateral norm 约束 excitatory 幅度，WTA 选 inhibitory 方向

        在 WTA 之前调用：先归一化 excitatory 贡献，再让 inhibitory 竞争。
        """
        if self.state.dim() == 1:
            norm = self.state.norm() + eps
            self.state = self.state / norm
        else:
            norm = self.state.norm(dim=-1, keepdim=True) + eps
            self.state = self.state / norm

    def get_effective_state(self) -> torch.Tensor:
        """P0#3: 返回有效场状态 = excitatory_state ⊙ inhibitory_mask。

        L2 归一化后用于 scoring。抑制掩码的乘法衰减在归一化后
        仍保留方向偏置（swamped 的维度对 cosine 贡献减小）。
        """
        if self.state.dim() == 1:
            effective = self.state * self.inhibitory_mask
        else:
            effective = self.state * self.inhibitory_mask
        # Normalize for stable cosine scoring
        norm = effective.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return effective / norm

    def _leave_one_out_state(self, exclude_id: str) -> torch.Tensor:
        """Field state with one neuron's contribution removed (H5 fix + BioOSS inhibitory).

        BioOSS 修复：同时撤销 excitatory 贡献（state - contrib）和 inhibitory 贡献
        （mask / decay），返回与 get_effective_state 一致的语义（state ⊙ mask）。
        原实现仅撤销 excitatory，inhibitory_mask 仍包含被排除 neuron 的衰减，
        导致 inhibitory neuron 评分时其自身衰减未被撤销。
        """
        # 1. 撤销 excitatory 贡献
        contrib = self._contributions.get(exclude_id)
        if contrib is not None:
            if self.state.dim() == 1:
                state_loo = self.state - contrib.squeeze(0)
            else:
                state_loo = self.state - contrib
        else:
            state_loo = self.state

        # 2. 撤销 inhibitory 贡献（BioOSS: mask / decay）
        inhibit_contrib = self._inhibit_contributions.get(exclude_id)
        if inhibit_contrib is not None:
            # decay ∈ (0, 1]，mask = mask * decay，撤销 = mask / decay
            if self.inhibitory_mask.dim() == 1:
                mask_loo = self.inhibitory_mask / (inhibit_contrib.squeeze(0) + 1e-8)
            else:
                mask_loo = self.inhibitory_mask / (inhibit_contrib + 1e-8)
            # 撤销后不应超过 1.0（无抑制状态）
            mask_loo = mask_loo.clamp(max=1.0)
        else:
            mask_loo = self.inhibitory_mask

        # 3. 返回 effective state（state ⊙ mask，与 get_effective_state 语义一致）
        effective = state_loo * mask_loo
        norm = effective.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return effective / norm

    def _condition(self, state: torch.Tensor) -> torch.Tensor:
        """Apply W_cond as a multiplicative gate (H8 fix).

        大规模扩展性修复：
        state 是 N 个单位向量之和，norm ≈ √N。
        N=5 时 norm≈2.2，sigmoid 正常；N=1000 时 norm≈31.6，
        state @ W_cond 放大 14 倍导致 sigmoid 饱和到 0/1，
        W_cond 学习到的门控模式完全失效。
        修复：过 sigmoid 前先归一化 state，保持门控的梯度有意义。
        （score() 内部用 cosine 归一化，只看方向不看幅度，所以输出归一化不影响得分）
        """
        if state.norm() < 1e-8:
            return state
        # 归一化防止 sigmoid 饱和
        if state.dim() == 1:
            state_n = state / (state.norm() + 1e-8)
        else:
            state_n = state / (state.norm(dim=-1, keepdim=True) + 1e-8)
        cond = torch.sigmoid(state_n @ self.W_cond)
        return state_n * cond

    def score(self, vector: torch.Tensor, neuron_id: Optional[str] = None) -> float:
        # P0#3: use effective state (excitatory ⊙ inhibitory_mask) for scoring
        effective = self.get_effective_state()
        score_state = self._leave_one_out_state(neuron_id) if neuron_id else effective
        cond = self._condition(score_state)
        # Normalise cond to a 2-D [..., D] tensor. When the field state is [D]
        # (batch_size=1) cond is [1, D] and broadcasts against a [B, D] vector;
        # when the state is per-sample [B, D] cond matches it sample-for-sample.
        # (Old code did cond.unsqueeze(0) which, for a 2-D [B, D] state, produced
        # [1, B, D] and silently formed a BxB outer product over the batch axis.)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        vec2 = vector if vector.dim() == 2 else vector.unsqueeze(0)
        v_norm = vec2 / (vec2.norm(dim=-1, keepdim=True) + 1e-8)
        sims = (v_norm * cond).sum(dim=-1) / (cond.norm(dim=-1, keepdim=True) + 1e-8)
        return float(sims.mean().item())

    def prediction_complementarity(
        self,
        neuron_a_logits: torch.Tensor,
        neuron_b_logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> float:
        """How much neuron B corrects neuron A's mistakes (H6 fix).

        Measures the log-loss reduction when B is consulted. A token is a
        "mistake" if A assigns low probability to the truth. We measure how
        often B is more right on those tokens.

        Without targets, falls back to disagreement-driven complementarity:
        tokens where A and B disagree give B "weight if B is more confident".
        With targets, it is the actual log-loss reduction from B on A's errors.
        """
        logp_a = torch.log_softmax(neuron_a_logits, dim=-1)
        logp_b = torch.log_softmax(neuron_b_logits, dim=-1)
        pa = torch.exp(logp_a)
        pb = torch.exp(logp_b)

        if targets is not None:
            shift_t = targets[:, 1:].contiguous() if targets.shape == neuron_a_logits.shape[:2] else targets
            shift_a = logp_a[:, :-1, :]
            shift_b = logp_b[:, :-1, :]
            if shift_t.dim() == 2:
                tflat = shift_t.reshape(-1)
                nll_a = -shift_a.reshape(-1, shift_a.size(-1)).gather(-1, tflat.unsqueeze(-1)).squeeze(-1)
                nll_b = -shift_b.reshape(-1, shift_b.size(-1)).gather(-1, tflat.unsqueeze(-1)).squeeze(-1)
            else:
                nll_a = -shift_a.gather(-1, shift_t)
                nll_b = -shift_b.gather(-1, shift_t)
            reduction = (nll_a - nll_b).clamp(min=0.0).mean()
            return float(reduction.item())
        raise_prob_b = (pb.max(dim=-1).values > pa.max(dim=-1).values)
        boost = raise_prob_b.float().mean()
        return float(boost.item())

    def directional_congestion(self, vector: torch.Tensor, active_vectors: List[torch.Tensor]) -> float:
        """计算 vector 与 active_vectors 的平均正向 cosine similarity。

        大规模扩展性修复：
        原实现为 O(N) Python 循环 + 每次 .item() CPU 同步，
        被 ensemble active_filter 外层调用后变成 O(N²)。
        N=1000 时 100 万次迭代 + CPU 同步，可达数分钟。
        改为矩阵乘法一次完成。
        """
        if not active_vectors:
            return 0.0
        # 统一压平到 [D]：处理 [B, D] 批量输入
        if vector.dim() == 2:
            vector = vector.mean(dim=0)
        v_norm = vector / (vector.norm() + 1e-8)

        # stack 所有 active_vectors 到 [N, D]
        flat_vecs = []
        for av in active_vectors:
            av_clean = av.mean(dim=0) if av.dim() == 2 else av
            flat_vecs.append(av_clean)
        stacked = torch.stack(flat_vecs, dim=0)  # [N, D]
        stacked_norm = stacked / (stacked.norm(dim=-1, keepdim=True) + 1e-8)

        # 一次矩阵乘法得到所有 cosine similarity [N]
        sims = (stacked_norm @ v_norm).clamp(min=0.0)
        return float(sims.mean().item())

    def compute_threshold(self, directional_congestion: float) -> float:
        # 动态阈值：拥塞越高门槛越高，但不超过 1.0（cosine similarity 上限）
        # 3.0 系数在上千 neuron 场景下 threshold 恒 > 1.0，导致多 neuron 共振失效
        # 0.7 系数让 threshold ∈ [0.30, 1.00]，高拥塞时只保留高度对齐的 neuron
        return min(1.0, 0.30 + directional_congestion * 0.7)

    def get_state(self) -> torch.Tensor:
        return self.state

    def get_normalised_state(self) -> torch.Tensor:
        """P0#3: 返回抑制掩码调制后的归一化场状态。"""
        return self.get_effective_state()

    def write_history(self, neuron_id: str) -> List[torch.Tensor]:
        # deque 支持 list() 转换和迭代，对外保持 list 语义
        return list(self._write_history.get(neuron_id, []))

    # ── DEAD CODE (R17, REMEDIATION_PLAN 2026-08-14)：生产零调用者
    # （inhibitory 语义由 write_inhibit/BioOSS 处理），保留审计证据。──
    def get_contribution_sign(self, neuron_id: str) -> int:
        """返回神经元对场的贡献符号（人脑启发：抑制性神经元返回 -1）。

        通过检查存储的 contribution 向量的"主导方向"判断符号：
        - 若 contribution 与某个参考正向（如第一个 excitatory 神经元）同向 → +1
        - 若反向 → -1

        简化实现：直接检查 contribution 的 L2 归一化前的原始符号不可行
        （L2 归一化后只有方向，符号已编码在方向中）。
        因此这里用 contribution 的首元素符号作为快速判断。
        """
        contrib = self._contributions.get(neuron_id)
        if contrib is None:
            return 0
        # 取首元素符号（contribution 是 L2 归一化的，首元素符号代表整体方向）
        first_elem = contrib.flatten()[0].item()
        return 1 if first_elem >= 0 else -1

    def clear_history(self) -> None:
        self._write_history.clear()

    def save_round_state(self) -> Dict[str, torch.Tensor]:
        """S12: 保存当前轮次的场状态快照（用于多轮对话持久化）。

        保存内容：
        - state: 当前场状态（含累积写入）
        - inhibitory_mask: 抑制掩码
        - contributions: 各神经元贡献（用于 leave-one-out 恢复）

        Returns:
            state_dict: 可直接传给 load_round_state 恢复
        """
        return {
            "state": self.state.detach().clone(),
            "inhibitory_mask": self.inhibitory_mask.detach().clone(),
            "contributions": {
                nid: contrib.detach().clone()
                for nid, contrib in self._contributions.items()
            },
            "inhibit_contributions": {
                nid: contrib.detach().clone()
                for nid, contrib in self._inhibit_contributions.items()
            },
            "batch_size": self._batch_size,
        }

    def load_round_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """S12: 加载之前保存的场状态（多轮对话恢复上一轮上下文）。

        Args:
            state_dict: save_round_state() 的返回值
        """
        self.state = state_dict["state"].clone()
        self.inhibitory_mask = state_dict["inhibitory_mask"].clone()
        self._contributions = {
            nid: contrib.clone()
            for nid, contrib in state_dict.get("contributions", {}).items()
        }
        self._inhibit_contributions = {
            nid: contrib.clone()
            for nid, contrib in state_dict.get("inhibit_contributions", {}).items()
        }
        self._batch_size = state_dict.get("batch_size", 1)
        # 清空 write_history，避免历史污染（新轮次重新累积）
        self._write_history.clear()
        self.n_active = len(self._contributions)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, n_writes={self.n_active}"
