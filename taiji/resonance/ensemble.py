"""Resonance ensemble — multi-round collaborative inference.

The ensemble orchestrates multiple ResonanceNeurons through the
ResonanceField over 3-5 rounds of collaborative inference.

Each round:
1. All active neurons run forward (first round: independently;
   subsequent rounds: conditioned on field state)
2. Each neuron writes its L2-normalised field vector
3. Resonance scores are computed (cosine similarity with field state)
4. Low-resonance neurons are filtered out via dynamic thresholding

P7: 支持域专用 vocab（每 neuron 独立 embedding + 独立 lm_head）。
同一批 token IDs 送给所有 neuron，但每 neuron 用自己的 embedding 编码。
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .field import ResonanceField
from .neuron import ResonanceNeuron


class ResonanceEnsemble:
    """Orchestrates multi-round resonance inference across multiple neurons.

    P7: 简化为直接共振——移除 ConfidenceGate/EarlyStop/QualityFilter/DivisionPath/DomainRouter，
    这些机制在旧 teacher-based 架构中设计，与新 P7 从头训练路径不兼容。

    Usage:
        ensemble = ResonanceEnsemble(neurons, field)
        result = ensemble.forward(shared_embeddings=shared_emb, return_logits=True)
    """

    def __init__(
        self,
        neurons: Dict[str, ResonanceNeuron],
        field: ResonanceField,
        max_rounds: int = 3,
        diversity_lambda: float = 0.01,
        logits_top_k: int = 64,
        stdp_tracker: Optional[Any] = None,
        coaction: Optional[Any] = None,
        neuromodulator: Optional[Any] = None,
        maturity: Optional[Any] = None,
        gamma_oscillator: Optional[Any] = None,
    ):
        self.neurons = neurons
        self.field = field
        self.max_rounds = max_rounds
        self.diversity_lambda = diversity_lambda

        # ── Bio-inspired trackers（P1 接线）──
        self.stdp_tracker = stdp_tracker
        self.coaction = coaction
        self.neuromodulator = neuromodulator
        # MaturityTracker: 幼稚态低共振权重（0.1），成熟态 1.0
        # 新生神经元先听后说，不污染集体意识场
        self.maturity = maturity
        # KoPE/Kuramoto: 相位耦合（共激活强的 neuron 相位同步）
        self.gamma_oscillator = gamma_oscillator

        # ── RSGN 融合: 几何坐标空间（神经元距离衰减先验）──
        from .geometry import NeuronGeometry
        self.geometry = NeuronGeometry(embedding_dim=8, sigma=0.5)
        self._init_geometry()

        # ── 大规模内存控制（B2/B3 fix）──
        self.logits_top_k = logits_top_k
        self._logits_keep_ids: Optional[set] = None

        # Tracking
        self.round_scores: List[Dict[str, float]] = []
        self.n_active_history: List[int] = []

    def _init_geometry(self) -> None:
        """初始化 NeuronGeometry：按域分组分配坐标，注册到 coaction。"""
        from collections import defaultdict
        domain_to_nids = defaultdict(list)
        for nid in self.neurons:
            # 从 nid 提取 domain（格式: "domain" 或 "domain_N"）
            domain = nid.split("_")[0] if "_" in nid else nid
            domain_to_nids[domain].append(nid)

        self.geometry.assign_domain_positions(
            dict(domain_to_nids),
            intra_domain_radius=0.2,
            inter_domain_radius=1.0,
        )

        # 注册到 coaction tracker（RSGN 距离先验自动生效）
        if self.coaction is not None and hasattr(self.coaction, "register_geometry"):
            self.coaction.register_geometry(self.geometry)

    def add_neuron(self, nid: str, neuron: ResonanceNeuron, from_split: Optional[str] = None) -> None:
        """运行时添加新神经元到 ensemble（neurogenesis 入口）。

        新神经元必须与现有神经元共享 field_dim 和 hidden_size
        （由 Cortex.add_neuron 保证，这里做防御性校验）。

        Args:
            nid: 神经元 ID（如 "zh_1"）
            neuron: ResonanceNeuron 实例
            from_split: 分裂父 neuron ID（LuminaNet splitting），
                        用于在几何空间中放置子 neuron 在父 neuron 附近
        """
        if nid in self.neurons:
            raise ValueError(f"神经元 {nid} 已存在于 ensemble")

        # 校验 field_dim 一致性
        if self.neurons:
            existing_field_dim = next(iter(self.neurons.values())).config.field_dim
            if neuron.config.field_dim != existing_field_dim:
                raise ValueError(
                    f"新神经元 field_dim={neuron.config.field_dim} 与 ensemble "
                    f"field_dim={existing_field_dim} 不一致"
                )
            existing_hidden = next(iter(self.neurons.values())).config.hidden_size
            if neuron.config.hidden_size != existing_hidden:
                raise ValueError(
                    f"新神经元 hidden_size={neuron.config.hidden_size} 与 ensemble "
                    f"hidden_size={existing_hidden} 不一致"
                )

        self.neurons[nid] = neuron
        # 确保 refractory_counter 在正确 device 上
        neuron.refractory_counter = neuron.refractory_counter.to(
            next(iter(self.neurons.values())).refractory_counter.device
        )

        # RSGN 融合: 新 neuron 加入几何空间
        # splitting 模式下靠近父 neuron，新建模式下靠近同域中心
        if hasattr(self, 'geometry') and self.geometry is not None:
            if from_split is not None and from_split in self.geometry.positions:
                # 分裂模式：子 neuron 在父 neuron 附近（小偏移）
                parent_pos = self.geometry.positions[from_split]
                offset = torch.randn_like(parent_pos) * 0.05
                self.geometry.assign_position(nid, parent_pos + offset)
            else:
                # 新建模式：在同域中心附近随机放置
                domain = nid.split("_")[0] if "_" in nid else nid
                domain_nids = [
                    dn for dn in self.geometry.positions
                    if (dn.split("_")[0] if "_" in dn else dn) == domain
                ]
                if domain_nids:
                    # 取同域 neuron 中心
                    center = torch.stack([
                        self.geometry.positions[dn] for dn in domain_nids
                    ]).mean(dim=0)
                    offset = torch.randn_like(center) * 0.05
                    self.geometry.assign_position(nid, center + offset)
                else:
                    # 新域：随机放置
                    pos = torch.randn(self.geometry.embedding_dim) * 0.3
                    self.geometry.assign_position(nid, pos)

    def _parallel_forward(
        self,
        active_ids,
        shared_embeddings: Optional[torch.Tensor],
        field_state,
        round_num: int,
        return_logits_filter,
        neuron_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        mm_logits_modality: Optional[str] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """并行 forward 多个神经元（人脑启发：神经元并行工作）。

        GPU 模式下用 CUDA stream 真并行，保留 per-neuron 独立性。
        CPU 模式下退化为串行（无 stream 开销）。

        所有 neuron 共享同一份 shared_embeddings（来自外部共享嵌入表），
        P8 多模态路径可通过 neuron_embeddings 为不同 neuron 提供不同输入。

        Args:
            active_ids: 要 forward 的 neuron id 集合
            shared_embeddings: [B, L, base_embed_dim] 共享嵌入（所有 neuron 共用）
            field_state: 场状态（round 1 为 None）
            round_num: 轮次
            return_logits_filter: callable(nid) -> bool，决定哪些 neuron 返回 logits
            neuron_embeddings: P8 路径，{nid: [B, L, base_embed_dim]} 预编码 embedding
            mm_logits_modality: 多模态输出模态，不为 None 时所有 neuron 用 mm_lm_head 输出

        Returns:
            (round_vecs, round_logits)
        """
        round_vecs: Dict[str, torch.Tensor] = {}
        round_logits: Dict[str, torch.Tensor] = {}

        # 确定参考 tensor（device 信息来源）
        ref_tensor = (neuron_embeddings[next(iter(neuron_embeddings))]
                      if neuron_embeddings else shared_embeddings)

        def _get_emb(nid: str) -> torch.Tensor:
            """获取 neuron 的输入 embedding，优先级：neuron_embeddings > shared_embeddings."""
            if neuron_embeddings is not None and nid in neuron_embeddings:
                return neuron_embeddings[nid]
            return shared_embeddings

        def _forward_neuron(nid: str, emb: torch.Tensor, need_logits: bool) -> Dict:
            """单个 neuron forward，统一封装多模态参数。"""
            kwargs = dict(
                field_state=field_state,
                round_num=round_num,
                return_logits=need_logits,
            )
            if mm_logits_modality is not None:
                kwargs["mm_logits_modality"] = mm_logits_modality
            return self.neurons[nid].forward(emb, **kwargs)

        is_cuda = ref_tensor.is_cuda
        if is_cuda and len(active_ids) > 1:
            # GPU 模式：CUDA stream 真并行
            streams = {nid: torch.cuda.Stream() for nid in active_ids}
            results: Dict[str, Dict] = {}

            for nid in active_ids:
                need_logits = return_logits_filter(nid)
                emb = _get_emb(nid)
                with torch.cuda.stream(streams[nid]):
                    results[nid] = _forward_neuron(nid, emb, need_logits)

            # 等待所有 stream 完成
            for nid in active_ids:
                torch.cuda.current_stream().wait_stream(streams[nid])

            for nid in active_ids:
                round_vecs[nid] = results[nid]["field_vector"]
                if return_logits_filter(nid):
                    round_logits[nid] = results[nid]["logits"]
        else:
            # CPU 模式或单神经元：串行
            for nid in active_ids:
                need_logits = return_logits_filter(nid)
                emb = _get_emb(nid)
                result = _forward_neuron(nid, emb, need_logits)
                round_vecs[nid] = result["field_vector"]
                if need_logits:
                    round_logits[nid] = result["logits"]

        return round_vecs, round_logits

    def forward(
        self,
        shared_embeddings: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        active_filter: bool = True,
        active_nids: Optional[List[str]] = None,
        neuron_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        mm_logits_modality: Optional[str] = None,
    ) -> Dict:
        """Run the full resonance loop.

        所有 neuron 共享同一份 shared_embeddings（来自外部共享嵌入表）。
        P8: 支持 neuron_embeddings 预编码路径（多模态）：
        - neuron_embeddings: {nid: [B, L, base_embed_dim]} 跳过共享嵌入

        至少提供 shared_embeddings 或 neuron_embeddings 之一。
        优先级：neuron_embeddings > shared_embeddings。

        Args:
            shared_embeddings: [B, L, base_embed_dim] 共享嵌入（所有 neuron 共用）
            return_logits: if True, each neuron also returns token logits
            active_filter: if True, filter out low-resonance neurons each round
            active_nids: 如果提供，只 forward 这些 neuron（Phase 5.1 丘脑路由用）
                        None 表示全部参与（默认行为，向后兼容）
            neuron_embeddings: P8 路径，{nid: [B, L, base_embed_dim]} 预编码 embedding
            mm_logits_modality: 多模态输出模态，不为 None 时所有 neuron 用 mm_lm_head 输出

        Returns:
            dict with:
            - field_state: final field state vector
            - weighted_logits: resonance-weighted average logits (if return_logits)
            - final_scores: per-neuron resonance scores (final round)
            - n_rounds: actual number of rounds completed
            - skipped_resonance: True if gating skipped the resonance loop
            - skip_reason: explanation if resonance was skipped
        """
        if shared_embeddings is None and neuron_embeddings is None:
            raise ValueError(
                "[ResonanceEnsemble.forward] 必须提供 shared_embeddings 或 neuron_embeddings"
            )
        # 参考 tensor：用于 batch_size 和 device 信息
        if neuron_embeddings is not None:
            ref = next(iter(neuron_embeddings.values()))
        else:
            ref = shared_embeddings

        self.field.reset(batch_size=ref.shape[0])
        self.round_scores = []
        self.n_active_history = []
        self._logits_keep_ids = None  # 每次 forward 重置

        # P0-2 fix (MAJOR-2): 重置所有 neurons 的 refractory_counter
        # 防止跨 forward 调用的状态泄漏（上次推理进入不应期的 neuron 不应影响本次）
        for nid in self.neurons:
            self.neurons[nid].refractory_counter.fill_(0)

        # P1-STDP: 每次推理开始时清空 firing history（一次推理内的发放时序）
        if self.stdp_tracker is not None:
            self.stdp_tracker._firing_history.clear()

        neuron_ids = list(self.neurons.keys())
        # 如果指定了 active_nids，只激活这些 neuron（精简模式）
        if active_nids is not None:
            active_ids = set(nid for nid in active_nids if nid in self.neurons)
            if not active_ids:
                # fallback: 全部 neuron
                active_ids = set(neuron_ids)
        else:
            active_ids = set(neuron_ids)

        vectors: Dict[str, torch.Tensor] = {}
        all_logits: Dict[str, torch.Tensor] = {}
        logits_history: List[torch.Tensor] = []

        # ── Round 1: all neurons run independently ──
        # 大规模内存优化（B2 peak fix）：
        # N > top_K 时，round 1 不为所有 neuron 请求 logits（避免 O(N) 峰值内存）
        # 只取 field_vector 算分，然后只为 best_nid 重新 forward 获取 logits（用于 gating）
        # N ≤ top_K 时保持原行为（全部请求，因为都会保留）
        large_scale = return_logits and len(active_ids) > self.logits_top_k
        round1_return_logits = return_logits and not large_scale

        # Q4 fix: 使用 _parallel_forward（GPU 自动 CUDA stream 并行）
        def round1_logits_filter(nid):
            return round1_return_logits

        round_vecs, round_logits = self._parallel_forward(
            active_ids,
            shared_embeddings,
            field_state=None,
            round_num=1,
            return_logits_filter=round1_logits_filter,
            neuron_embeddings=neuron_embeddings,
            mm_logits_modality=mm_logits_modality,
        )

        # Write round 1 to field
        # P1-2: 从 NeuromodulatorState 读取 field_write_scale（去甲肾上腺素驱动）
        write_scale = (self.neuromodulator.get_field_write_scale()
                       if self.neuromodulator is not None else 1.0)
        for nid in active_ids:
            # P0#3: 抑制性神经元走 write_inhibit（乘法衰减），兴奋性走 write（累加）
            neuron = self.neurons[nid]
            # MaturityTracker: 幼稚态低共振权重（0.1），成熟态 1.0
            maturity_w = (self.maturity.get_resonance_weight(nid)
                          if self.maturity is not None else 1.0)
            if neuron.is_inhibitory:
                self.field.write_inhibit(nid, round_vecs[nid], weight=maturity_w)
            else:
                self.field.write(nid, round_vecs[nid], scale=write_scale * maturity_w)
            # P1-STDP: 记录 round 1 发放（用于 sleep 期 STDP 强化）
            if self.stdp_tracker is not None:
                self.stdp_tracker.record_firing(nid, 1, round_vecs[nid])
            # P1-Coactivation: 记录共激活（同轮 forward 的 neuron 互为共激活）
            if self.coaction is not None:
                self.coaction.update(active_ids, round_num=1)

        # NeuronSpark 融合：lateral-inhibition normalization
        # 场状态 L2 归一化，防止单一 neuron 方向主导 magnitude。
        # 与 WTA 互补：lateral norm 约束 excitatory 幅度，WTA 选 inhibitory 方向。
        try:
            self.field.lateral_inhibition_norm()
        except Exception:
            pass  # 非关键，失败不影响推理

        # Deviance detection 融合：inhibitory neuron 竞争性抑制（WTA）
        # 多个 inhibitory neuron 写入后，只保留 top-1 最强抑制方向，
        # 避免全场过度衰减。只有 ≥2 个 inhibitory neuron 时才触发竞争。
        n_inhibitory = sum(1 for nid in active_ids if self.neurons[nid].is_inhibitory)
        if n_inhibitory >= 2:
            try:
                self.field.apply_inhibitory_wta(top_k=1)
            except Exception:
                pass  # WTA 失败非关键，保持原有累积抑制

        # P0-2 fix: 不应期错峰 — 不再全部 enter_refractory（否则 round 2+ 全部 refractory 无人写入）
        # 改为：只让 round 1 分数排名前 top_K 的 neuron 进入不应期
        # 这样 round 2+ 中分数较低的 neuron 有机会写入，实现信息轮替
        scores: Dict[str, float] = {}
        for nid in active_ids:
            scores[nid] = self.field.score(round_vecs[nid], neuron_id=nid)
        self.round_scores.append(scores)

        # 按 score 降序排序，只让 top-K 进入不应期（K = min(half, logits_top_k)）
        # P1-2: refractory_multiplier 由 NeuromodulatorState 提供（血清素驱动）
        refractory_mult = (self.neuromodulator.get_refractory_multiplier()
                           if self.neuromodulator is not None else 1.0)
        ranked_round1 = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        refractory_k = max(1, min(len(ranked_round1) // 2, self.logits_top_k))
        for nid, _ in ranked_round1[:refractory_k]:
            self.neurons[nid].enter_refractory(multiplier=refractory_mult)

        # ── B2/B3 fix: round 1 后确定 top-K ──
        if return_logits:
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            self._logits_keep_ids = {nid for nid, _ in ranked[:self.logits_top_k]}

            if large_scale:
                # 大规模：只为 best_nid 重新 forward 获取 logits（gating 需要）
                # 代价是 1 次额外 forward，但避免了 N 份 logits 同时存活
                best_nid = ranked[0][0]
                best_kwargs = dict(
                    field_state=None,
                    round_num=1,
                    return_logits=True,
                )
                if mm_logits_modality is not None:
                    best_kwargs["mm_logits_modality"] = mm_logits_modality
                best_emb = (neuron_embeddings[best_nid]
                            if neuron_embeddings is not None and best_nid in neuron_embeddings
                            else shared_embeddings)
                best_result = self.neurons[best_nid].forward(best_emb, **best_kwargs)
                round_logits[best_nid] = best_result["logits"]
            else:
                # 小规模：round 1 已获取所有 logits，丢弃非 top-K
                if len(round_logits) > self.logits_top_k:
                    non_keep = set(round_logits.keys()) - self._logits_keep_ids
                    for nid in non_keep:
                        del round_logits[nid]

        # Track round 1 for logits averaging
        if return_logits and round_logits:
            logits_history.append(self._average_logits(round_logits))
        vectors = round_vecs
        all_logits = round_logits

        # 修复：round 1 正常完成后也记录 n_active（之前只在 skip 或 round 2+ 记录）
        self.n_active_history.append(len(active_ids))

        # ── Rounds 2+: conditioned resonance ──
        for round_num in range(2, self.max_rounds + 1):
            # P0-2 fix: round 2+ 也基于当前 _logits_keep_ids 过滤，但每轮会重新计算
            def round2_logits_filter(nid):
                return return_logits and (
                    self._logits_keep_ids is None or nid in self._logits_keep_ids
                )

            round_vecs, round_logits = self._parallel_forward(
                active_ids,
                shared_embeddings,
                field_state=self.field.get_normalised_state(),
                round_num=round_num,
                return_logits_filter=round2_logits_filter,
                neuron_embeddings=neuron_embeddings,
                mm_logits_modality=mm_logits_modality,
            )

            # 人脑启发：不应期调度（错峰写入）
            writable_ids = []
            refractory_ids = []
            for nid in active_ids:
                if self.neurons[nid].in_refractory:
                    refractory_ids.append(nid)
                else:
                    writable_ids.append(nid)

            for nid in writable_ids:
                # P0#3: 抑制性神经元走 write_inhibit，兴奋性走 update
                neuron = self.neurons[nid]
                # MaturityTracker: 幼稚态低共振权重（0.1），成熟态 1.0
                maturity_w = (self.maturity.get_resonance_weight(nid)
                              if self.maturity is not None else 1.0)
                if neuron.is_inhibitory:
                    self.field.write_inhibit(nid, round_vecs[nid], weight=maturity_w)
                else:
                    # P1-2: round 2+ 也应用 neuromodulator 调质
                    self.field.update(nid, round_vecs[nid], scale=write_scale * maturity_w)
                self.neurons[nid].enter_refractory(multiplier=refractory_mult)
                # P1-STDP: 记录 round 2+ 发放
                if self.stdp_tracker is not None:
                    self.stdp_tracker.record_firing(nid, round_num, round_vecs[nid])
            # P1-Coactivation: 更新共激活
            if self.coaction is not None and writable_ids:
                self.coaction.update(writable_ids, round_num=round_num)

            # NeuronSpark lateral-inhibition norm (round 2+)
            try:
                self.field.lateral_inhibition_norm()
            except Exception:
                pass

            # P0-2 fix: leave-one-out 双重减法 bug 修复
            # 原 bug：这里减去 old_contrib，但 field.score() 内部 _leave_one_out_state 又减一次
            # 修复：减去后清除 _contributions[nid]，让 _leave_one_out_state 返回原 state
            for nid in refractory_ids:
                old_contrib = self.field._contributions.get(nid)
                if old_contrib is not None:
                    if self.field.state.dim() == 1 and old_contrib.dim() == 1:
                        self.field.state = self.field.state - old_contrib
                    elif self.field.state.dim() == old_contrib.dim():
                        self.field.state = self.field.state - old_contrib
                    else:
                        self.field.state = self.field.state - old_contrib.squeeze(0)
                    # P0-2 fix: 清除 contribution 记录，避免 leave-one-out 双重减法
                    del self.field._contributions[nid]

            scores = {}
            for nid in active_ids:
                scores[nid] = self.field.score(round_vecs[nid], neuron_id=nid)
            self.round_scores.append(scores)

            # P0-2 fix: 每轮基于当前 scores 重新计算 _logits_keep_ids（原 bug：round 1 后冻结）
            if return_logits:
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                self._logits_keep_ids = {nid for nid, _ in ranked[:self.logits_top_k]}

            if active_filter and len(active_ids) > 1:
                # P0-2 fix: directional_congestion 排除自身（原 bug：自指导致小 N 时 threshold 过高）
                filtered = set()
                active_list = list(active_ids)
                for nid in active_list:
                    other_vecs = [round_vecs[o] for o in active_list if o != nid]
                    if not other_vecs:
                        filtered.add(nid)
                        continue
                    congestion = self.field.directional_congestion(round_vecs[nid], other_vecs)
                    threshold = self.field.compute_threshold(congestion)
                    if scores[nid] >= threshold:
                        filtered.add(nid)
                if not filtered:
                    best = max(active_ids, key=lambda nid: scores[nid])
                    filtered.add(best)
                self.field.scores = scores
                if len(filtered) <= 1 and round_num >= 2:
                    active_ids = filtered
                    self.n_active_history.append(len(active_ids))
                    vectors = round_vecs
                    all_logits = round_logits
                    break
                active_ids = filtered

            # 人脑启发：每轮结束递减所有神经元的不应期计数器
            for nid in self.neurons:
                self.neurons[nid].tick_refractory()

            # KoPE/Kuramoto: 相位耦合 — 共激活强的 neuron 相位相互牵引
            if self.gamma_oscillator is not None and hasattr(self.gamma_oscillator, "kuramoto_step"):
                try:
                    self.gamma_oscillator.kuramoto_step(
                        coupling_strength=0.05,
                        active_ids=active_ids,
                        coactivation=self.coaction,
                    )
                except Exception:
                    pass  # 相位耦合失败不影响共振主流程

            self.n_active_history.append(len(active_ids))
            vectors = round_vecs
            all_logits = round_logits

        # ── Final output ──
        result = {
            "field_state": self.field.get_state(),
            "final_scores": self.round_scores[-1] if self.round_scores else {},
            "n_rounds": len(self.round_scores),
            "n_active_history": self.n_active_history,
            "skipped_resonance": False,
            "skip_reason": None,
        }

        if return_logits and all_logits:
            # P7: 返回每 neuron 的原始 logits（域 vocab 空间可能不同）
            # 供 _generate_p7 提取目标 neuron logits 用于 decoding
            result["neuron_logits"] = all_logits

            # P7: 不同 neuron vocab 大小不同时跳过加权合并
            vocab_sizes = [logits.shape[-1] for logits in all_logits.values()]
            if len(set(vocab_sizes)) == 1:
                self._compute_per_position_weights(
                    all_logits, vectors, scores, result, ref,
                )
            # else: neuron_logits already in result, _generate_p7 handles extraction

        return result

    def forward_train(
        self,
        shared_embeddings: torch.Tensor,
        temperature: float = 1.0,
        gamma_oscillator=None,
    ) -> Dict[str, torch.Tensor]:
        """单轮全可微前向，用于联合训练。

        与 forward()（推理）的区别：
        - 单轮（无多轮共振）
        - 全可微（无 .item()、无 argmax、无 hard top-K）
        - 内联计算共振分（绕过 field.score() 的 detach）
        - 返回 fused_logits + 负载均衡 loss
        - 支持振荡同步门控（gamma_oscillator）

        前向传播时所有神经元参与，共振场聚合，反向传播流经聚合权重
        → 神经元学习如何写入场、如何协同输出。

        振荡同步（可选）：
          gamma_oscillator 提供 per-neuron gate_factor ∈ [0.2, 1.0]
          同相位的神经元 gate≈1.0（增强），反相位的 gate≈0.2（衰减）
          模拟人脑 Gamma 振荡的 feature binding 机制

        Args:
            shared_embeddings: [B, L, base_embed_dim] 共享嵌入
            temperature: softmax 温度（低=更尖锐选择）
            gamma_oscillator: GammaOscillator 实例（可选，None=不门控）

        Returns:
            dict with:
            - fused_logits: [B, L, V] 软加权聚合 logits
            - weights: [N] 软路由权重
            - scores: [N] 共振分
            - balance_loss: scalar 负载均衡 loss（负熵，越小越均匀）
            - individual_logits: {nid: [B, L, V]} 个体 logits（分析用）
        """
        active_ids = list(self.neurons.keys())
        N = len(active_ids)

        # 1. 所有神经元前向（return_logits=True，梯度贯穿全程）
        all_vecs = []
        all_logits = []
        for nid in active_ids:
            result = self.neurons[nid].forward(shared_embeddings, return_logits=True)
            all_vecs.append(result["field_vector"])   # [B, D]
            all_logits.append(result["logits"])        # [B, L, V]

        all_vecs = torch.stack(all_vecs)      # [N, B, D]
        all_logits = torch.stack(all_logits)  # [N, B, L, V]

        # 2. 场状态 = 所有 field_vector 归一化后求和（可微加法）
        all_vecs_norm = F.normalize(all_vecs, dim=-1)  # [N, B, D]
        field_state = all_vecs_norm.sum(dim=0)         # [B, D]

        # 3. Leave-one-out 共振分（可微 cosine similarity）
        # loo_state[i] = field_state - all_vecs_norm[i]（去掉自己的贡献）
        loo_state = field_state.unsqueeze(0) - all_vecs_norm  # [N, B, D]
        loo_norm = F.normalize(loo_state, dim=-1)             # [N, B, D]
        scores = (all_vecs_norm * loo_norm).sum(dim=-1)       # [N, B] cosine sim
        scores = scores.mean(dim=1)                           # [N] batch 平均

        # 振荡同步门控：相位对齐的神经元获得更高权重（feature binding）
        if gamma_oscillator is not None:
            gate_factors = gamma_oscillator.batch_gate_factors(active_ids)  # [N]
            scores = scores * gate_factors.to(scores.device)

        # 4. 软加权聚合（可微，无 .item()——关键修复点）
        weights = F.softmax(scores / temperature, dim=0)      # [N]
        fused_logits = torch.einsum('n,nblv->blv', weights, all_logits)  # [B, L, V]

        # 5. 负载均衡 loss（负熵：uniform时=-log(N)，collapse时→0）
        balance_loss = -(weights * torch.log(weights + 1e-8)).sum()

        return {
            "fused_logits": fused_logits,
            "weights": weights,
            "scores": scores,
            "balance_loss": balance_loss,
            "individual_logits": {nid: all_logits[i]
                                   for i, nid in enumerate(active_ids)},
        }

    def _average_logits(
        self, logits_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute simple average of logits across neurons for early stop.

        P7: 不同 neuron 的 logits 可能在不同 vocab 空间（10k-20k）。
        先 pad 到统一大小再 stack。
        """
        if not logits_dict:
            return torch.zeros(1)

        logits_list = list(logits_dict.values())
        # Check if all logits have the same shape
        shapes = [logits.shape[-1] for logits in logits_list]
        if len(set(shapes)) == 1:
            stacked = torch.stack(logits_list)
            return stacked.mean(dim=0)

        # P7: neurons have different vocab sizes; pad to max
        max_vocab = max(shapes)
        device = logits_list[0].device
        padded = []
        for logits in logits_list:
            if logits.shape[-1] < max_vocab:
                pad = torch.zeros(
                    *logits.shape[:-1], max_vocab - logits.shape[-1],
                    device=device, dtype=logits.dtype,
                )
                logits = torch.cat([logits, pad], dim=-1)
            padded.append(logits)
        stacked = torch.stack(padded)
        return stacked.mean(dim=0)

    def _dynamic_logit_fusion(
        self,
        all_logits: Dict[str, torch.Tensor],
        scores: Dict[str, float],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """MoCo-inspired dynamic logit fusion.

        Each step re-computes field scores and dynamically weights all neurons' logits.
        This replaces static weighting with adaptive, context-aware fusion.

        Args:
            all_logits: {nid: [B, L, vocab]} per-neuron logits
            scores: {nid: float} current resonance scores
            temperature: softmax temperature for weighting (lower = sharper selection)

        Returns:
            fused_logits: [B, L, vocab] dynamically weighted logits
        """
        if not all_logits:
            raise ValueError("[_dynamic_logit_fusion] all_logits is empty")

        neuron_ids = list(all_logits.keys())
        n_neurons = len(neuron_ids)

        # Get reference tensor for device/dtype
        ref_logits = next(iter(all_logits.values()))
        device = ref_logits.device

        # 1. Compute dynamic weights from field scores
        score_vals = torch.tensor(
            [float(scores.get(nid, 0.0)) for nid in neuron_ids],
            device=device,
        )
        # Apply temperature sharpening (MoCo-style)
        weights = F.softmax(score_vals / temperature, dim=0)  # [N]

        # 2. Check if all logits have the same vocab size
        vocab_sizes = [logits.shape[-1] for logits in all_logits.values()]
        if len(set(vocab_sizes)) == 1:
            # Same vocab: direct weighted sum
            fused_logits = None
            for i, (nid, logits) in enumerate(all_logits.items()):
                w = weights[i].item()
                if fused_logits is None:
                    fused_logits = w * logits
                else:
                    fused_logits = fused_logits + w * logits
            return fused_logits

        # 3. Different vocab sizes: pad to max (P7 compatibility)
        max_vocab = max(vocab_sizes)
        fused_logits = torch.zeros(
            ref_logits.shape[0], ref_logits.shape[1], max_vocab,
            device=device, dtype=ref_logits.dtype,
        )

        for i, (nid, logits) in enumerate(all_logits.items()):
            w = weights[i].item()
            vocab_size = logits.shape[-1]
            if vocab_size < max_vocab:
                pad = torch.zeros(
                    logits.shape[0], logits.shape[1], max_vocab - vocab_size,
                    device=device, dtype=logits.dtype,
                )
                logits = torch.cat([logits, pad], dim=-1)
            fused_logits = fused_logits + w * logits

        return fused_logits

    def _compute_per_position_weights(
        self,
        all_logits: Dict[str, torch.Tensor],
        vectors: Dict[str, torch.Tensor],
        scores: Dict[str, float],
        result: dict,
        ref: torch.Tensor,
    ) -> None:
        """Per-position routing (v2): logit-entropy weighting + complementarity.

        Each position independently picks the neuron that is most confident.
        Complementarity scores boost neurons bringing new information.
        Memory-efficient: process one neuron at a time for entropy.

        Only called when all neurons share the same vocab size.
        """
        neuron_ids = list(all_logits.keys())
        entropies = []
        for nid in neuron_ids:
            log_probs = F.log_softmax(all_logits[nid], dim=-1)
            probs = torch.exp(log_probs)
            ent = -(probs * log_probs).sum(dim=-1)  # [B, L]
            entropies.append(ent)
        ent_stack = torch.stack(entropies)  # [N, B, L]
        # Lower entropy = more confident = higher weight.
        # H7: sharpen confidence temperature 2.0 -> 3.0 so a clearly
        # more-confident neuron dominates its positions more decisively.
        confidence = 1.0 / (ent_stack + 1e-8)  # [N, B, L]
        position_weights = F.softmax(confidence * 3.0, dim=0)  # [N, B, L]

        # H5: lift neurons the field actually resonated with. final_scores
        # are the last round's leave-one-out resonance scores in [0,1]; map
        # them to a multiplicative boost in [1,2] so the field's verdict
        # survives into per-position routing instead of being washed out
        # by the per-token softmax.
        final_scores = self.round_scores[-1] if self.round_scores else scores
        score_vals = torch.tensor(
            [float(final_scores.get(nid, 0.0)) for nid in neuron_ids],
            device=ref.device,
        )
        position_weights = position_weights * (1.0 + score_vals).unsqueeze(-1).unsqueeze(-1)

        # H6: reward neurons that correct the others' mistakes. This
        # replaces the legacy geometric orthogonality term (kept on the
        # field as complementarity_score for diagnostics only); routing
        # now uses prediction_complementarity, as field.py documents.
        if hasattr(self.field, 'prediction_complementarity') and len(neuron_ids) > 1:
            comp_vals = []
            for i, nid in enumerate(neuron_ids):
                other_logits = [all_logits[o] for j, o in enumerate(neuron_ids) if j != i]
                c = 0.0
                for other in other_logits:
                    c += self.field.prediction_complementarity(other, all_logits[nid])
                comp_vals.append(c)
            comp_boost = torch.tensor(comp_vals, device=ref.device)
            position_weights = position_weights * (1.0 + comp_boost).unsqueeze(-1).unsqueeze(-1)

        # Non-zero floor so no specialist is ever fully silenced (a 0%
        # neuron contributes nothing and can never be learned from),
        # then renormalise so the mixture still sums to 1 over neurons.
        position_weights = position_weights.clamp(min=0.01)
        position_weights = position_weights / position_weights.sum(dim=0, keepdim=True)

        # Apply per-position weights (memory-efficient: one at a time)
        weighted_logits = None
        for i, (nid, logits) in enumerate(all_logits.items()):
            w = position_weights[i]  # [B, L]
            if weighted_logits is None:
                weighted_logits = w.unsqueeze(-1) * logits
            else:
                weighted_logits = weighted_logits + w.unsqueeze(-1) * logits
        result["weighted_logits"] = weighted_logits
        result["final_weights"] = {
            nid: float(position_weights[i].mean().item())
            for i, nid in enumerate(neuron_ids)
        }

    def evaluate_ppl(
        self,
        dataloader,
        shared_embedding: nn.Embedding,
        tokenizer=None,
        max_batches: int = 50,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Evaluate perplexity over a dataloader using the resonance ensemble.

        Uses teacher forcing: feeds the full sequence, gets predictions
        at all positions, computes cross-entropy loss.

        Args:
            dataloader: yields batches of token_ids [B, L]
            shared_embedding: the shared base embedding (Level 0)
            tokenizer: optional tokenizer for decoding (debug only)
            max_batches: maximum number of batches to evaluate
            verbose: print progress

        Returns:
            dict with 'ppl', 'loss', 'n_tokens'
        """
        total_loss = 0.0
        total_tokens = 0

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            # Handle different batch formats
            if isinstance(batch, dict):
                input_ids = batch.get("input_ids") or batch.get("tokens")
                target_ids = batch.get("labels") or batch.get("targets")
                if target_ids is None:
                    target_ids = input_ids
            elif isinstance(batch, torch.Tensor):
                input_ids = batch
                target_ids = batch
            elif isinstance(batch, (list, tuple)):
                input_ids = batch[0]
                target_ids = batch[1] if len(batch) > 1 else batch[0]
            else:
                continue

            if input_ids is None or input_ids.numel() == 0:
                continue

            # Get shared embeddings
            with torch.no_grad():
                shared_emb = shared_embedding(input_ids)  # [B, L, base_dim]

                # Run ensemble with logits
                result = self.forward(shared_emb, return_logits=True)

                if "weighted_logits" not in result:
                    continue

                logits = result["weighted_logits"]  # [B, L, vocab]

                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = target_ids[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    ignore_index=-100,
                )

                # 修复：cross_entropy(mean, ignore_index=-100) 只对非忽略 token 求均值，
                # 因此乘以非忽略 token 数（而非总 numel），避免 padding 高估 PPL
                n_valid = (shift_targets != -100).sum().item()
                total_loss += loss.item() * n_valid
                total_tokens += n_valid

            if verbose and (batch_idx + 1) % 10 == 0:
                current_ppl = math.exp(total_loss / max(total_tokens, 1))
                print(f"  Batch {batch_idx + 1}/{max_batches}, PPL: {current_ppl:.2f}")

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(avg_loss)

        return {"ppl": ppl, "loss": avg_loss, "n_tokens": total_tokens}

    @staticmethod
    def evaluate_single_neuron(
        neuron: ResonanceNeuron,
        dataloader,
        shared_embedding: nn.Embedding,
        max_batches: int = 50,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Evaluate PPL for a single neuron (baseline comparison)."""
        total_loss = 0.0
        total_tokens = 0

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            if isinstance(batch, dict):
                input_ids = batch.get("input_ids") or batch.get("tokens")
                target_ids = batch.get("labels") or batch.get("targets")
                if target_ids is None:
                    target_ids = input_ids
            elif isinstance(batch, torch.Tensor):
                input_ids = batch
                target_ids = batch
            elif isinstance(batch, (list, tuple)):
                input_ids = batch[0]
                target_ids = batch[1] if len(batch) > 1 else batch[0]
            else:
                continue

            if input_ids is None or input_ids.numel() == 0:
                continue

            with torch.no_grad():
                shared_emb = shared_embedding(input_ids)
                result = neuron.forward(shared_emb, return_logits=True)
                logits = result["logits"]

                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = target_ids[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    ignore_index=-100,
                )

                n_valid = (shift_targets != -100).sum().item()
                total_loss += loss.item() * n_valid
                total_tokens += n_valid

            if verbose and (batch_idx + 1) % 10 == 0:
                current_ppl = math.exp(total_loss / max(total_tokens, 1))
                print(f"  Batch {batch_idx + 1}/{max_batches}, PPL: {current_ppl:.2f}")

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(avg_loss)

        return {"ppl": ppl, "loss": avg_loss, "n_tokens": total_tokens}
