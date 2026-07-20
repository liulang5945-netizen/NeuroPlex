"""Resonance ensemble — multi-round collaborative inference with gating.

The ensemble orchestrates multiple ResonanceNeurons through the
ResonanceField over 3-5 rounds of collaborative inference.

Each round:
1. All active neurons run forward (first round: independently;
   subsequent rounds: conditioned on field state)
2. Each neuron writes its L2-normalised field vector
3. Resonance scores are computed (cosine similarity with field state)
4. Low-resonance neurons are filtered out via dynamic thresholding

Experiment 12 revealed three critical mechanisms (now integrated):
- ConfidenceGate: skip resonance when prediction is already confident
- EarlyStopResonance: stop iterating when logits converge
- ResonanceTrigger: full trigger conditions for activation
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
from .gating import ConfidenceGate, EarlyStopResonance, ResonanceTrigger
from .quality import QualityFilter
from .division import DivisionPath


class ResonanceEnsemble:
    """Orchestrates multi-round resonance inference across multiple neurons.

    v2: Integrated three gating mechanisms from Experiment 12:
    - ConfidenceGate: skip resonance for confident predictions
    - EarlyStopResonance: stop when logits converge
    - ResonanceTrigger: combined trigger conditions

    Usage:
        # Without gating (backward compatible):
        ensemble = ResonanceEnsemble(neurons, field)
        result = ensemble.forward(shared_embeddings)

        # With gating:
        ensemble = ResonanceEnsemble(
            neurons, field,
            confidence_gate=ConfidenceGate(threshold=0.9),
            early_stop=EarlyStopResonance(),
        )
        result = ensemble.forward(shared_embeddings, return_logits=True)
    """

    def __init__(
        self,
        neurons: Dict[str, ResonanceNeuron],
        field: ResonanceField,
        max_rounds: int = 3,
        diversity_lambda: float = 0.01,
        confidence_gate: Optional[ConfidenceGate] = None,
        early_stop: Optional[EarlyStopResonance] = None,
        resonance_trigger: Optional[ResonanceTrigger] = None,
        quality_filter: Optional[QualityFilter] = None,
        division_path: Optional[DivisionPath] = None,
        logits_top_k: int = 16,
        domain_router: Optional["DomainRouter"] = None,
        stdp_tracker: Optional[Any] = None,
        coaction: Optional[Any] = None,
        neuromodulator: Optional[Any] = None,
    ):
        self.neurons = neurons
        self.field = field
        self.max_rounds = max_rounds
        self.diversity_lambda = diversity_lambda

        # ── Gating mechanisms (Experiment 12) ──
        self.confidence_gate = confidence_gate
        self.early_stop = early_stop
        self.resonance_trigger = resonance_trigger

        # ── Quality filtering (Experiment 9) ──
        self.quality_filter = quality_filter

        # ── Division-of-labor path ──
        self.division_path = division_path

        # ── Domain-aware routing (P2-2: field_vector × anchor) ──
        self.domain_router = domain_router

        # ── Bio-inspired trackers (P1 接线) ──
        # STDPTracker: 推理期记录发放时序，sleep 期 apply_updates 强化 side_channels
        self.stdp_tracker = stdp_tracker
        # CoactivationTracker: 推理期更新共激活，驱动动态部落化
        self.coaction = coaction
        # NeuromodulatorState: 多巴胺/血清素调整 lr / refractory / field_write_scale
        self.neuromodulator = neuromodulator

        # ── 大规模内存控制（B2/B3 fix）──
        self.logits_top_k = logits_top_k
        self._logits_keep_ids: Optional[set] = None

        # Tracking
        self.round_scores: List[Dict[str, float]] = []
        self.n_active_history: List[int] = []

    def _parallel_forward(
        self,
        active_ids,
        shared_embeddings: torch.Tensor,
        field_state,
        round_num: int,
        return_logits_filter,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """并行 forward 多个神经元（人脑启发：神经元并行工作）。

        GPU 模式下用 CUDA stream 真并行，保留 per-neuron 独立性。
        CPU 模式下退化为串行（无 stream 开销）。

        Args:
            active_ids: 要 forward 的 neuron id 集合
            shared_embeddings: 共享嵌入
            field_state: 场状态（round 1 为 None）
            round_num: 轮次
            return_logits_filter: callable(nid) -> bool，决定哪些 neuron 返回 logits

        Returns:
            (round_vecs, round_logits)
        """
        round_vecs: Dict[str, torch.Tensor] = {}
        round_logits: Dict[str, torch.Tensor] = {}

        is_cuda = shared_embeddings.is_cuda
        if is_cuda and len(active_ids) > 1:
            # GPU 模式：CUDA stream 真并行
            # 每个 neuron 在自己的 stream 上独立 forward
            streams = {nid: torch.cuda.Stream() for nid in active_ids}
            results: Dict[str, Dict] = {}

            for nid in active_ids:
                neuron = self.neurons[nid]
                need_logits = return_logits_filter(nid)
                with torch.cuda.stream(streams[nid]):
                    results[nid] = neuron.forward(
                        shared_embeddings,
                        field_state=field_state,
                        round_num=round_num,
                        return_logits=need_logits,
                    )

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
                neuron = self.neurons[nid]
                need_logits = return_logits_filter(nid)
                result = neuron.forward(
                    shared_embeddings,
                    field_state=field_state,
                    round_num=round_num,
                    return_logits=need_logits,
                )
                round_vecs[nid] = result["field_vector"]
                if need_logits:
                    round_logits[nid] = result["logits"]

        return round_vecs, round_logits

    def forward(
        self,
        shared_embeddings: torch.Tensor,
        return_logits: bool = False,
        active_filter: bool = True,
        enable_gating: bool = True,
        active_nids: Optional[List[str]] = None,
    ) -> Dict:
        """Run the full resonance loop with optional gating.

        Args:
            shared_embeddings: [B, L, base_embed_dim]
            return_logits: if True, each neuron also returns token logits
            active_filter: if True, filter out low-resonance neurons each round
            enable_gating: if False, skip all gating (backward compatible mode)
            active_nids: 如果提供，只 forward 这些 neuron（Phase 5.1 丘脑路由用）
                         None 表示全部参与（默认行为，向后兼容）

        Returns:
            dict with:
            - field_state: final field state vector
            - weighted_logits: resonance-weighted average logits (if return_logits)
            - final_scores: per-neuron resonance scores (final round)
            - n_rounds: actual number of rounds completed
            - skipped_resonance: True if gating skipped the resonance loop
            - skip_reason: explanation if resonance was skipped
        """
        self.field.reset(batch_size=shared_embeddings.shape[0])
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
        # Phase 5.1: 如果 ThalamicRouter 指定了 active_nids，只激活这些 neuron
        if active_nids is not None:
            active_ids = set(nid for nid in active_nids if nid in self.neurons)
            if not active_ids:
                # fallback: 全部 neuron
                active_ids = set(neuron_ids)
        else:
            active_ids = set(neuron_ids)

        # ── Quality filter: exclude weak neurons before resonance ──
        if self.quality_filter is not None:
            filtered_ids = self.quality_filter.filter(list(active_ids))
            if not filtered_ids:
                raise ValueError("All neurons filtered out by quality filter")
            active_ids = set(filtered_ids)

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
        )

        # Write round 1 to field
        # P1-2: 从 NeuromodulatorState 读取 field_write_scale（去甲肾上腺素驱动）
        write_scale = (self.neuromodulator.get_field_write_scale()
                       if self.neuromodulator is not None else 1.0)
        for nid in active_ids:
            self.field.write(nid, round_vecs[nid], scale=write_scale)
            # P1-STDP: 记录 round 1 发放（用于 sleep 期 STDP 强化）
            if self.stdp_tracker is not None:
                self.stdp_tracker.record_firing(nid, 1, round_vecs[nid])
            # P1-Coactivation: 记录共激活（同轮 forward 的 neuron 互为共激活）
            if self.coaction is not None:
                self.coaction.update(active_ids, round_num=1)

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
                best_result = self.neurons[best_nid].forward(
                    shared_embeddings,
                    field_state=None,
                    round_num=1,
                    return_logits=True,
                )
                round_logits[best_nid] = best_result["logits"]
            else:
                # 小规模：round 1 已获取所有 logits，丢弃非 top-K
                if len(round_logits) > self.logits_top_k:
                    non_keep = set(round_logits.keys()) - self._logits_keep_ids
                    for nid in non_keep:
                        del round_logits[nid]

        # ── Gating check: should we resonate? ──
        if enable_gating and return_logits:
            if self.resonance_trigger is not None:
                best_nid = max(scores, key=scores.get)
                should_res, reason = self.resonance_trigger.should_resonate(
                    round_logits[best_nid], round_vecs
                )
                if not should_res:
                    self.n_active_history.append(len(active_ids))
                    return {
                        "field_state": self.field.get_state(),
                        "weighted_logits": round_logits[best_nid],
                        "final_scores": scores,
                        "n_rounds": 1,
                        "n_active_history": self.n_active_history,
                        "skipped_resonance": True,
                        "skip_reason": reason,
                    }
            elif self.confidence_gate is not None:
                best_nid = max(scores, key=scores.get)
                if not self.confidence_gate.should_resonate(round_logits[best_nid]):
                    confidence = self.confidence_gate.get_confidence(round_logits[best_nid])
                    self.n_active_history.append(len(active_ids))
                    return {
                        "field_state": self.field.get_state(),
                        "weighted_logits": round_logits[best_nid],
                        "final_scores": scores,
                        "n_rounds": 1,
                        "n_active_history": self.n_active_history,
                        "skipped_resonance": True,
                        "skip_reason": f"confident prediction (max_prob={confidence:.3f})",
                    }

        # Track round 1 for early stop
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
                # P1-2: round 2+ 也应用 neuromodulator 调质
                self.field.update(nid, round_vecs[nid], scale=write_scale)
                self.neurons[nid].enter_refractory(multiplier=refractory_mult)
                # P1-STDP: 记录 round 2+ 发放
                if self.stdp_tracker is not None:
                    self.stdp_tracker.record_firing(nid, round_num, round_vecs[nid])
            # P1-Coactivation: 更新共激活
            if self.coaction is not None and writable_ids:
                self.coaction.update(writable_ids, round_num=round_num)

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

            self.n_active_history.append(len(active_ids))
            vectors = round_vecs
            all_logits = round_logits

            # ── Early stop check ──
            if enable_gating and self.early_stop is not None and return_logits:
                logits_history.append(self._average_logits(all_logits))
                if self.early_stop.should_stop(logits_history):
                    break

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
            final_scores = self.round_scores[-1] if self.round_scores else scores

            if self.domain_router is not None:
                # P2-2: Domain-aware routing
                # 用最后一轮的 field_vector 与各 neuron 的 domain anchor 计算相似度，
                # 相似度越高 -> 输入越属于该 neuron 的本域 -> 给更高权重
                # 取代 entropy-based weighting（neuron 在非本域也强行自信会误导 entropy 路由）
                last_vecs = {nid: vectors.get(nid, torch.zeros(1)) for nid in all_logits}
                final_weights_dict = self.domain_router.route(last_vecs)

                weight_list = [final_weights_dict.get(nid, 0.0) for nid in all_logits.keys()]
                weights = torch.tensor(weight_list, device=shared_embeddings.device)

                weighted_logits = None
                for i, (nid, logits) in enumerate(all_logits.items()):
                    w = weights[i]
                    if weighted_logits is None:
                        weighted_logits = w * logits
                    else:
                        weighted_logits = weighted_logits + w * logits
                result["weighted_logits"] = weighted_logits
                result["final_weights"] = {
                    nid: float(weights[i].item())
                    for i, nid in enumerate(all_logits.keys())
                }
            elif self.division_path is not None:
                # Use division-of-labor weighting (scale layering + cluster dominance)
                # Build clusters from neuron domains (simple: one cluster per neuron for now)
                clusters = {"default": {nid: vectors.get(nid, torch.zeros(1)) for nid in all_logits}}
                neuron_specs = {
                    nid: getattr(self.neurons[nid].config, "spec", "compact")
                    for nid in all_logits
                }
                # Use the first neuron's field vector as input_vector proxy
                input_vec = next(iter(vectors.values()), torch.zeros(1))
                final_weights_dict = self.division_path.compute_final_weights(
                    input_vector=input_vec.mean(dim=0) if input_vec.dim() > 1 else input_vec,
                    clusters=clusters,
                    neuron_specs=neuron_specs,
                    resonance_scores=final_scores,
                )
                weight_list = [final_weights_dict.get(nid, 0.0) for nid in all_logits.keys()]
                weights = torch.tensor(weight_list, device=shared_embeddings.device)

                # 修复：应用 division_path 权重生成 weighted_logits
                weighted_logits = None
                for i, (nid, logits) in enumerate(all_logits.items()):
                    w = weights[i]
                    if weighted_logits is None:
                        weighted_logits = w * logits
                    else:
                        weighted_logits = weighted_logits + w * logits
                result["weighted_logits"] = weighted_logits
                result["final_weights"] = {
                    nid: float(weights[i].item())
                    for i, nid in enumerate(all_logits.keys())
                }
            else:
                # Per-position routing (v2): logit-entropy weighting + complementarity.
                # Each position independently picks the neuron that is most confident.
                # Complementarity scores boost neurons bringing new information.
                # Memory-efficient: process one neuron at a time for entropy.
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
                score_vals = torch.tensor(
                    [float(final_scores.get(nid, 0.0)) for nid in neuron_ids],
                    device=shared_embeddings.device,
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
                    comp_boost = torch.tensor(comp_vals, device=shared_embeddings.device)
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

        return result

    def _average_logits(
        self, logits_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute simple average of logits across neurons for early stop."""
        if not logits_dict:
            return torch.zeros(1)
        stacked = torch.stack(list(logits_dict.values()))
        return stacked.mean(dim=0)

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
