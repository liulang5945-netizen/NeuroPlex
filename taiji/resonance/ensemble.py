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
        max_rounds: int = 3,  # 协作轮数：2 轮让 side_signals 生效，3 轮充分收敛（>3 收益递减）
        diversity_lambda: float = 0.01,  # 多样性正则化系数：防止神经元退化相同，0.01 为弱约束
        logits_top_k: int = 64,  # 融合时每神经元保留 top-64 logits，降低通信成本
        stdp_tracker: Optional[Any] = None,
        coaction: Optional[Any] = None,
        neuromodulator: Optional[Any] = None,
        maturity: Optional[Any] = None,
        gamma_oscillator: Optional[Any] = None,
        shared_expert_id: Optional[str] = None,
        shared_expert_weight: float = 0.3,  # 共享专家基础权重 0.3，域神经元分配剩余 0.7（借鉴 DeepSeek V3）
        geometry: Optional[Any] = None,  # S7: 外部传入 NeuronGeometry（拓扑构建时已创建）
    ):
        self.neurons = neurons
        self.field = field
        self.max_rounds = max_rounds
        self.diversity_lambda = diversity_lambda

        # ── Shared Expert（借鉴 Kimi K3 / DeepSeek V3）──
        # general 神经元 always-active，提供基础语言能力
        # 融合时获得固定基础权重，域特定神经元分配剩余权重
        self.shared_expert_id = shared_expert_id
        self.shared_expert_weight = shared_expert_weight

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
        # S7: 优先使用外部传入的 geometry（与拓扑构建共享同一实例）
        if geometry is not None:
            self.geometry = geometry
            # 确保新加入的 neuron 有坐标
            self._init_geometry()
        else:
            from .geometry import NeuronGeometry
            self.geometry = NeuronGeometry(embedding_dim=8, sigma=0.5)
            self._init_geometry()

        # ── 大规模内存控制（B2/B3 fix）──
        self.logits_top_k = logits_top_k
        self._logits_keep_ids: Optional[set] = None

        # Tracking
        self.round_scores: List[Dict[str, float]] = []
        self.n_active_history: List[int] = []

        # ── Auxiliary-loss-free balancing ──
        # 每条 side_channel 的利用率统计（EMA），用于启发式 bias 更新
        # 低利用率的 channel 获得正 bias，增强其调制效果
        self._channel_usage: Dict[str, float] = {}  # "post->pre" -> EMA usage score
        self._channel_usage_ema_alpha = 0.99
        self._balancing_update_interval = 50  # 每 50 步更新一次 bias
        self._step_count = 0

        # ── Cross-spec field projector ──
        # 混合规格协作：不同神经元 field_dim 不同时，投影到 field.dim 统一写入
        # 投影层随机初始化（未训练），仅用于维度对齐使 side_channels 能跨规格工作
        # 正式使用需要通过 side_channels 微调训练这些投影层
        self._cross_spec_projectors: Dict[str, nn.Linear] = {}  # forward: field_dim -> unified
        self._cross_spec_back_projectors: Dict[str, nn.Linear] = {}  # backward: unified -> field_dim
        field_dim = self.field.dim
        for nid, neuron in self.neurons.items():
            nfd = neuron.config.field_dim
            if nfd != field_dim:
                proj = nn.Linear(nfd, field_dim, bias=False)
                nn.init.normal_(proj.weight, std=field_dim ** -0.5)
                self._cross_spec_projectors[nid] = proj
                # 反向投影：round 2 conditioning 时将 field.state 投影回 neuron.field_dim
                back_proj = nn.Linear(field_dim, nfd, bias=False)
                nn.init.normal_(back_proj.weight, std=nfd ** -0.5)
                self._cross_spec_back_projectors[nid] = back_proj
        if self._cross_spec_projectors:
            print(f"  [ensemble] 检测到混合 field_dim，创建 {len(self._cross_spec_projectors)} 个跨规格投影层"
                  f"（含反向投影）", flush=True)

    def _project_vec(self, nid: str, vec: torch.Tensor) -> torch.Tensor:
        """Cross-spec projection: 将 neuron 的 field_vector 投影到 field.dim。

        混合规格协作时，不同神经元的 field_dim 不同，需要投影到统一维度才能写入 field。
        """
        if nid in self._cross_spec_projectors:
            return self._cross_spec_projectors[nid](vec)
        return vec

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

    def _update_channel_usage(self, side_signals_per_neuron, round_vecs):
        """更新 side_channel 利用率统计（EMA）。

        每条 channel 的利用率 = proj 输出范数（越大说明信号越强）。
        低利用率的 channel 在 bias 更新时获得正偏置。
        """
        if side_signals_per_neuron is None:
            return
        with torch.no_grad():
            for post_id, signals in side_signals_per_neuron.items():
                post_neuron = self.neurons[post_id]
                for pre_id, sig in signals.items():
                    key = f"{post_id}->{pre_id}"
                    # 计算 proj 范数作为利用率指标
                    if pre_id in post_neuron.excite_channels:
                        proj = post_neuron.excite_channels[pre_id](sig)
                        usage = proj.norm().item()
                    elif pre_id in post_neuron.inhibit_channels:
                        proj = post_neuron.inhibit_channels[pre_id](sig)
                        usage = proj.norm().item()
                    else:
                        continue
                    # EMA 更新
                    if key in self._channel_usage:
                        alpha = self._channel_usage_ema_alpha
                        self._channel_usage[key] = alpha * self._channel_usage[key] + (1 - alpha) * usage
                    else:
                        self._channel_usage[key] = usage

    def _update_channel_biases(self):
        """Auxiliary-loss-free balancing: 启发式更新 channel bias。

        低利用率的 channel 获得正 bias，增强其调制效果。
        高利用率的 channel bias 衰减到 0。
        不通过梯度更新，避免污染主损失。
        """
        if not self._channel_usage:
            return
        # 计算平均利用率
        avg_usage = sum(self._channel_usage.values()) / len(self._channel_usage)
        if avg_usage < 1e-8:
            return

        with torch.no_grad():
            for key, usage in self._channel_usage.items():
                post_id, pre_id = key.split("->")
                post_neuron = self.neurons[post_id]
                # 低利用率 → 正 bias（增强），高利用率 → bias 衰减
                # 阈值依据（Auxiliary-loss-free balancing，借鉴 DeepSeek V3）：
                # - ratio<0.5（使用率<均值一半）→ 增强该通道，防止死通道
                # - ratio>1.5（使用率>均值 1.5 倍）→ 衰减，防止过载
                # - bias_delta 0.1/-0.05：增强幅度大于衰减，偏向"复活死通道"
                ratio = usage / avg_usage  # <1 说明低利用率
                if ratio < 0.5:
                    bias_delta = 0.1 * (1.0 - ratio)
                elif ratio > 1.5:
                    bias_delta = -0.05
                else:
                    bias_delta = 0.0

                # 更新 excite bias
                bias_attr = f"excite_bias_{pre_id}"
                if hasattr(post_neuron, bias_attr):
                    bias_buf = getattr(post_neuron, bias_attr)
                    bias_buf.add_(bias_delta)
                    # 限制 bias 范围 [-1.0, 2.0]
                    bias_buf.clamp_(-1.0, 2.0)

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
        side_signals: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        temp_gain: float = 1.0,
        ffn_gain: float = 1.0,
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
            side_signals: {post_nid: {pre_nid: field_vector}} per-pair 突触信号
            temp_gain: S9 注意力温度增益（norepinephrine 驱动，所有 neuron 共享）
            ffn_gain: S9 FFN 输出增益（dopamine 驱动，所有 neuron 共享）

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
            # Cross-spec back-projection: 将 field.state 投影回 neuron.field_dim
            fs = field_state
            if fs is not None and nid in self._cross_spec_back_projectors:
                fs = self._cross_spec_back_projectors[nid](fs)
            kwargs = dict(
                field_state=fs,
                round_num=round_num,
                return_logits=need_logits,
                temp_gain=temp_gain,
                ffn_gain=ffn_gain,
            )
            if side_signals is not None and nid in side_signals:
                kwargs["side_signals"] = side_signals[nid]
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
        fusion_mode: str = "per_position",
        field_conditioning: bool = True,
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
                        支持字符串模式：'auto_topK'/'auto_all'/'auto_top1'（稀疏激活）
            neuron_embeddings: P8 路径，{nid: [B, L, base_embed_dim]} 预编码 embedding
            mm_logits_modality: 多模态输出模态，不为 None 时所有 neuron 用 mm_lm_head 输出
            fusion_mode: 推理融合模式（方向③ 残差预测编码）
                        - "per_position"（默认）：每位置按熵/置信度独立路由（向后兼容）
                        - "residual"：族长完整预测 + 其他神经元残差修正

        Returns:
            dict with:
            - field_state: final field state vector
            - weighted_logits: 融合后 logits (if return_logits)
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

        # Shared Expert: general 神经元始终激活（不受路由/精简模式影响）
        if self.shared_expert_id and self.shared_expert_id in self.neurons:
            active_ids.add(self.shared_expert_id)

        vectors: Dict[str, torch.Tensor] = {}
        all_logits: Dict[str, torch.Tensor] = {}
        logits_history: List[torch.Tensor] = []

        # S9: 从神经调质计算 Transformer 内部 gain（所有 neuron 共享全局调质水平）
        # - temp_gain: norepinephrine 驱动注意力温度（高 NE → 聚焦）
        # - ffn_gain: dopamine 驱动 FFN 输出强度（高 DA → 强化）
        # 调质为 None 时 gain=1.0（标准 Transformer，向后兼容）
        if self.neuromodulator is not None:
            temp_gain = float(self.neuromodulator.get_attention_temp_gain())
            ffn_gain = float(self.neuromodulator.get_ffn_gain())
        else:
            temp_gain = 1.0
            ffn_gain = 1.0

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
            temp_gain=temp_gain,
            ffn_gain=ffn_gain,
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
            # Cross-spec projection: 将不同 field_dim 的向量投影到 field.dim
            vec = self._project_vec(nid, round_vecs[nid])
            if neuron.is_inhibitory:
                self.field.write_inhibit(nid, vec, weight=maturity_w)
            else:
                self.field.write(nid, vec, scale=write_scale * maturity_w)
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
            scores[nid] = self.field.score(self._project_vec(nid, round_vecs[nid]), neuron_id=nid)
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
                    temp_gain=temp_gain,
                    ffn_gain=ffn_gain,
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

        # ── Side-channel construction (per-pair synaptic projection) ──
        # Round 1 后，每个 post 神经元接收其他 pre 神经元的原始 field_vector，
        # 通过各自的 side_channels 投影到 hidden 空间进行调制。
        side_signals_per_neuron: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        if self.max_rounds >= 2:
            side_signals_per_neuron = {nid: {} for nid in active_ids}
            for post_id in active_ids:
                for pre_id in active_ids:
                    if post_id == pre_id:
                        continue
                    post_neuron = self.neurons[post_id]
                    if (pre_id in post_neuron.excite_channels or
                            pre_id in post_neuron.inhibit_channels):
                        side_signals_per_neuron[post_id][pre_id] = round_vecs[pre_id]

            # Auxiliary-loss-free balancing: 更新 channel 利用率统计
            self._update_channel_usage(side_signals_per_neuron, round_vecs)
            self._step_count += 1
            if self._step_count % self._balancing_update_interval == 0:
                self._update_channel_biases()

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
                field_state=self.field.get_normalised_state() if field_conditioning else None,
                round_num=round_num,
                return_logits_filter=round2_logits_filter,
                neuron_embeddings=neuron_embeddings,
                mm_logits_modality=mm_logits_modality,
                side_signals=side_signals_per_neuron,
                temp_gain=temp_gain,
                ffn_gain=ffn_gain,
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
                vec = self._project_vec(nid, round_vecs[nid])
                if neuron.is_inhibitory:
                    self.field.write_inhibit(nid, vec, weight=maturity_w)
                else:
                    # P1-2: round 2+ 也应用 neuromodulator 调质
                    self.field.update(nid, vec, scale=write_scale * maturity_w)
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
                scores[nid] = self.field.score(self._project_vec(nid, round_vecs[nid]), neuron_id=nid)
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
                    other_vecs = [self._project_vec(o, round_vecs[o]) for o in active_list if o != nid]
                    if not other_vecs:
                        filtered.add(nid)
                        continue
                    congestion = self.field.directional_congestion(
                        self._project_vec(nid, round_vecs[nid]), other_vecs
                    )
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

            # C10: 每轮动态更新 side_signals（原 bug: round 1 后构建一次，rounds 2+ 复用）
            # 让神经元间的信号传递随共振进行而演化，而非停留在 round 1 的快照
            if round_num < self.max_rounds and side_signals_per_neuron is not None:
                side_signals_per_neuron = {nid: {} for nid in active_ids}
                for post_id in active_ids:
                    post_neuron = self.neurons[post_id]
                    for pre_id in active_ids:
                        if post_id == pre_id:
                            continue
                        if (pre_id in post_neuron.excite_channels or
                                pre_id in post_neuron.inhibit_channels):
                            side_signals_per_neuron[post_id][pre_id] = round_vecs[pre_id]

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
            same_vocab = len(set(vocab_sizes)) == 1

            if fusion_mode == "residual" and same_vocab and len(all_logits) >= 2:
                # 方向③：残差预测编码（推理路径）
                # 族长(共振分最高)完整预测 + 其他神经元残差修正
                self._residual_logit_fusion(all_logits, scores, result, ref)
            elif same_vocab:
                self._compute_per_position_weights(
                    all_logits, vectors, scores, result, ref,
                )
            # else: neuron_logits already in result, _generate_p7 handles extraction

            # Shared Expert 重新加权（借鉴 Kimi K3 / DeepSeek V3）
            # general 神经元获得固定基础权重，域特定神经元按原逻辑分配剩余权重
            # final = shared_weight * shared_logits + (1-shared_weight) * original_fused
            if (
                self.shared_expert_id
                and self.shared_expert_id in all_logits
                and "weighted_logits" in result
            ):
                shared_logits = all_logits[self.shared_expert_id]
                original_fused = result["weighted_logits"]
                sw = self.shared_expert_weight
                result["weighted_logits"] = sw * shared_logits + (1.0 - sw) * original_fused
                # 更新 final_weights 反映 Shared Expert 的固定权重
                if "final_weights" in result:
                    weights = result["final_weights"]
                    # 域特定神经元权重缩放到 (1-sw)
                    for nid in weights:
                        if nid != self.shared_expert_id:
                            weights[nid] = weights[nid] * (1.0 - sw)
                    weights[self.shared_expert_id] = sw

        return result

    def forward_train(
        self,
        shared_embeddings: Optional[torch.Tensor] = None,
        neuron_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        n_rounds: int = 2,
        temperature: float = 1.0,
        fusion_mode: str = "soft",
        gamma_oscillator: Optional[Any] = None,
        neuromodulator: Optional[Any] = None,
        return_individual_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """全可微多轮共振训练路径（S1 修复：让共振可端到端训练）。

        与 forward()（推理路径）的核心区别：
        - 全可微：无 hard top-K、无 refractory/active_filter、不使用 field.update/score（含 detach）
        - 多轮共振：round 1 独立前向，round 2+ 注入 side_signals + field_state
          → side_channels 在训练中真正生效，neuron 学习"如何写入场、如何协同"
        - 跨规格投影接入：round_vecs 经正向投影到 unified，field_state 经反向投影回 neuron.field_dim
        - 调质接入：norepinephrine 影响 field_write scale（可微乘法）
        - Gamma 振荡接入：相位推进 + Kuramoto 耦合 + gate_factors 调制 scores
        - 新增 diversity_loss：field_vector 间余弦相似度，防退化相同
        - STDP/Coaction 记录但不影响梯度（局部 Hebbian 规则，反向已处理）

        融合模式：
          - "soft"（默认，全可微）：fused = Σ softmax(score/temp) × logits
          - "residual"（残差预测编码）：族长用 straight-through estimator
            （argmax 在 no_grad 内选 leader，但权重 softmax 可微，梯度流经 other_weights）

        Args:
            shared_embeddings: [B, L, base_embed_dim] 共享嵌入（所有 neuron 共用）
            neuron_embeddings: {nid: [B, L, base_embed_dim]} P7 路径，per-neuron 嵌入
            n_rounds: 共振轮数（默认 2，round 1 独立 + round 2 注入 side_signals）
            temperature: softmax 温度（低=更尖锐选择）
            fusion_mode: "soft"（默认，全可微）/ "residual"（残差预测编码）
            gamma_oscillator: GammaOscillator 实例（None 时回退到 self.gamma_oscillator）
            neuromodulator: NeuromodulatorState 实例（None 时回退到 self.neuromodulator）
            return_individual_logits: 是否返回 individual_logits（节省内存，默认 False）

        Returns:
            dict with:
            - fused_logits: [B, L, V] 融合 logits
            - weights: [N] 或 [N-1] softmax 融合权重
            - scores: [N] 共振分（含 gamma 门控）
            - balance_loss: scalar 负载均衡 loss（负熵，越小越均匀）
            - diversity_loss: scalar field_vector 多样性 loss（越小越多样）
            - field_state: [B, D] 最终场状态（unified 维度）
            - n_rounds: int 实际共振轮数
            - individual_logits: {nid: [B, L, V]}（仅 return_individual_logits=True）
        """
        if shared_embeddings is None and neuron_embeddings is None:
            raise ValueError(
                "[forward_train] 必须提供 shared_embeddings 或 neuron_embeddings"
            )

        active_ids = list(self.neurons.keys())
        N = len(active_ids)

        if gamma_oscillator is None:
            gamma_oscillator = self.gamma_oscillator
        if neuromodulator is None:
            neuromodulator = self.neuromodulator

        # 调质影响（norepinephrine 驱动）：
        # 不直接乘 vec_unified（会被 F.normalize 抵消），
        # 而是影响共振分 scores → 影响融合权重 softmax(scores)
        # 高 norepinephrine → 高 scores → 高融合权重（警觉 → 强贡献）
        write_scale = 1.0
        if neuromodulator is not None:
            try:
                write_scale = float(neuromodulator.get_field_write_scale())
            except Exception:
                write_scale = 1.0

        # S9: 神经调质门控 Transformer 内部计算（注入 attention/FFN，进入梯度流）
        # - temp_gain: norepinephrine → 注意力温度（高 NE → 聚焦，低 NE → 泛化）
        # - ffn_gain: dopamine → FFN 输出强度（高 DA → 强化，低 DA → 衰减）
        # 这让调质从"融合层 scores 缩放器"升级为"Transformer 内部门控"，
        # 真正进入梯度流，成为可学习的生物学一等公民。
        temp_gain = 1.0
        ffn_gain = 1.0
        if neuromodulator is not None:
            try:
                temp_gain = float(neuromodulator.get_attention_temp_gain())
            except Exception:
                temp_gain = 1.0
            try:
                ffn_gain = float(neuromodulator.get_ffn_gain())
            except Exception:
                ffn_gain = 1.0

        def _get_emb(nid: str) -> torch.Tensor:
            if neuron_embeddings is not None and nid in neuron_embeddings:
                return neuron_embeddings[nid]
            return shared_embeddings

        # 重置 STDP firing history（一次训练步内的发放时序）
        if self.stdp_tracker is not None:
            try:
                self.stdp_tracker._firing_history.clear()
            except Exception:
                pass

        # ── 多轮可微前向 ──
        # 不使用 self.field（含 detach 副作用），直接维护 field_state tensor
        # 维护两套 vecs：
        #   - round_vecs_raw: 原始 neuron.field_dim 维度（用于 side_signals，因为
        #     excite/inhibit_channels 在建立时按 pre neuron.field_dim 注册）
        #   - round_vecs_unified: 投影到 unified 维度（用于维护 field_state）
        field_state: Optional[torch.Tensor] = None  # round 1 时为 None
        round_vecs_raw: Dict[str, torch.Tensor] = {}      # 原始 field_dim 维度
        round_vecs_unified: Dict[str, torch.Tensor] = {}  # unified 维度
        final_logits: Dict[str, torch.Tensor] = {}        # 最后一轮每个 neuron 的 logits

        for round_num in range(1, n_rounds + 1):
            round_vecs_raw_new: Dict[str, torch.Tensor] = {}
            round_vecs_unified_new: Dict[str, torch.Tensor] = {}

            # 构建 side_signals（round 2+ 才有，per-pair synaptic 投影）
            # 用原始 field_dim 维度的 vecs（excite_channels 在建立时按 pre.field_dim 注册）
            side_signals_per_neuron: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
            if round_num > 1 and round_vecs_raw:
                side_signals_per_neuron = {nid: {} for nid in active_ids}
                for post_id in active_ids:
                    post_neuron = self.neurons[post_id]
                    for pre_id in active_ids:
                        if post_id == pre_id:
                            continue
                        if (pre_id in post_neuron.excite_channels or
                                pre_id in post_neuron.inhibit_channels):
                            side_signals_per_neuron[post_id][pre_id] = round_vecs_raw[pre_id]

            # 全可微前向（串行：batch 内已并行，neuron.forward 全可微）
            for nid in active_ids:
                emb = _get_emb(nid)

                # 跨规格反投影：field_state(unified) → neuron.field_dim
                fs = field_state
                if fs is not None and nid in self._cross_spec_back_projectors:
                    fs = self._cross_spec_back_projectors[nid](fs)

                kwargs: Dict[str, Any] = dict(
                    field_state=fs,
                    round_num=round_num,
                    return_logits=True,
                    temp_gain=temp_gain,
                    ffn_gain=ffn_gain,
                )
                if side_signals_per_neuron is not None:
                    kwargs["side_signals"] = side_signals_per_neuron[nid]

                result = self.neurons[nid].forward(emb, **kwargs)

                # 原始 field_vector（neuron.field_dim，用于 side_signals）
                vec_raw = result["field_vector"]  # [B, neuron.field_dim]
                # 跨规格正向投影：neuron.field_dim → unified（用于维护 field_state）
                vec_unified = self._project_vec(nid, vec_raw)  # [B, field.dim]
                # 注：write_scale 不在此处乘（F.normalize 会抵消），
                # 而是在 scores 计算后乘，影响融合权重

                round_vecs_raw_new[nid] = vec_raw
                round_vecs_unified_new[nid] = vec_unified
                if round_num == n_rounds:
                    final_logits[nid] = result["logits"]

            # 更新 field_state（可微加法，无 detach）
            # field_state = sum of L2-normalized vecs (unified 维度)
            all_vecs_norm = F.normalize(
                torch.stack([round_vecs_unified_new[nid] for nid in active_ids]),
                dim=-1,
            )  # [N, B, D]
            field_state = all_vecs_norm.sum(dim=0)  # [B, D]

            round_vecs_raw = round_vecs_raw_new
            round_vecs_unified = round_vecs_unified_new

            # STDP 记录（不影响梯度，本地 Hebbian 规则；用 unified 维度保持一致）
            if self.stdp_tracker is not None:
                for nid in active_ids:
                    try:
                        self.stdp_tracker.record_firing(nid, round_num, round_vecs_unified[nid])
                    except Exception:
                        pass

            # Coactivation 记录（不影响梯度）
            if self.coaction is not None:
                try:
                    self.coaction.update(active_ids, round_num=round_num)
                except Exception:
                    pass

            # Gamma 振荡：推进相位 + Kuramoto 耦合
            if gamma_oscillator is not None:
                try:
                    gamma_oscillator.tick()
                    if hasattr(gamma_oscillator, "kuramoto_step"):
                        gamma_oscillator.kuramoto_step(
                            coupling_strength=0.05,
                            active_ids=active_ids,
                            coactivation=self.coaction,
                        )
                except Exception:
                    pass  # 振荡失败不影响训练主流程

        # ── 计算共振分（Leave-one-out cosine similarity，全可微）──
        # 用 unified 维度的 vecs（保证不同 field_dim 的 neuron 在同一空间比较）
        all_vecs_norm = F.normalize(
            torch.stack([round_vecs_unified[nid] for nid in active_ids]),
            dim=-1,
        )  # [N, B, D]
        field_state_full = all_vecs_norm.sum(dim=0)  # [B, D]
        loo_state = field_state_full.unsqueeze(0) - all_vecs_norm  # [N, B, D]
        loo_norm = F.normalize(loo_state, dim=-1)
        scores = (all_vecs_norm * loo_norm).sum(dim=-1)  # [N, B]
        scores = scores.mean(dim=1)                      # [N] batch 平均

        # 调质影响：norepinephrine 高 → scores 增强 → 融合权重增大（警觉 → 强贡献）
        # write_scale 是 Python float（不可微，但调质本身是外部状态，非可学习参数）
        if neuromodulator is not None and write_scale != 1.0:
            scores = scores * write_scale

        # Gamma 门控：相位对齐的神经元获得更高权重（feature binding）
        if gamma_oscillator is not None:
            try:
                gate_factors = gamma_oscillator.batch_gate_factors(active_ids)  # [N]
                scores = scores * gate_factors.to(scores.device)
            except Exception:
                pass

        # ── 融合聚合 ──
        # 检查 vocab 大小一致（跨 vocab 联合训练路径未实现，缺口 M）
        vocab_sizes = [final_logits[nid].shape[-1] for nid in active_ids]
        if len(set(vocab_sizes)) != 1:
            raise RuntimeError(
                f"[forward_train] 要求所有 neuron vocab 大小一致（跨 vocab 联合训练"
                f"路径未实现，见缺口 M）。当前 vocab_sizes: "
                f"{dict(zip(active_ids, vocab_sizes))}"
            )

        all_logits = torch.stack([final_logits[nid] for nid in active_ids])  # [N, B, L, V]

        if fusion_mode == "residual" and N >= 2:
            # 残差预测编码：straight-through estimator（选择不可微，权重可微）
            with torch.no_grad():
                leader_idx = int(scores.argmax().item())
            leader_logits = all_logits[leader_idx]  # [B, L, V]
            other_indices = [i for i in range(N) if i != leader_idx]
            other_scores = scores[other_indices]
            other_logits = all_logits[other_indices]
            other_weights = F.softmax(other_scores / temperature, dim=0)  # [N-1]
            residual = torch.einsum('n,nblv->blv', other_weights, other_logits)
            fused_logits = leader_logits + residual
            balance_loss = -(other_weights * torch.log(other_weights + 1e-8)).sum()
            weights = other_weights
        else:
            # soft 软加权融合（默认，全可微）
            weights = F.softmax(scores / temperature, dim=0)  # [N]
            fused_logits = torch.einsum('n,nblv->blv', weights, all_logits)
            balance_loss = -(weights * torch.log(weights + 1e-8)).sum()

        # ── 多样性 loss（field_vector 间余弦相似度，防退化相同）──
        if N >= 2:
            # all_vecs_norm: [N, B, D] → batch 平均后 [N, D]
            vecs_batch = all_vecs_norm.mean(dim=1)  # [N, D]
            vecs_norm = F.normalize(vecs_batch, dim=-1)
            sim_matrix = torch.einsum('nd,md->nm', vecs_norm, vecs_norm)  # [N, N]
            mask = torch.triu(
                torch.ones(N, N, device=sim_matrix.device, dtype=torch.bool),
                diagonal=1,
            )
            diversity_loss = sim_matrix[mask].mean()  # 越小越好
        else:
            diversity_loss = torch.tensor(0.0, device=weights.device)

        result: Dict[str, torch.Tensor] = {
            "fused_logits": fused_logits,
            "weights": weights,
            "scores": scores,
            "balance_loss": balance_loss,
            "diversity_loss": diversity_loss,
            "field_state": field_state_full,
            "n_rounds": n_rounds,
        }

        if return_individual_logits:
            result["individual_logits"] = {nid: final_logits[nid] for nid in active_ids}

        return result

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

        # H5 (disabled 2026-07-28): resonance score boost removed for independent-embedding
        # neurons. field.score() compares field vectors across different embedding spaces,
        # causing score inversion (worst-PPL neuron gets highest score). Per-position routing
        # now relies purely on logits quality (entropy + prediction_complementarity below),
        # which is embedding-space-agnostic and aligns with project_memory constraint:
        # "Ensemble collaboration must use logits fusion instead of field space residual".

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

    def _residual_logit_fusion(
        self,
        all_logits: Dict[str, torch.Tensor],
        scores: Dict[str, float],
        result: dict,
        ref: torch.Tensor,
        temperature: float = 1.0,
    ) -> None:
        """方向③：残差预测编码（推理路径）。

        族长(共振分最高)给出完整预测 logits，
        其他神经元预测族长的残差（纠正族长预测错误的部分），
        最终 fused = leader_logits + Σ(w_i × other_logits_i)。

        与训练路径 forward_train(fusion_mode='residual') 对称：
        - 训练时族长获完整梯度（快速成强）
        - 推理时族长给完整预测（能力最强），其他做残差修正

        与 _compute_per_position_weights 的区别：
        - per_position: 每位置独立选最自信神经元（熵路由）
        - residual: 族长全局主导 + 其他全局修正（层级预测）
        residual 更符合"族长带领"的人脑启发结构，且与训练路径一致。

        Args:
            all_logits: {nid: [B, L, V]} 所有激活神经元的 logits（同 vocab）
            scores: {nid: float} 最终共振分
            result: forward() 的 result dict，写入 weighted_logits 和 final_weights
            ref: 参考 tensor（device 信息）
            temperature: softmax 温度（低=更尖锐选择）
        """
        neuron_ids = list(all_logits.keys())
        n_neurons = len(neuron_ids)
        if n_neurons < 2:
            # 单神经元退化：直接用它的 logits
            result["weighted_logits"] = all_logits[neuron_ids[0]]
            result["final_weights"] = {neuron_ids[0]: 1.0}
            return

        # 1. 选族长（共振分最高）
        leader_nid = max(neuron_ids, key=lambda n: scores.get(n, 0.0))
        leader_logits = all_logits[leader_nid]  # [B, L, V]

        # 2. 其他神经元权重（softmax，族长不参与权重分配）
        other_nids = [n for n in neuron_ids if n != leader_nid]
        other_scores = torch.tensor(
            [float(scores.get(n, 0.0)) for n in other_nids],
            device=ref.device,
        )
        weights = F.softmax(other_scores / temperature, dim=0)  # [N-1]

        # 3. 残差聚合：fused = 族长完整预测 + Σ(w_i × 其他神经元修正)
        fused_logits = leader_logits.clone()
        for i, nid in enumerate(other_nids):
            fused_logits = fused_logits + weights[i] * all_logits[nid]

        result["weighted_logits"] = fused_logits
        # final_weights: 族长标记为 1.0（完整预测），其他按 softmax 权重
        result["final_weights"] = {leader_nid: 1.0}
        for i, nid in enumerate(other_nids):
            result["final_weights"][nid] = float(weights[i].item())
        result["leader_nid"] = leader_nid

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
