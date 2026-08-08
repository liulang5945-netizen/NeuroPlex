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
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .field import ResonanceField
from .neuron import ResonanceNeuron
from .spatial_diffusion import SpatialDiffuser
from .translator import build_logits_alignment_matrix


class CrossSpecProjector(nn.Module):
    """T6: 跨规格投影 MLP（Linear + GELU + Linear, 残差 + 零初始化第二层）。

    替代旧的单层 nn.Linear，提供非线性投影能力。

    兼容性设计（零破坏升级）：
    - linear1: Linear(in_dim, out_dim, bias=False)，权重初始化与旧 nn.Linear 一致
    - linear2: Linear(out_dim, out_dim, bias=False)，权重零初始化
    - forward: y = linear1(x) + linear2(gelu(linear1(x)))
    - 初始时 linear2 输出 = 0，故 y = linear1(x)（与旧单层 Linear 完全一致）
    - 训练后 linear2 学到非零权重，引入非线性变换能力

    旧 checkpoint 兼容加载：
    - 旧格式 state_dict: {"weight": tensor}  → 映射到 linear1.weight, linear2 保持零初始化
    - 新格式 state_dict: {"linear1.weight": ..., "linear2.weight": ...}  → 直接加载
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.normal_(self.linear1.weight, std=out_dim ** -0.5)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        nn.init.zeros_(self.linear2.weight)  # 零初始化：初始时 linear2 输出=0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear1(x)
        return h + self.linear2(self.gelu(h))

    def load_legacy_linear_state(self, legacy_weight: torch.Tensor) -> None:
        """从旧单层 Linear 的 state_dict 兼容加载。

        旧 nn.Linear 的 weight 形状为 [out_dim, in_dim]，直接赋值给 linear1.weight。
        linear2 保持零初始化（初始行为与旧 Linear 一致）。
        """
        assert self.linear1.weight.shape == legacy_weight.shape, (
            f"legacy weight shape {legacy_weight.shape} != linear1.weight shape {self.linear1.weight.shape}"
        )
        with torch.no_grad():
            self.linear1.weight.copy_(legacy_weight)
            # linear2 保持零初始化（构造时已设置）


class SparseRouter(nn.Module):
    """§4.0c: Probe-based Sparse Router（基于探针的稀疏路由器）。

    基于 round 1 probe（每神经元独立前向）的响应，为每个神经元产生路由分，
    选择 top-K 神经元参与 round 2+ 的深度协作。

    核心思路：
    - round 1 所有神经元独立前向（已在 forward_train 中执行，零额外开销）
    - Router 基于 round 1 的 field_vector + confidence + score_vec 评分
    - top-K 选择（forward: hard mask, backward: STE soft softmax）

    向后兼容：
    - use_sparse_router=False（默认）时完全不启用，退化为稠密模式
    - Router 参数只在 use_sparse_router=True 时创建

    详见 plans/ARCHITECTURE_COMPROMISE_AUDIT.md §4.0c
    """

    def __init__(
        self,
        field_dim: int,
        score_dim: Optional[int],
        hidden_dim: int = 128,
        top_k: int = 3,
        warmup_steps: int = 0,  # Phase 0/1 warm-up 步数
        shared_expert_id: Optional[str] = None,  # shared_expert 始终激活，不参与 top-K
        # ── §4.0d: 熵驱动动态 K ──
        dynamic_k: bool = True,  # True=熵驱动动态K（简单样本少选，复杂样本多选）
        k_min: int = 1,          # 动态 K 下限
        k_max: Optional[int] = None,  # 动态 K 上限（None = N-1）
    ):
        super().__init__()
        self.field_dim = field_dim
        self.score_dim = score_dim
        self.top_k = top_k
        self.warmup_steps = warmup_steps
        self.shared_expert_id = shared_expert_id
        # §4.0d: 动态 K 配置
        self.dynamic_k = dynamic_k
        self.k_min = max(1, k_min)
        self.k_max = k_max

        # per-neuron 特征维度：field_vector + confidence(1) + score_vec(if exists)
        in_dim = field_dim + 1 + (score_dim if score_dim is not None else 0)
        self.router_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # 初始化：最后一层小权重，初始路由分接近均匀
        nn.init.normal_(self.router_mlp[0].weight, std=in_dim ** -0.5)
        nn.init.zeros_(self.router_mlp[0].bias)
        nn.init.normal_(self.router_mlp[2].weight, std=hidden_dim ** -0.5)
        nn.init.zeros_(self.router_mlp[2].bias)

    def forward(
        self,
        active_ids: List[str],
        round_vecs_unified: Dict[str, torch.Tensor],  # {nid: [B, D_field]}
        round_confidences: Dict[str, torch.Tensor],   # {nid: [B]}
        round_score_vecs: Optional[Dict[str, torch.Tensor]],  # {nid: [B, D_score]} or None
        step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """计算路由分和 per-sample top-K 选择（§4.0c + §4.0d）。

        Args:
            active_ids: 当前激活的神经元 id 列表
            round_vecs_unified: round 1 每神经元的场向量（unified 维度）
            round_confidences: round 1 每神经元的置信度
            round_score_vecs: round 1 每神经元的评分向量（None 时不用）
            step: 当前训练步数（用于 warm-up）

        Returns:
            dict with:
            - routing_scores: [B, N] 路由分
            - hard_mask: [B, N] per-sample hard top-K 选择（forward）
            - soft_weights: [B, N] soft softmax 权重（backward 梯度流）
            - final_weights: [B, N] STE 融合权重（forward=hard, backward=soft）
            - load_balance_loss: scalar 负载均衡 loss（Switch 风格）
            - k_per_sample: [B] 每样本实际 K 值（熵驱动，含 shared_expert）
            - top_k_ids: List[List[int]] 每样本 top-K 的全局索引
        """
        N = len(active_ids)
        device = next(iter(round_vecs_unified.values())).device
        B = round_vecs_unified[active_ids[0]].shape[0]

        # 构建 per-neuron 特征 [N, B, in_dim]
        feats = []
        for nid in active_ids:
            vec = round_vecs_unified[nid]  # [B, D_field]
            conf = round_confidences[nid].unsqueeze(-1)  # [B, 1]
            parts = [vec, conf]
            if self.score_dim is not None and round_score_vecs is not None:
                parts.append(round_score_vecs[nid])  # [B, D_score]
            feats.append(torch.cat(parts, dim=-1))  # [B, in_dim]
        feat_stack = torch.stack(feats)  # [N, B, in_dim]

        # Router 评分：per-neuron MLP -> [N, B, 1] -> [N, B] -> [B, N]
        routing_scores = self.router_mlp(feat_stack).squeeze(-1)  # [N, B]
        routing_scores = routing_scores.t()  # [B, N]

        # soft weights（backward 梯度流）
        soft_weights = F.softmax(routing_scores, dim=-1)  # [B, N]

        # ── Warm-up 阶段决定 K ──
        # Phase 0 (step < warmup): K=N（全选），Router 只学习评分
        # Phase 1 (warmup <= step < 3*warmup): K 线性从 N 降到 top_k
        # Phase 2 (step >= 3*warmup): 熵驱动动态 K（dynamic_k=True）或固定 top_k
        if step < self.warmup_steps:
            base_k = N
            use_dynamic = False
        elif step < self.warmup_steps * 3:
            progress = (step - self.warmup_steps) / max(1, self.warmup_steps * 2)
            base_k = max(self.top_k, int(N - progress * (N - self.top_k)))
            use_dynamic = False
        else:
            base_k = self.top_k
            use_dynamic = self.dynamic_k

        # §4.0d: 熵驱动动态 K（每样本独立）
        # 低熵（Router 99% 确定某神经元）→ K 小（省算力）
        # 高熵（Router 犹豫）→ K 大（保能力）
        if use_dynamic:
            entropy = -(soft_weights * torch.log(soft_weights + 1e-8)).sum(-1)  # [B]
            logN = math.log(max(N, 2))
            norm_entropy = (entropy / logN).clamp(0.0, 1.0)  # [B] 0~1
            k_max_eff = self.k_max if self.k_max is not None else max(base_k, N - 1)
            k_range = max(k_max_eff - self.k_min, 1)
            k_per_sample = self.k_min + (norm_entropy * k_range).round().long()  # [B]
            k_per_sample = k_per_sample.clamp(self.k_min, N)
        else:
            k_per_sample = torch.full((B,), base_k, device=device, dtype=torch.long)

        # ── per-sample top-K（含 shared_expert 始终激活）──
        hard_mask = torch.zeros_like(routing_scores)  # [B, N]
        top_k_ids: List[List[int]] = []
        if self.shared_expert_id is not None and self.shared_expert_id in active_ids:
            shared_idx = active_ids.index(self.shared_expert_id)
            hard_mask[:, shared_idx] = 1.0  # shared 始终激活
            for b in range(B):
                k_b = max(int(k_per_sample[b]) - 1, 1)  # 扣掉 shared 占位
                k_b = min(k_b, N - 1)
                domain_scores = routing_scores[b].clone()
                domain_scores[shared_idx] = float('-inf')  # 排除 shared
                topk = domain_scores.topk(k_b).indices  # [k_b]
                hard_mask[b, topk] = 1.0
                # 更新 k_per_sample 反映实际（shared + domain）
                k_per_sample[b] = k_b + 1
                top_k_ids.append(topk.tolist() + [shared_idx])
        else:
            for b in range(B):
                k_b = max(int(k_per_sample[b]), 1)
                k_b = min(k_b, N)
                topk = routing_scores[b].topk(k_b).indices  # [k_b]
                hard_mask[b, topk] = 1.0
                k_per_sample[b] = k_b
                top_k_ids.append(topk.tolist())

        # selected weights: 只在 hard_mask 选中神经元上归一化
        masked_scores = routing_scores.masked_fill(hard_mask == 0, float('-inf'))
        selected_weights = F.softmax(masked_scores, dim=-1)  # [B, N]
        # 处理全 -inf 行（不应发生，但防御）
        selected_weights = torch.nan_to_num(selected_weights, nan=0.0)

        # STE: forward=selected_weights(hard), backward=grad(soft_weights)
        final_weights = soft_weights + (selected_weights - soft_weights).detach()

        # 负载均衡 loss（Switch Transformer 风格）
        # f_i: 神经元 i 被选中的批次比例（hard, detach）
        f = hard_mask.mean(dim=0).detach()  # [N]
        # P_i: Router 对神经元 i 的平均概率（soft, 可微——提供梯度信号）
        # 注意：P 不 detach，否则负载均衡 loss 对 Router 无梯度（修复 §4.0d）
        P = soft_weights.mean(dim=0)  # [N]
        load_balance_loss = N * (f * P).sum()

        return {
            "routing_scores": routing_scores,    # [B, N]
            "hard_mask": hard_mask,              # [B, N] per-sample
            "soft_weights": soft_weights,        # [B, N]
            "final_weights": final_weights,      # [B, N] STE
            "load_balance_loss": load_balance_loss,  # scalar
            "k_per_sample": k_per_sample,        # [B] 每样本实际 K
            "top_k_ids": top_k_ids,              # List[List[int]] 每样本 top-K 全局索引
        }


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
        spatial_diffusion_enabled: bool = False,  # C7: 空间场扩散（图拉普拉斯）
        spatial_diffusion_alpha: float = 0.1,  # C7: 扩散强度（0=关闭，0.1=温和，可由调质驱动）
        # ── C9: 自适应停止（推理路径）──
        # 固定 max_rounds 在多数输入下浪费算力（2 轮即收敛），
        # 自适应停止基于两个信号提前 break：
        # 1. 分数收敛：轮间 scores 变化 < convergence_threshold（共振已稳定）
        # 2. 主导明确：top1 分数 / top2 分数 > dominance_ratio（胜出者明确，继续轮无意义）
        # min_rounds 保证 side_signals 至少生效一次（默认 2）
        adaptive_stop: bool = False,  # C9: 默认关闭（向后兼容，固定 max_rounds）
        convergence_threshold: float = 0.01,  # C9: 轮间分数平均绝对变化 < 此值视为收敛
        dominance_ratio: float = 2.0,  # C9: top1/top2 > 此值视为主导明确
        min_rounds: int = 2,  # C9: 最少轮数（保证 side_signals 生效）
        # ── C14: shared_expert_weight 动态化 ──
        # False（默认）= 固定 shared_expert_weight（向后兼容）
        # True = MLP([max_domain_score, field_state]) → sigmoid → per-sample sw
        shared_weight_dynamic: bool = False,
        # ── §4.0c: Sparse Router（自适应激活）──
        # False（默认）= 稠密模式（向后兼容，所有神经元参与 round 2+）
        # True = Probe-based Sparse Router，round 1 后选 top-K 参与 round 2+
        use_sparse_router: bool = False,
        sparse_router_top_k: int = 3,
        sparse_router_warmup_steps: int = 2000,  # Phase 0/1 warm-up 步数
    ):
        self.neurons = neurons
        # 任务级并行（人脑：多线程处理不同任务）：
        # 默认场 _field（训练/诊断/无并发路径）；推理 forward 每任务用
        # thread-local 独立共振场（_get_task_field），跨任务互不干扰。
        self._field = field
        self._local = threading.local()
        self._last_forward_round_scores: List[Dict[str, float]] = []
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
        # C23-C（2026-08-08）：最后一轮可微演化相位（forward_train 用，梯度到 ω/K）
        self._last_evolved_phasors = None

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

        # ── C7: 空间场扩散（图拉普拉斯）──
        # 将"场是单一 D 维向量"升级为"空间场 + 扩散动力学"
        # 信号在神经元空间中传播：V' = normalize(V + alpha * L @ V)
        # L 是基于 NeuronGeometry 距离的对称归一化图拉普拉斯
        # 向后兼容：spatial_diffusion_enabled=False（默认）或 alpha=0 时完全退化
        self.spatial_diffusion_alpha = spatial_diffusion_alpha
        if spatial_diffusion_enabled and spatial_diffusion_alpha > 0:
            neuron_ids = list(self.neurons.keys())
            self.spatial_diffuser: Optional[SpatialDiffuser] = SpatialDiffuser(
                neuron_ids=neuron_ids,
                geometry=self.geometry,
                alpha=spatial_diffusion_alpha,
            )
        else:
            self.spatial_diffuser = None

        # ── C9: 自适应停止配置 ──
        self.adaptive_stop = adaptive_stop
        self.convergence_threshold = convergence_threshold
        self.dominance_ratio = dominance_ratio
        self.min_rounds = max(2, min(min_rounds, max_rounds))  # 至少 2 轮，不超过 max_rounds

        # ── C14: shared_expert_weight 动态化（方案 C: 共振分数 + 场状态联合）──
        # 原 sw 固定 0.3，无法随任务难度/共振强度调整。
        # 动态化后：共振强（域专精主导）→ sw 低；共振弱（无域主导）→ sw 高（shared 兜底）
        # 输入：[max_domain_score (标量), field_state_pool (D 维)] → MLP → sigmoid → [B,1]
        # - max_domain_score: 排除 shared_expert 的最高共振分（batch 级，反映"谁主导"）
        # - field_state: per-sample 场状态（反映"协作质量"，场信号丰富→协作充分）
        # 初始化偏置使初始 sw ≈ shared_expert_weight（向后兼容起点）
        # dynamic=False（默认）时退化为固定 shared_expert_weight（完全向后兼容）
        self.shared_weight_dynamic = shared_weight_dynamic
        # R3: consensus 融合参数（默认值，可被 __init__ 参数覆盖）
        self.consensus_k = 5       # 每神经元投票的 top-k token 数
        self.consensus_alpha = 0.5  # 共识加成强度（0=关闭，1=全员同意时权重翻倍）
        if (shared_weight_dynamic
                and shared_expert_id is not None
                and shared_expert_id in self.neurons):
            field_dim = self.field.dim
            # MLP: Linear(1 + field_dim, hidden) + GELU + Linear(hidden, 1)
            hidden = max(64, field_dim // 4)
            self.shared_weight_mlp = nn.Sequential(
                nn.Linear(1 + field_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            # 初始化使初始输出 ≈ logit(shared_expert_weight)
            # sigmoid(logit(0.3)) = 0.3，故最后一层 bias = logit(0.3)
            import math as _math
            with torch.no_grad():
                nn.init.zeros_(self.shared_weight_mlp[2].weight)
                self.shared_weight_mlp[2].bias.fill_(_math.log(shared_expert_weight / (1.0 - shared_expert_weight)))
        else:
            self.shared_weight_mlp = None

        # ── 大规模内存控制（B2/B3 fix）──
        self.logits_top_k = logits_top_k

        # ── Auxiliary-loss-free balancing ──
        # 每条 side_channel 的利用率统计（EMA），用于启发式 bias 更新
        # 低利用率的 channel 获得正 bias，增强其调制效果
        self._channel_usage: Dict[str, float] = {}  # "post->pre" -> EMA usage score
        self._channel_usage_ema_alpha = 0.99
        self._balancing_update_interval = 50  # 每 50 步更新一次 bias
        self._step_count = 0

        # ── Cross-spec field projector (T6: 升级为 2 层 MLP) ──
        # 混合规格协作：不同神经元 field_dim 不同时，投影到 field.dim 统一写入
        # T6 升级：从单层 Linear → CrossSpecProjector (Linear + GELU + Linear, 残差 + 零初始化)
        # - 第一层保留旧 Linear 语义（旧 checkpoint 可直接加载到 linear1.weight）
        # - 第二层零初始化，初始时 y = linear1(x)（与旧单层 Linear 完全一致，零破坏）
        # - 训练后第二层学到非线性变换，上限提升（单层线性 → 2 层 MLP 有非线性能力）
        self._cross_spec_projectors: Dict[str, "CrossSpecProjector"] = {}  # forward: field_dim -> unified
        self._cross_spec_back_projectors: Dict[str, "CrossSpecProjector"] = {}  # backward: unified -> field_dim
        field_dim = self.field.dim
        for nid, neuron in self.neurons.items():
            nfd = neuron.config.field_dim
            if nfd != field_dim:
                self._cross_spec_projectors[nid] = CrossSpecProjector(nfd, field_dim)
                self._cross_spec_back_projectors[nid] = CrossSpecProjector(field_dim, nfd)
        if self._cross_spec_projectors:
            print(f"  [ensemble] 检测到混合 field_dim，创建 {len(self._cross_spec_projectors)} 个跨规格投影层"
                  f"（T6: 2 层 MLP, 含反向投影）", flush=True)

        # ── C12: 评分投影（field_score_proj）──
        # 场状态投影到共享评分空间，与 neuron.score_proj 配对。
        # 让评分学习与写入学习解耦：大神经元对场状态方向的主导被 field_score_proj 抵消，
        # 小神经元能获得公平的共振分。
        # 启用条件：所有 neuron 的 score_dim 一致且不为 None。
        all_score_dims = [n.config.score_dim for n in self.neurons.values()]
        if all(sd is not None for sd in all_score_dims) and len(set(all_score_dims)) == 1:
            sd = all_score_dims[0]
            self.field_score_proj = nn.Linear(field_dim, sd, bias=False)
            nn.init.normal_(self.field_score_proj.weight, std=field_dim ** -0.5)
            self.score_dim = sd
        else:
            self.field_score_proj = None
            self.score_dim = None

        # ── §4.0c: Sparse Router 初始化 ──
        # use_sparse_router=False（默认）时不创建 Router，完全向后兼容
        # use_sparse_router=True 时创建 SparseRouter，round 1 后选 top-K
        self.use_sparse_router = use_sparse_router
        if use_sparse_router:
            self.sparse_router = SparseRouter(
                field_dim=self.field.dim,
                score_dim=self.score_dim,
                hidden_dim=128,
                top_k=sparse_router_top_k,
                warmup_steps=sparse_router_warmup_steps,
                shared_expert_id=shared_expert_id,
            )
            print(f"  [ensemble] Sparse Router 已启用（top_k={sparse_router_top_k}, "
                  f"warmup={sparse_router_warmup_steps}步, shared={shared_expert_id}）", flush=True)
        else:
            self.sparse_router = None

        # ── 缺口 M: 跨 vocab 联合训练 ──
        # tokenizer hub（与 cortex 同源，set_tokenizer_hub 注入）：
        # forward_train 在 vocab 不一致时，用词库转译矩阵把各 neuron logits
        # 投影到 target domain 空间再融合。
        self._tokenizer_hub = None
        # 词库转译矩阵缓存：{(src_domain, tgt_domain): {"fp": ..., "matrix": COO}}
        self._logits_alignment_cache: Dict[tuple, dict] = {}
        # 可编辑词库规则层（AlignmentRules）：人工覆盖自动转译，见 set_alignment_rules
        self._alignment_rules = None

        # ── C16b: per-neuron NLL EMA 统计（2026-08-08）──
        # 解决 C15 contrastive 监督的 NLL 跨 neuron 不可比（general 256K 空间英文
        # 主导 → code 域对几乎所有文本 NLL 全局最低 → quality_head 学到 code 独占）。
        # 语义修正：ideal = softmax(-(NLL-μ)/σ / τ)——"谁相对自己通常水平预测更好
        # 谁上"（per-neuron z-score），而非绝对 NLL 排序。推理路径不变。
        # 结构 {nid: {"mean": float, "ms": float, "count": float}}，Welford/EMA 混合。
        self._nll_ema: Dict[str, dict] = {}

    # =========================================================================
    # 任务级并行（人脑：多线程处理不同任务）——
    # 1) field 属性：推理 forward 期间返回本线程独立共振场（thread-local），
    #    其余路径（训练/诊断）返回默认场 _field。跨任务共振场互不污染。
    # 2) forward scratch（round_scores/_router_* 等）全部 thread-local，
    #    并发推理时每任务独立；forward 结束写穿到 _last_forward_round_scores
    #    供外部诊断读取（如 sleep_engine 分裂选择）。
    # =========================================================================

    @property
    def field(self) -> "ResonanceField":
        """任务级并行：推理 forward 期间返回本线程的独立共振场，
        其余路径（训练/诊断）返回默认场 _field。"""
        return getattr(self._local, "task_field", None) or self._field

    def _get_task_field(self) -> "ResonanceField":
        """获取本线程的任务共振场（懒创建 + 缓存）。

        继承默认场的 W_cond（门控参数）与 gamma gate（apply_gamma_gate 的
        monkey-patch），保证 per-task 场与默认场行为一致。
        """
        f = getattr(self._local, "task_field", None)
        if f is None:
            f = ResonanceField(dim=self._field.dim, device=self._field._device)
            with torch.no_grad():
                f.W_cond.copy_(self._field.W_cond)
            if self.gamma_oscillator is not None:
                from taiji.resonance.gamma_oscillator import apply_gamma_gate
                apply_gamma_gate(f, self.gamma_oscillator)
            self._local.task_field = f
        return f

    # ── thread-local forward scratch ──
    def _fstate(self, name: str, default=None):
        d = getattr(self._local, "fstate", None)
        if d is None:
            d = {}
            self._local.fstate = d
        return d.get(name, default)

    def _set_fstate(self, name: str, value) -> None:
        d = getattr(self._local, "fstate", None)
        if d is None:
            d = {}
            self._local.fstate = d
        d[name] = value

    @property
    def _logits_keep_ids(self):
        return self._fstate("_logits_keep_ids")

    @_logits_keep_ids.setter
    def _logits_keep_ids(self, v):
        self._set_fstate("_logits_keep_ids", v)

    @property
    def _last_router_result(self):
        return self._fstate("_last_router_result")

    @_last_router_result.setter
    def _last_router_result(self, v):
        self._set_fstate("_last_router_result", v)

    @property
    def _router_active_ids(self):
        return self._fstate("_router_active_ids")

    @_router_active_ids.setter
    def _router_active_ids(self, v):
        self._set_fstate("_router_active_ids", v)

    @property
    def _router_hard_mask(self):
        return self._fstate("_router_hard_mask")

    @_router_hard_mask.setter
    def _router_hard_mask(self, v):
        self._set_fstate("_router_hard_mask", v)

    @property
    def round_scores(self) -> List[Dict[str, float]]:
        v = self._fstate("round_scores")
        return v if v is not None else []

    @round_scores.setter
    def round_scores(self, v):
        self._set_fstate("round_scores", v)

    @property
    def n_active_history(self) -> List[int]:
        v = self._fstate("n_active_history")
        return v if v is not None else []

    @n_active_history.setter
    def n_active_history(self, v):
        self._set_fstate("n_active_history", v)

    def set_tokenizer_hub(self, tokenizer_hub) -> None:
        """注入 TokenizerHub（跨 vocab 联合训练需要访问各域 tokenizer）。

        与 cortex.set_tokenizer_hub 同源；不传则跨 vocab 训练路径无法构建
        词库转译矩阵（vocab 一致时完全不需要，向后兼容）。
        """
        self._tokenizer_hub = tokenizer_hub

    def set_alignment_rules(self, rules) -> None:
        """注入可编辑词库规则层（AlignmentRules，人工覆盖自动转译）。

        规则增删后（version 变化）词库转译矩阵缓存自动失效重建。

        Args:
            rules: AlignmentRules 实例（None 时清除规则层）。
        """
        self._alignment_rules = rules

    def _get_neuron_tokenizer(self, nid: str):
        """从 hub 解析 neuron 的域 tokenizer。

        nid 即域名（如 "code"/"math"）时直接用；否则取 domain 前缀
        （如 "zh_aug0_dialogue" → "zh"）。返回 None 表示无法解析。
        """
        if self._tokenizer_hub is None:
            return None
        sp = self._tokenizer_hub.get_tokenizer(nid)
        if sp is not None:
            return sp
        domain = nid.split("_")[0]
        return self._tokenizer_hub.get_tokenizer(domain)

    def _project_logits_to_target(
        self, final_logits: Dict[str, torch.Tensor], active_ids: List[str],
        target_domain: str,
    ) -> torch.Tensor:
        """缺口 M: 把各 neuron logits 用词库转译矩阵投影到 target 域空间。

        返回 [N, B, L, V_tgt]（已 stack，可直接参与融合）。
        """
        if self._tokenizer_hub is None:
            raise RuntimeError(
                "[forward_train] 跨 vocab 联合训练需要 tokenizer hub。"
                "请先调用 ensemble.set_tokenizer_hub(hub)。"
            )
        target_sp = self._tokenizer_hub.get_tokenizer(target_domain)
        if target_sp is None:
            raise RuntimeError(
                f"[forward_train] target_domain='{target_domain}' 不在 tokenizer hub 中。"
            )
        tgt_vocab = target_sp.GetPieceSize() if hasattr(target_sp, "GetPieceSize") else 0
        if tgt_vocab <= 0:
            raise RuntimeError(
                f"[forward_train] target tokenizer '{target_domain}' 无有效 vocab。"
            )

        projected = []
        for nid in active_ids:
            logits = final_logits[nid]  # [B, L, V_i]
            if logits.shape[-1] == tgt_vocab:
                projected.append(logits)
                continue
            src_sp = self._get_neuron_tokenizer(nid)
            if src_sp is None:
                raise RuntimeError(
                    f"[forward_train] neuron '{nid}' 的 tokenizer 无法从 hub 解析，"
                    f"无法投影到 target domain '{target_domain}'。"
                )
            matrix = build_logits_alignment_matrix(
                src_sp, target_sp,
                source_domain=nid, target_domain=target_domain,
                cache=self._logits_alignment_cache,
                overrides=self._alignment_rules,
                source_vocab_size=logits.shape[-1],
            )
            # [B*L, V_i] @ [V_i, V_tgt] → [B*L, V_tgt] → [B, L, V_tgt]
            b, l, vi = logits.shape
            logits_2d = logits.reshape(-1, vi)
            proj_2d = torch.sparse.mm(logits_2d, matrix.to(logits_2d.dtype))
            projected.append(proj_2d.reshape(b, l, tgt_vocab))

        return torch.stack(projected)  # [N, B, L, V_tgt]

    def _project_vec(self, nid: str, vec: torch.Tensor) -> torch.Tensor:
        """Cross-spec projection: 将 neuron 的 field_vector 投影到 field.dim。

        混合规格协作时，不同神经元的 field_dim 不同，需要投影到统一维度才能写入 field。
        """
        if nid in self._cross_spec_projectors:
            return self._cross_spec_projectors[nid](vec)
        return vec

    # ── C23-B（2026-08-08）：相位绑定辅助（场写入按相位同步加权）──

    def _phase_binding_map(self, active_ids) -> Optional[dict]:
        """当前激活 neuron 的相位绑定图：{nid: binding ∈ [-1,1]}。

        同相群体 → +1（绑结成知觉单元），异相 → -1（解绑）。
        共激活调制与 Kuramoto 耦合一致（"共激活→相位同步→绑结"闭环）。
        """
        if self.gamma_oscillator is None:
            return None
        try:
            return self.gamma_oscillator.pairwise_binding(
                list(active_ids), coactivation=self.coaction,
            )
        except Exception:
            return None

    def _phase_binding_scale(self) -> float:
        """绑定强度 β（GammaOscillator.binding_scale，0 = 关闭）。"""
        if self.gamma_oscillator is None:
            return 0.0
        return float(getattr(self.gamma_oscillator, "binding_scale", 0.0))

    def _check_adaptive_stop(
        self,
        current_scores: Dict[str, float],
        prev_scores: Optional[Dict[str, float]],
        round_num: int,
    ) -> tuple:
        """C9: 检查是否应提前停止共振。

        两个停止信号（任一满足即停止）：
        1. 分数收敛：轮间 scores 平均绝对变化 < convergence_threshold
        2. 主导明确：top1/top2 > dominance_ratio（胜出者明确）

        Args:
            current_scores: 当前轮 scores {nid: float}
            prev_scores: 上一轮 scores（None 时只检查主导信号）
            round_num: 当前轮次

        Returns:
            (should_stop: bool, reason: str)
        """
        if not self.adaptive_stop:
            return False, ""
        # min_rounds 以下不停止（保证 side_signals 生效）
        if round_num < self.min_rounds:
            return False, ""
        # 已到 max_rounds 自然停止（不需要提前 break）
        if round_num >= self.max_rounds:
            return False, ""

        # 信号 1: 分数收敛
        if prev_scores is not None and len(current_scores) > 0:
            common = set(current_scores.keys()) & set(prev_scores.keys())
            if len(common) >= 2:
                diffs = [abs(current_scores[n] - prev_scores[n]) for n in common]
                avg_diff = sum(diffs) / len(diffs)
                if avg_diff < self.convergence_threshold:
                    return True, f"converged(avg_diff={avg_diff:.4f}<{self.convergence_threshold})"

        # 信号 2: 主导明确（top1 / top2）
        if len(current_scores) >= 2:
            sorted_scores = sorted(current_scores.values(), reverse=True)
            top1, top2 = sorted_scores[0], sorted_scores[1]
            # top2 接近 0 时跳过（避免除零），只看 top1 是否显著
            if top2 > 1e-6 and (top1 / top2) > self.dominance_ratio:
                return True, f"dominant(top1/top2={top1/top2:.2f}>{self.dominance_ratio})"

        return False, ""

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
                # 快照隔离容错：post 神经元可能已被并发增删移除
                post_neuron = self.neurons.get(post_id)
                if post_neuron is None:
                    continue
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
                # 快照隔离容错：post 神经元可能已被并发增删移除
                post_neuron = self.neurons.get(post_id)
                if post_neuron is None:
                    continue
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

        混合规格热插拔：field_dim 允许不同（跨规格投影层在下方自动补建），
        hidden_size 必须一致（共享 embedding 契约，防御性校验）。

        Args:
            nid: 神经元 ID（如 "zh_1"）
            neuron: ResonanceNeuron 实例
            from_split: 分裂父 neuron ID（LuminaNet splitting），
                        用于在几何空间中放置子 neuron 在父 neuron 附近
        """
        if nid in self.neurons:
            raise ValueError(f"神经元 {nid} 已存在于 ensemble")

        # 校验 hidden_size 一致性（field_dim 混合规格由跨规格投影层处理）
        if self.neurons:
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

        # 混合规格热插拔（2026-08-06 修复）：新 neuron field_dim ≠ unified 时
        # 必须补建跨规格投影层，否则推理 _project_vec 用 identity 导致维度错配崩溃
        if neuron.config.field_dim != self.field.dim:
            self._cross_spec_projectors[nid] = CrossSpecProjector(
                neuron.config.field_dim, self.field.dim)
            self._cross_spec_back_projectors[nid] = CrossSpecProjector(
                self.field.dim, neuron.config.field_dim)

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
        side_signals: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        temp_gain: float = 1.0,
        ffn_gain: float = 1.0,
        nmap: Optional[Dict] = None,
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
            side_signals: {post_nid: {pre_nid: field_vector}} per-pair 突触信号
            temp_gain: S9 注意力温度增益（norepinephrine 驱动，所有 neuron 共享）
            ffn_gain: S9 FFN 输出增益（dopamine 驱动，所有 neuron 共享）

        Returns:
            (round_vecs, round_logits, round_confidences, round_score_vecs, round_quality_logits)
            - round_vecs: {nid: [B, field_dim]} L2-normalized field vectors
            - round_logits: {nid: [B, L, vocab]} optional logits
            - round_confidences: {nid: [B]} C8 per-sample confidence ∈ [0, 1]
            - round_score_vecs: {nid: [B, score_dim]} C12 评分投影向量（score_dim=None 时为空 dict）
            - round_quality_logits: {nid: [B, 1]} C15 预测质量 logit（round 1，无场耦合）
        """
        round_vecs: Dict[str, torch.Tensor] = {}
        round_logits: Dict[str, torch.Tensor] = {}
        round_confidences: Dict[str, torch.Tensor] = {}  # C8: per-sample confidence
        round_score_vecs: Dict[str, torch.Tensor] = {}  # C12: 评分投影向量
        round_quality_logits: Dict[str, torch.Tensor] = {}  # C15: 预测质量 logit（round 1）

        # 快照隔离：调用方（forward/forward_train）传入 nmap，缺省用 self.neurons
        nmap = nmap if nmap is not None else self.neurons

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
            return nmap[nid].forward(emb, **kwargs)

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
                # C8: 提取 per-sample confidence
                round_confidences[nid] = results[nid].get(
                    "field_confidence", torch.ones(results[nid]["field_vector"].shape[0])
                )
                # C12: 提取评分投影向量
                if "score_vec" in results[nid]:
                    round_score_vecs[nid] = results[nid]["score_vec"]
                # C15: 提取预测质量 logit（round 1 独立前向，无场耦合）
                if round_num == 1 and "quality_logit" in results[nid]:
                    round_quality_logits[nid] = results[nid]["quality_logit"]
                if return_logits_filter(nid):
                    round_logits[nid] = results[nid]["logits"]
        else:
            # CPU 模式或单神经元：串行
            for nid in active_ids:
                need_logits = return_logits_filter(nid)
                emb = _get_emb(nid)
                result = _forward_neuron(nid, emb, need_logits)
                round_vecs[nid] = result["field_vector"]
                # C8: 提取 per-sample confidence
                round_confidences[nid] = result.get(
                    "field_confidence", torch.ones(result["field_vector"].shape[0])
                )
                # C12: 提取评分投影向量
                if "score_vec" in result:
                    round_score_vecs[nid] = result["score_vec"]
                # C15: 提取预测质量 logit（round 1 独立前向，无场耦合）
                if round_num == 1 and "quality_logit" in result:
                    round_quality_logits[nid] = result["quality_logit"]
                if need_logits:
                    round_logits[nid] = result["logits"]

        return round_vecs, round_logits, round_confidences, round_score_vecs, round_quality_logits

    def forward(
        self,
        shared_embeddings: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        active_filter: bool = True,
        active_nids: Optional[List[str]] = None,
        neuron_embeddings: Optional[Dict[str, torch.Tensor]] = None,
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
            fusion_mode: 推理融合模式（方向③ 残差预测编码）
                        - "per_position"（默认）：每位置按熵/置信度独立路由（向后兼容）
                        - "residual"：族长完整预测 + 其他神经元残差修正
                        - "division"：统一空间（同 vocab）max-prob 分工路由——每位置
                          直接交给 max-prob 最高的 neuron（共享 general lm_head 后置信度
                          天然尖锐，无转译投影稀释）

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

        # 任务级并行：本任务用独立共振场（thread-local），跨任务互不干扰；
        # forward 期间 self.field 属性解析到本线程的任务场，其余路径回默认场
        self._get_task_field().reset(batch_size=ref.shape[0])
        self.round_scores = []
        self.n_active_history = []
        self._logits_keep_ids = None  # 每次 forward 重置
        # §4.0c+d: 初始化 Router 缓存（每次 forward 重置）
        self._last_router_result = None
        self._router_active_ids = None
        self._router_hard_mask = None

        # 快照隔离（人脑：神经元动态增删不影响工作中的推理）
        # 浅拷贝：推理全程用 nmap，增删只改原 dict——被删的 neuron 在快照中
        # 仍持有对象引用，推理不崩（统计性影响，符合人脑冗余/并行特性）。
        # 训练微调同对象共享引用 → 自然反映到后续推理。
        nmap = dict(self.neurons)

        # P0-2 fix (MAJOR-2): 重置所有 neurons 的 refractory_counter
        # 防止跨 forward 调用的状态泄漏（上次推理进入不应期的 neuron 不应影响本次）
        for nid in nmap:
            nmap[nid].refractory_counter.fill_(0)

        # P1-STDP: 每次推理开始时清空 firing history（一次推理内的发放时序）
        if self.stdp_tracker is not None:
            self.stdp_tracker._firing_history.clear()

        neuron_ids = list(nmap.keys())
        # 如果指定了 active_nids，只激活这些 neuron（精简模式）
        # 注意：用有序 list 而非 set——set 迭代顺序不确定会导致 round_logits /
        # 融合结果 dict 顺序随机，路由监控/诊断索引错位（2026-08-06 统一空间路由
        # 验证中发现：active_ids 为 set 时 all_logits.keys() 每次运行顺序不同）。
        if active_nids is not None:
            seen: set = set()
            active_ids = [nid for nid in active_nids
                          if nid in nmap and not (nid in seen or seen.add(nid))]
            if not active_ids:
                # fallback: 全部 neuron
                active_ids = list(neuron_ids)
        else:
            active_ids = list(neuron_ids)

        # Shared Expert: general 神经元始终激活（不受路由/精简模式影响）
        if self.shared_expert_id and self.shared_expert_id in nmap \
                and self.shared_expert_id not in active_ids:
            active_ids.append(self.shared_expert_id)

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
        # C19（2026-08-08）：快照 round1 全量 neuron（共振过滤会缩小 active_ids，
        # 但回合级质量信号需要全部候选，final 聚合用此快照）
        round1_active_ids = list(active_ids)
        large_scale = return_logits and len(active_ids) > self.logits_top_k
        round1_return_logits = return_logits and not large_scale

        # Q4 fix: 使用 _parallel_forward（GPU 自动 CUDA stream 并行）
        def round1_logits_filter(nid):
            return round1_return_logits

        round_vecs, round_logits, round_confidences, round_score_vecs_r1, round_quality_logits_r1 = self._parallel_forward(
            active_ids,
            shared_embeddings,
            field_state=None,
            round_num=1,
            return_logits_filter=round1_logits_filter,
            neuron_embeddings=neuron_embeddings,
            temp_gain=temp_gain,
            ffn_gain=ffn_gain,
            nmap=nmap,
        )

        # Write round 1 to field
        # P1-2: 从 NeuromodulatorState 读取 field_write_scale（去甲肾上腺素驱动）
        write_scale = (self.neuromodulator.get_field_write_scale()
                       if self.neuromodulator is not None else 1.0)
        # C23-B（2026-08-08）：相位绑定写入——同相群体写入增强（场状态由相位同步塑造）
        binding_map = self._phase_binding_map(active_ids)
        binding_bs = self._phase_binding_scale()
        for nid in active_ids:
            # P0#3: 抑制性神经元走 write_inhibit（乘法衰减），兴奋性走 write（累加）
            neuron = nmap[nid]
            # MaturityTracker: 幼稚态低共振权重（0.1），成熟态 1.0
            maturity_w = (self.maturity.get_resonance_weight(nid)
                          if self.maturity is not None else 1.0)
            # Cross-spec projection: 将不同 field_dim 的向量投影到 field.dim
            vec = self._project_vec(nid, round_vecs[nid])
            # C1: 亚型写入增益（PV+ 强写入, SOM+ 弱写入等）
            vec = vec * neuron.write_gain
            if neuron.is_inhibitory:
                # C23-B：抑制性 neuron 相位锁定 gamma（生物：PV+ 与兴奋群体同步发放），
                # 同相 → 抑制增强（绑定单元内部抑制同样参与信息整合）
                if binding_map and binding_bs != 0.0:
                    vec = vec * (1.0 + binding_bs * binding_map.get(nid, 0.0))
                self.field.write_inhibit(nid, vec, weight=maturity_w)
            else:
                # C8: per-sample confidence 调制写入幅度（高置信度→大写入）
                scale = write_scale * maturity_w * round_confidences[nid]
                if binding_map and binding_bs != 0.0:
                    scale = scale * (1.0 + binding_bs * binding_map.get(nid, 0.0))
                self.field.write(nid, vec, scale=scale)
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
        n_inhibitory = sum(1 for nid in active_ids if nmap[nid].is_inhibitory)
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
        # C1: 再乘以亚型不应期乘数（PV+ 短不应期, SOM+ 长不应期等）
        neuromod_mult = (self.neuromodulator.get_refractory_multiplier()
                         if self.neuromodulator is not None else 1.0)
        ranked_round1 = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        refractory_k = max(1, min(len(ranked_round1) // 2, self.logits_top_k))
        for nid, _ in ranked_round1[:refractory_k]:
            # C1: 亚型不应期乘数 × 调质乘数
            subtype_mult = nmap[nid].refractory_multiplier
            nmap[nid].enter_refractory(multiplier=neuromod_mult * subtype_mult)

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
                best_emb = (neuron_embeddings[best_nid]
                            if neuron_embeddings is not None and best_nid in neuron_embeddings
                            else shared_embeddings)
                best_result = nmap[best_nid].forward(best_emb, **best_kwargs)
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

        # ── §4.0c+d: Sparse Router 接入（推理路径）──
        # round 1 后用 Router 选 per-sample top-K
        # round 2+ 用 per-sample mask 控制 side_signals/field 写入
        # （每样本只被其 top-K 神经元影响 = per-sample 稀疏信号流）
        # use_sparse_router=False 时跳过（向后兼容）
        # 推理时不用 STE，直接 hard top-K（与训练 forward 的 hard 一致）
        if (
            self.use_sparse_router
            and self.sparse_router is not None
            and self.max_rounds >= 2
        ):
            # 构建 round_vecs_unified（投影到 unified 维度）
            round_vecs_unified_r1 = {
                nid: self._project_vec(nid, round_vecs[nid]) for nid in active_ids
            }
            # active_ids 是 set，转 list 保证顺序稳定
            router_active_list = sorted(active_ids)
            router_result = self.sparse_router(
                active_ids=router_active_list,
                round_vecs_unified=round_vecs_unified_r1,
                round_confidences=round_confidences,
                round_score_vecs=round_score_vecs_r1 if round_score_vecs_r1 else None,
                step=10**9,  # 推理时固定 Phase 2（熵驱动动态 K）
            )
            # 缓存 router_result + per-sample hard_mask [B, N]
            self._last_router_result = router_result
            self._router_active_ids = router_active_list
            self._router_hard_mask = router_result["hard_mask"]
        else:
            self._last_router_result = None
            self._router_hard_mask = None

        # ── Side-channel construction (per-pair synaptic projection) ──
        # Round 1 后，每个 post 神经元接收其他 pre 神经元的原始 field_vector，
        # 通过各自的 side_channels 投影到 hidden 空间进行调制。
        side_signals_per_neuron: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        if self.max_rounds >= 2:
            side_signals_per_neuron = {nid: {} for nid in active_ids}
            # §4.0d: per-sample 稀疏——post 神经元只接收该样本 top-K 的 pre 信号
            router_mask = self._router_hard_mask  # [B, N] or None
            for post_id in active_ids:
                for pre_id in active_ids:
                    if post_id == pre_id:
                        continue
                    post_neuron = nmap[post_id]
                    if (pre_id in post_neuron.excite_channels or
                            pre_id in post_neuron.inhibit_channels):
                        pre_vec = round_vecs[pre_id]  # [B, D]
                        if router_mask is not None:
                            pre_mask = router_mask[:, self._router_active_ids.index(pre_id)]  # [B]
                            side_signals_per_neuron[post_id][pre_id] = pre_vec * pre_mask.unsqueeze(-1)
                        else:
                            side_signals_per_neuron[post_id][pre_id] = pre_vec

            # Auxiliary-loss-free balancing: 更新 channel 利用率统计
            self._update_channel_usage(side_signals_per_neuron, round_vecs)
            self._step_count += 1
            if self._step_count % self._balancing_update_interval == 0:
                self._update_channel_biases()

        # ── Rounds 2+: conditioned resonance ──
        prev_round_scores: Optional[Dict[str, float]] = self.round_scores[-1] if self.round_scores else None
        adaptive_stop_reason: Optional[str] = None
        for round_num in range(2, self.max_rounds + 1):
            # P0-2 fix: round 2+ 也基于当前 _logits_keep_ids 过滤，但每轮会重新计算
            def round2_logits_filter(nid):
                return return_logits and (
                    self._logits_keep_ids is None or nid in self._logits_keep_ids
                )

            round_vecs, round_logits, round_confidences, _round_score_vecs, _round_quality_logits = self._parallel_forward(
                active_ids,
                shared_embeddings,
                field_state=self.field.get_normalised_state() if field_conditioning else None,
                round_num=round_num,
                return_logits_filter=round2_logits_filter,
                neuron_embeddings=neuron_embeddings,
                side_signals=side_signals_per_neuron,
                temp_gain=temp_gain,
                ffn_gain=ffn_gain,
                nmap=nmap,
            )

            # 人脑启发：不应期调度（错峰写入）
            writable_ids = []
            refractory_ids = []
            for nid in active_ids:
                if nmap[nid].in_refractory:
                    refractory_ids.append(nid)
                else:
                    writable_ids.append(nid)

            # C23-B（2026-08-08）：每轮重算相位绑定（相位已随 Kuramoto 演化，
            # 同相群体随轮次逐步绑结 → 写入增强；异相解绑 → 写入衰减）
            binding_map = self._phase_binding_map(writable_ids)
            binding_bs = self._phase_binding_scale()

            for nid in writable_ids:
                # P0#3: 抑制性神经元走 write_inhibit，兴奋性走 update
                neuron = nmap[nid]
                # MaturityTracker: 幼稚态低共振权重（0.1），成熟态 1.0
                maturity_w = (self.maturity.get_resonance_weight(nid)
                              if self.maturity is not None else 1.0)
                vec = self._project_vec(nid, round_vecs[nid])
                # C1: 亚型写入增益（PV+ 强写入, SOM+ 弱写入等）
                vec = vec * neuron.write_gain
                # §4.0d: per-sample 稀疏——只在该样本被 top-K 选中时写入场
                if self._router_hard_mask is not None:
                    vec_mask = self._router_hard_mask[:, self._router_active_ids.index(nid)]  # [B]
                    vec = vec * vec_mask.unsqueeze(-1)
                if neuron.is_inhibitory:
                    # C23-B：抑制性 neuron 同相 → 抑制增强（与 round1 一致）
                    if binding_map and binding_bs != 0.0:
                        vec = vec * (1.0 + binding_bs * binding_map.get(nid, 0.0))
                    self.field.write_inhibit(nid, vec, weight=maturity_w)
                else:
                    # P1-2: round 2+ 也应用 neuromodulator 调质
                    # C8: per-sample confidence 调制写入幅度
                    scale = write_scale * maturity_w * round_confidences[nid]
                    if binding_map and binding_bs != 0.0:
                        scale = scale * (1.0 + binding_bs * binding_map.get(nid, 0.0))
                    self.field.update(nid, vec, scale=scale)
                nmap[nid].enter_refractory(
                    multiplier=neuromod_mult * nmap[nid].refractory_multiplier
                )
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

            # C23（2026-08-08）：相位同步本体化——pairwise binding 调制共振分
            # 推理路径此前只做 Kuramoto 相位演化（无消费端，纯装饰）；现在
            # 同相 neuron 群体（同 domain 同相位）binding 高 → 共振分增强 →
            # 融合权重集中（绑结成知觉单元）；异相解绑。与 forward_train 一致。
            if self.gamma_oscillator is not None:
                try:
                    binding = self.gamma_oscillator.pairwise_binding(
                        list(active_ids), coactivation=self.coaction,
                    )
                    bs = getattr(self.gamma_oscillator, "binding_scale", 0.0)
                    if binding and bs != 0.0:
                        for nid in scores:
                            scores[nid] = scores[nid] * (1.0 + bs * binding.get(nid, 0.0))
                except Exception:
                    pass  # 相位绑定失败不影响共振主流程
            self.round_scores.append(scores)

            # P0-2 fix: 每轮基于当前 scores 重新计算 _logits_keep_ids（原 bug：round 1 后冻结）
            if return_logits:
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                self._logits_keep_ids = {nid for nid, _ in ranked[:self.logits_top_k]}

            # C9: 自适应停止检查（在 active_filter 之前，用完整 scores 判断）
            should_stop, reason = self._check_adaptive_stop(
                current_scores=scores,
                prev_scores=prev_round_scores,
                round_num=round_num,
            )
            if should_stop:
                adaptive_stop_reason = reason
                self.n_active_history.append(len(active_ids))
                vectors = round_vecs
                all_logits = round_logits
                break

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
                # §4.0d: per-sample 稀疏——每轮重建时同样应用 top-K mask
                router_mask = self._router_hard_mask  # [B, N] or None
                for post_id in active_ids:
                    post_neuron = nmap[post_id]
                    for pre_id in active_ids:
                        if post_id == pre_id:
                            continue
                        if (pre_id in post_neuron.excite_channels or
                                pre_id in post_neuron.inhibit_channels):
                            pre_vec = round_vecs[pre_id]
                            if router_mask is not None:
                                pre_mask = router_mask[:, self._router_active_ids.index(pre_id)]  # [B]
                                side_signals_per_neuron[post_id][pre_id] = pre_vec * pre_mask.unsqueeze(-1)
                            else:
                                side_signals_per_neuron[post_id][pre_id] = pre_vec

            # 人脑启发：每轮结束递减所有神经元的不应期计数器
            for nid in nmap:
                nmap[nid].tick_refractory()

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
            # C9: 更新 prev_round_scores 供下一轮收敛检查
            prev_round_scores = scores

        # ── Final output ──
        # 写穿 thread-local round_scores → 共享镜像（供外部诊断如 sleep_engine 分裂选择）
        self._last_forward_round_scores = self.round_scores
        result = {
            "field_state": self.field.get_state(),
            "final_scores": self.round_scores[-1] if self.round_scores else {},
            "n_rounds": len(self.round_scores),
            "n_active_history": self.n_active_history,
            "skipped_resonance": False,
            "skip_reason": None,
            "adaptive_stopped": adaptive_stop_reason is not None,  # C9
            "adaptive_stop_reason": adaptive_stop_reason,  # C9
            # C21（2026-08-08）：round1 独立 logits（无场条件化）——executive 模式
            # leader 生成用（协作/共振分只用于任务模式判定，不污染 leader 的域
            # 词表能力输出；round2 场条件化会注入混合域信号，稀释 dialogue 的 zh 能力）
            "round1_logits": round_logits,  # {nid: [B, L, V]}（非 large_scale 时全量）
        }

        # C19（2026-08-08）：推理路径也暴露 round1 quality_logits（回合级聚合）
        # ——ExecutiveController（_executive_route）用 probe forward 拿回合级
        # 质量信号做任务模式判定。此前只在 forward_train 收集（C15 训练监督用）。
        # 聚合用 round1 快照（round1_active_ids），不用共振过滤后的 active_ids——
        # 否则共振收敛只保留最强 neuron，回合级判定退化为单 neuron 信号。
        if round_quality_logits_r1:
            ql = torch.stack([
                round_quality_logits_r1[nid] for nid in round1_active_ids
                if nid in round_quality_logits_r1
            ]).mean(dim=1)  # [N, 1]
            result["quality_logits"] = ql.squeeze(-1)  # [N]（与 round1 顺序一致）
        else:
            result["quality_logits"] = None

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
            elif fusion_mode == "consensus" and same_vocab and len(all_logits) >= 2:
                # R3: 共识投票融合（推理路径）
                # 多神经元 top-k 预测一致性加权（集体智慧）
                self._consensus_logit_fusion(all_logits, scores, result, ref)
            elif fusion_mode in ("soft", "score") and same_vocab and len(all_logits) >= 2:
                # 缺口 N 修复（2026-08-06）：共振分融合（与训练 forward_train 对齐）
                # 训练 "soft" = softmax(scores/temp) 融合（C12 让共振分与 NLL 排序对齐）。
                # 推理此前 "soft" 实际走 per_position（entropy 启发式），无视训练学的
                # 共振分 → 训练-推理不一致（训练分布 PPL 2.2 vs 推理 12.6，生成质量差）。
                # A/B 实验：共振分融合生成明显更通顺。统一 "soft" 语义 = 共振分融合，
                # 让推理直接复用训练学到的协作权重。per_position 保留为显式选项。
                self._score_logit_fusion(all_logits, scores, result, ref)
            elif fusion_mode == "division" and same_vocab and len(all_logits) >= 2:
                # 统一空间 max-prob 分工路由（2026-08-06）：共享 general lm_head 后
                # 所有 neuron 原生输出在 256K 空间，max-prob 天然尖锐（旧诊断
                # max-prob≈0.001 是静态稀疏投影稀释所致，已消除）。per-position
                # 硬路由把每个 token 位置交给最自信的 neuron——域内文本由域 neuron
                # 全位置胜出（≈个体无伤害），跨域语义桥接时不同位置不同 neuron 分工。
                # 注意：跨 neuron 的 logits 尺度不可比（code 平均 max-prob 0.566 vs
                # zh 0.374），裸 max-prob 分工在协作层未训练时 PPL 劣于最强个体——
                # 正确校准需协作层训练（C12/Sparse Router 学习路由）。division_norm
                # 提供 per-neuron 相对归一化（供诊断，实测会过度平坦化对角消失）。
                self._division_logit_fusion(all_logits, result, ref, normalize=False)
            elif fusion_mode == "division_norm" and same_vocab and len(all_logits) >= 2:
                # per-neuron 相对归一化分工（消除系统性尺度偏差，供对照诊断）
                self._division_logit_fusion(all_logits, result, ref, normalize=True)
            elif same_vocab:
                self._compute_per_position_weights(
                    all_logits, vectors, scores, result, ref,
                )
            else:
                # 跨 vocab（混合阵容）收敛（2026-08-07）：与训练口径一致——
                # 投影到统一目标空间（general 256K）+ confidence routing
                # （min(原生,投影) max-prob × trust(scores)）。
                # 此前该分支不做融合，把责任甩给 cortex._generate_p7 的
                # MoCo 动态融合（_dynamic_logit_fusion，已删除）→ 训练-推理
                # 融合路径不一致。现在 forward 直接产出 weighted_logits，
                # cortex 已优先使用该键（weighted_logits 分支）。
                nids = list(all_logits.keys())
                if self._tokenizer_hub is not None and len(nids) >= 2:
                    try:
                        proj = self._project_logits_to_target(all_logits, nids, "general")
                        native_list = [all_logits[nid] for nid in nids]
                        scores_t = torch.tensor(
                            [scores.get(nid, 0.0) for nid in nids],
                            dtype=ref.dtype, device=ref.device,
                        )
                        # C15: 推理也优先用预测质量 logit（round 1 独立前向，无场耦合）
                        quality_t = None
                        if round_quality_logits_r1:
                            dl = [round_quality_logits_r1[nid] for nid in nids if nid in round_quality_logits_r1]
                            if len(dl) == len(nids):
                                quality_t = torch.stack(dl).mean(dim=1).squeeze(-1)  # [N]
                        fused, rw = self._confidence_routing_fusion(
                            proj, native_list, nids, "general",
                            scores=scores_t,
                            quality_logits=quality_t,
                        )
                        result["weighted_logits"] = fused
                        result["weights"] = rw.detach().cpu().tolist()
                        result["fusion_mode"] = "confidence_routing"
                    except Exception as e:
                        # 投影/融合失败（hub 缺域等）→ 保持原行为（cortex fallback）
                        result["fusion_mode"] = "neuron_logits_only"
                        result["fusion_error"] = str(e)
                else:
                    result["fusion_mode"] = "neuron_logits_only"

            # Shared Expert 重新加权（借鉴 Kimi K3 / DeepSeek V3）
            # general 神经元获得基础权重，域特定神经元按原逻辑分配剩余权重
            # final = shared_weight * shared_logits + (1-shared_weight) * original_fused
            # C14: shared_weight 动态化（方案 C: 共振分数 + 场状态联合）
            # - dynamic=False: 固定 shared_expert_weight（标量，向后兼容）
            # - dynamic=True:  sw = sigmoid(MLP([max_domain_score, field_state]))  # per-sample [B,1]
            if (
                self.shared_expert_id
                and self.shared_expert_id in all_logits
                and "weighted_logits" in result
            ):
                shared_logits = all_logits[self.shared_expert_id]
                original_fused = result["weighted_logits"]

                # C14: 计算 shared_weight
                if self.shared_weight_mlp is not None:
                    # field_state: [B,D] 或 [D]（训练单线程：显式用默认场）
                    field_state = self._field.get_state()
                    if field_state.dim() == 1:
                        field_state = field_state.unsqueeze(0)  # [1,D]
                    B = field_state.shape[0]
                    # max_domain_score: 排除 shared_expert 的最高共振分（batch 级）
                    domain_scores = [s for nid, s in scores.items() if nid != self.shared_expert_id]
                    max_domain_score = max(domain_scores) if domain_scores else 0.0
                    max_score_tensor = torch.full(
                        (B, 1), float(max_domain_score),
                        device=field_state.device, dtype=field_state.dtype,
                    )
                    # 联合输入 [B, 1+D] → MLP → sigmoid → [B,1]
                    mlp_input = torch.cat([max_score_tensor, field_state], dim=-1)
                    sw = torch.sigmoid(self.shared_weight_mlp(mlp_input))  # [B,1]
                    # 扩展到 [B,1,1] 与 [B,L,V] 广播
                    sw_broadcast = sw.unsqueeze(-1)
                    # 标量记录（final_weights 仍用标量，保持接口兼容）
                    sw_scalar = float(sw.mean().item())
                else:
                    sw = self.shared_expert_weight  # 标量 float
                    sw_broadcast = sw
                    sw_scalar = sw

                result["weighted_logits"] = sw_broadcast * shared_logits + (1.0 - sw_broadcast) * original_fused
                # 更新 final_weights 反映 Shared Expert 的权重
                if "final_weights" in result:
                    weights = result["final_weights"]
                    # 域特定神经元权重缩放到 (1-sw)
                    for nid in weights:
                        if nid != self.shared_expert_id:
                            weights[nid] = weights[nid] * (1.0 - sw_scalar)
                    weights[self.shared_expert_id] = sw_scalar
                # C14: 记录 per-sample sw（供调试/监控）
                if self.shared_weight_mlp is not None:
                    result["shared_weight_per_sample"] = sw.detach().squeeze(-1)  # [B]

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
        targets: Optional[torch.Tensor] = None,  # C12: per-neuron NLL 排序对比信号
        # ── C20（2026-08-08）：回合级监督 answer_mask ──
        # [B, L] bool，1=answer（回复）部分。per_neuron_nll 只对 answer 部分算
        # 回合级 NLL——prompt 部分所有 neuron 都能续写（无区分度），answer 才是
        # "谁能生成好这个回复"的回合粒度真实质量信号。None（默认）= 全序列（C16d 兼容）。
        answer_mask: Optional[torch.Tensor] = None,
        # ── T9: field_conditioning warm-up ──
        # True（默认）= round 2+ 注入 field_state（向后兼容）
        # False = round 2+ 也不注入（warm-up 阶段，neuron 独立学习）
        # field_state 仍会维护（累积），启用后注入的是学习到的场状态
        field_conditioning: bool = True,
        # ── §4.0c: Sparse Router ──
        # step: 当前训练步数（用于 Router warm-up），仅 use_sparse_router=True 时生效
        step: int = 0,
        # ── 缺口 M: 跨 vocab 联合训练 ──
        # vocab 不一致时，把各 neuron logits 用词库转译矩阵投影到 target 域空间再融合。
        # target_domain: batch 的目标域（如 "zh"），对应 batch_align_and_embed 的 domain_sp。
        # None（默认）= vocab 一致路径，无需投影（向后兼容）。
        target_domain: Optional[str] = None,
        # ── 路由信任覆盖（实验/上界诊断用）──
        # 覆盖 _confidence_routing_fusion 的 trust（softmax(scores/temp)），
        # 直接指定 per-neuron 信任系数。None（默认）= 正常 scores 校准路径（向后兼容）。
        trust_override: Optional[torch.Tensor] = None,
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

        # 快照隔离（训练与推理并发时，增删不影响正在进行的训练步）
        nmap = dict(self.neurons)
        active_ids = list(nmap.keys())
        N = len(active_ids)
        # §4.0c+d: 初始化 Router 缓存（每步重置）
        self._last_router_result = None
        self._router_active_ids = None
        self._router_hard_mask = None

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
        # C1: 调质不应期乘数（用于 round 2+ enter_refractory）
        neuromod_mult = 1.0
        if neuromodulator is not None:
            try:
                neuromod_mult = float(neuromodulator.get_refractory_multiplier())
            except Exception:
                neuromod_mult = 1.0

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
        round_quality_logits_all: Dict[str, torch.Tensor] = {}  # C15: round 1 预测质量 logit（全程保留）

        for round_num in range(1, n_rounds + 1):
            round_vecs_raw_new: Dict[str, torch.Tensor] = {}
            round_vecs_unified_new: Dict[str, torch.Tensor] = {}
            round_confidences_new: Dict[str, torch.Tensor] = {}  # C8: per-sample confidence
            round_score_vecs_new: Dict[str, torch.Tensor] = {}  # C12: 评分投影向量
            # C15: 预测质量 logit 只收集 round 1（写入循环外的 round_quality_logits_all，
            # 避免 round 2 重新初始化清空）

            # 构建 side_signals（round 2+ 才有，per-pair synaptic 投影）
            # 用原始 field_dim 维度的 vecs（excite_channels 在建立时按 pre.field_dim 注册）
            side_signals_per_neuron: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
            if round_num > 1 and round_vecs_raw:
                side_signals_per_neuron = {nid: {} for nid in active_ids}
                # §4.0d: per-sample 稀疏——post 神经元只接收该样本 top-K 的 pre 信号
                # 非 top-K 的 pre 在该样本的信号被 mask 为零（每样本激活不同组合）
                router_mask = self._router_hard_mask  # [B, N] or None
                for post_id in active_ids:
                    post_neuron = nmap[post_id]
                    for pre_id in active_ids:
                        if post_id == pre_id:
                            continue
                        if (pre_id in post_neuron.excite_channels or
                                pre_id in post_neuron.inhibit_channels):
                            pre_vec = round_vecs_raw[pre_id]  # [B, D]
                            if router_mask is not None:
                                pre_mask = router_mask[:, active_ids.index(pre_id)]  # [B]
                                side_signals_per_neuron[post_id][pre_id] = pre_vec * pre_mask.unsqueeze(-1)
                            else:
                                side_signals_per_neuron[post_id][pre_id] = pre_vec

            # 全可微前向（串行：batch 内已并行，neuron.forward 全可微）
            for nid in active_ids:
                emb = _get_emb(nid)

                # 跨规格反投影：field_state(unified) → neuron.field_dim
                # T9: field_conditioning=False 时（warm-up 阶段）不注入 field_state
                fs = field_state if field_conditioning else None
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

                result = nmap[nid].forward(emb, **kwargs)

                # 原始 field_vector（neuron.field_dim，用于 side_signals）
                vec_raw = result["field_vector"]  # [B, neuron.field_dim]
                # 跨规格正向投影：neuron.field_dim → unified（用于维护 field_state）
                vec_unified = self._project_vec(nid, vec_raw)  # [B, field.dim]
                # C8: 收集 per-sample confidence（attention entropy 驱动）
                confidence = result.get(
                    "field_confidence", torch.ones(vec_raw.shape[0], device=vec_raw.device)
                )

                round_vecs_raw_new[nid] = vec_raw
                round_vecs_unified_new[nid] = vec_unified
                round_confidences_new[nid] = confidence
                # C12: 收集评分投影向量
                if "score_vec" in result:
                    round_score_vecs_new[nid] = result["score_vec"]
                # C15: 收集预测质量 logit（round 1 独立前向，无场耦合）
                if round_num == 1 and "quality_logit" in result:
                    round_quality_logits_all[nid] = result["quality_logit"]
                if round_num == n_rounds:
                    final_logits[nid] = result["logits"]

            # 更新 field_state（可微加法，无 detach）
            # field_state = sum of (confidence * L2-normalized vecs) (unified 维度)
            all_vecs_norm = F.normalize(
                torch.stack([round_vecs_unified_new[nid] for nid in active_ids]),
                dim=-1,
            )  # [N, B, D]
            # C8: per-sample confidence 加权（高置信度 neuron 贡献更大）
            all_confidences = torch.stack([
                round_confidences_new[nid] for nid in active_ids
            ])  # [N, B]
            all_vecs_weighted = all_vecs_norm * all_confidences.unsqueeze(-1)  # [N, B, D]
            # §4.0d: per-sample 稀疏——只累加每样本 top-K 神经元的写入
            # 非 top-K 神经元在该样本的 round 2+ 写入被 mask 为零
            if self._router_hard_mask is not None and round_num > 1:
                mask_t = self._router_hard_mask.t().unsqueeze(-1)  # [N, B, 1]
                all_vecs_weighted = all_vecs_weighted * mask_t
            # C7: 空间场扩散（图拉普拉斯，近邻神经元信号互相扩散）
            if self.spatial_diffuser is not None:
                all_vecs_weighted = self.spatial_diffuser.diffuse(
                    all_vecs_weighted, active_ids=active_ids
                )
            field_state = all_vecs_weighted.sum(dim=0)  # [B, D]

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
                    if getattr(gamma_oscillator, "differentiable", False):
                        # C23-C：可微演化——最后一轮的可微 new_p 供 scores/场绑定
                        # （梯度经 new_p → dtheta → ω/K，打通 ω/K 梯度路径）；
                        # 状态推进由 kuramoto_step 内部 detach 完成
                        self._last_evolved_phasors = gamma_oscillator.evolve(
                            list(active_ids), coactivation=self.coaction,
                        )
                        gamma_oscillator.kuramoto_step(
                            active_ids=active_ids, coactivation=self.coaction,
                        )
                    elif hasattr(gamma_oscillator, "kuramoto_step"):
                        gamma_oscillator.kuramoto_step(
                            coupling_strength=0.05,
                            active_ids=active_ids,
                            coactivation=self.coaction,
                        )
                except Exception:
                    pass  # 振荡失败不影响训练主流程

            # ── §4.0c+d: Sparse Router 接入 ──
            # round 1 结束后，Router 选 per-sample top-K
            # round 2+ 用 per-sample mask 控制 side_signals/field_state 注入
            # （每样本只被其 top-K 神经元影响 = per-sample 稀疏信号流）
            # use_sparse_router=False 时跳过（向后兼容）
            # 注意：round 2+ 不重置缓存（保留 round 1 结果供融合阶段使用）
            if (
                self.use_sparse_router
                and self.sparse_router is not None
                and round_num == 1
                and n_rounds > 1
            ):
                router_result = self.sparse_router(
                    active_ids=active_ids,
                    round_vecs_unified=round_vecs_unified,
                    round_confidences=round_confidences_new,
                    round_score_vecs=round_score_vecs_new if round_score_vecs_new else None,
                    step=step,
                )
                # 缓存 router_result 供融合阶段 + round 2+ 使用
                self._last_router_result = router_result
                self._router_active_ids = list(active_ids)
                # per-sample hard_mask [B, N] 缓存（round 2+ 信号流 mask）
                self._router_hard_mask = router_result["hard_mask"]

        # ── 计算共振分（Leave-one-out cosine similarity，全可微）──
        # 用 unified 维度的 vecs（保证不同 field_dim 的 neuron 在同一空间比较）
        all_vecs_norm = F.normalize(
            torch.stack([round_vecs_unified[nid] for nid in active_ids]),
            dim=-1,
        )  # [N, B, D] 单位向量（方向）
        # C8: per-sample confidence 加权（高置信度 neuron 对场状态贡献更大）
        all_confidences = torch.stack([
            round_confidences_new[nid] for nid in active_ids
        ])  # [N, B]
        all_vecs_weighted = all_vecs_norm * all_confidences.unsqueeze(-1)  # [N, B, D]
        # C23-B（2026-08-08）：相位同步塑造训练场状态——同相群体写入增强、
        # 异相解绑（与推理 forward 场写入一致；让相位绑结成为可学习信号）
        if gamma_oscillator is not None:
            try:
                bs = float(getattr(gamma_oscillator, "binding_scale", 0.0))
                if bs != 0.0:
                    if getattr(gamma_oscillator, "differentiable", False):
                        # C23-C：可微绑定（用最后一轮可微演化相位 → ω/K 梯度路径）
                        ev_p = getattr(self, "_last_evolved_phasors", None)
                        if ev_p is not None and ev_p.shape[0] == len(active_ids):
                            bvec = gamma_oscillator.binding_tensor(
                                list(active_ids), coactivation=self.coaction,
                                phasors=ev_p,
                            ).to(all_vecs_weighted.device)
                        else:
                            bvec = gamma_oscillator.binding_tensor(
                                list(active_ids), coactivation=self.coaction,
                            ).to(all_vecs_weighted.device)
                    else:
                        # C23-B：标量绑定（GammaOscillator）
                        binding = gamma_oscillator.pairwise_binding(
                            list(active_ids), coactivation=self.coaction,
                        )
                        bvec = None
                        if binding:
                            bvec = torch.tensor(
                                [binding.get(nid, 0.0) for nid in active_ids],
                                dtype=torch.float32, device=all_vecs_weighted.device,
                            )
                    if bvec is not None:
                        all_vecs_weighted = all_vecs_weighted * (
                            1.0 + bs * bvec.unsqueeze(1).unsqueeze(2)
                        )  # [N, B, D]
            except Exception:
                pass
        # C7: 空间场扩散（评分时也用扩散后的 vectors，保持训练一致性）
        if self.spatial_diffuser is not None:
            all_vecs_weighted = self.spatial_diffuser.diffuse(
                all_vecs_weighted, active_ids=active_ids
            )
        field_state_full = all_vecs_weighted.sum(dim=0)  # [B, D] 加权场状态
        # 2026-08-07 fix（routing_loss 梯度泄漏）：field_state 含所有 neuron 的 vec，
        # 若全可微则 scores[n] 的梯度会流向其他 neuron——batch 轮转下每步 routing_loss
        # 都在同时"拉"所有 neuron 的 field vectors（code batch 抬 code、math batch 抬
        # math，方向互相拉锯），有效信号被稀释 80%（实测 RL→其他 8 neuron 梯度是
        # 自身 3.8 倍）。detach field_state，只保留自身 vec 的贡献（-all_vecs_weighted）：
        # scores[n] 梯度只流向自身 neuron，域判别语义 = "我"与"除我之外"的场一致性。
        loo_state = field_state_full.detach().unsqueeze(0) - all_vecs_weighted  # [N, B, D]

        # C12: 评分投影（score_dim 空间，与写入空间分离）
        # 让大神经元对场状态方向的主导被 field_score_proj 抵消，
        # 小神经元能获得公平的共振分。
        if self.field_score_proj is not None and round_score_vecs_new:
            # neuron 评分向量 [N, B, score_dim]（已归一化）
            all_score_vecs = torch.stack([
                round_score_vecs_new[nid] for nid in active_ids
            ])  # [N, B, score_dim]
            # 场状态投影到评分空间 [N, B, score_dim]
            loo_score = self.field_score_proj(loo_state)  # [N, B, score_dim]
            loo_score_norm = F.normalize(loo_score, dim=-1)
            # 评分 = cosine(score_vec, field_score_proj(loo_state))
            scores = (all_score_vecs * loo_score_norm).sum(dim=-1)  # [N, B]
        else:
            # 向后兼容：直接用 field_vector cosine
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
                bs = getattr(gamma_oscillator, "binding_scale", 0.0)
                if bs != 0.0:
                    if getattr(gamma_oscillator, "differentiable", False):
                        # C23-C（2026-08-08）：可微相位绑定（PhasorDynamics）——
                        # binding 用最后一轮可微演化相位（evolve 输出）：梯度经
                        # binding → new_p → dtheta → ω/K（ω/K 梯度路径打通），
                        # phasors 亦收到梯度（task_gradient_step 切向更新）
                        ev_p = getattr(self, "_last_evolved_phasors", None)
                        if ev_p is not None and ev_p.shape[0] == len(active_ids):
                            bvec = gamma_oscillator.binding_tensor(
                                list(active_ids), coactivation=self.coaction,
                                phasors=ev_p,
                            ).to(scores.device)
                        else:
                            bvec = gamma_oscillator.binding_tensor(
                                list(active_ids), coactivation=self.coaction,
                            ).to(scores.device)
                        scores = scores * (1.0 + bs * bvec)
                    else:
                        # C23（2026-08-08）：标量相位绑定（GammaOscillator）——
                        # 同相群体增强/异相解绑（相位从"对全局相位的标量门控"
                        # 升级为 neuron 之间的关系度量，共振本体化）
                        binding = gamma_oscillator.pairwise_binding(
                            list(active_ids), coactivation=self.coaction,
                        )
                        if binding:
                            bvec = torch.tensor(
                                [binding[nid] for nid in active_ids],
                                dtype=torch.float32, device=scores.device,
                            )
                            scores = scores * (1.0 + bs * bvec)
            except Exception:
                pass

        # ── 融合聚合 ──
        # 缺口 M: 跨 vocab 联合训练——vocab 不一致时，用词库转译矩阵把各 neuron
        # logits 投影到 target_domain 空间再融合（softmax 前线性投影，近似概率转移）。
        # vocab 一致时走原路径（零开销，向后兼容）。
        vocab_sizes = [final_logits[nid].shape[-1] for nid in active_ids]
        cross_vocab = len(set(vocab_sizes)) != 1
        if cross_vocab:
            if target_domain is None:
                raise RuntimeError(
                    f"[forward_train] 检测到跨 vocab（vocab_sizes="
                    f"{dict(zip(active_ids, vocab_sizes))}），需要传入 target_domain "
                    f"（对应 batch_align_and_embed 的 domain_sp 域）才能融合。"
                )
            all_logits = self._project_logits_to_target(
                final_logits, active_ids, target_domain
            )
        else:
            all_logits = torch.stack([final_logits[nid] for nid in active_ids])  # [N, B, L, V]

        # ── §4.0c: Sparse Router 融合（per-sample，STE 可微）──
        # use_sparse_router=True 且有 router_result 时，用 Router 的 final_weights 融合
        # 否则退化为原 soft/residual 融合（向后兼容）
        if self._last_router_result is not None and fusion_mode != "residual":
            # 从 Router final_weights [B, N_round1] 中取当前 active_ids 对应列
            router_final_weights = self._last_router_result["final_weights"]  # [B, N_round1]
            router_load_balance_loss = self._last_router_result["load_balance_loss"]
            # 当前 active_ids 是 round 1 后的并集，是 _router_active_ids 的子集
            col_indices = torch.tensor(
                [self._router_active_ids.index(nid) for nid in active_ids],
                device=router_final_weights.device,
            )
            weights_per_sample = router_final_weights.index_select(-1, col_indices)  # [B, N_active]
            # 重新归一化（取子集后和可能 < 1）
            weights_per_sample = weights_per_sample / (weights_per_sample.sum(dim=-1, keepdim=True) + 1e-8)
            # per-sample 融合：fused = Σ_bn weights[b,n] * logits[n]
            fused_logits = torch.einsum('bn,nblv->blv', weights_per_sample, all_logits)
            # batch 平均权重（用于返回和监控）
            weights = weights_per_sample.mean(dim=0)  # [N_active]
            # 负载均衡 loss 用 Router 的 Switch 风格 loss（替换原负熵）
            balance_loss = router_load_balance_loss
        elif fusion_mode == "residual" and N >= 2:
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
            if cross_vocab:
                # 缺口 M + 分工路由（2026-08-06 实验验证）：跨 vocab 时按位置置信度
                # 路由（softmax max-prob）。稀疏转译投影使域外 neuron logits 近均匀
                # （max-prob ~0.01-0.02），原生 neuron 保持尖锐（~0.6）→ 每个 token
                # 交给最自信的 neuron = 分工。域内：原生 neuron 全位置胜出（≈个体，
                # 无伤害）；跨域（zh 提问→code 输出）：zh token→zh、code token→code。
                native_list = [final_logits[nid] for nid in active_ids]
                # C15: 预测质量 logits [N]（round 1 独立前向，batch 平均）。
                # fusion 信任与 contrastive_loss 共用（监督 quality 对齐 NLL 排序）
                quality_logits_t = None
                if round_quality_logits_all:
                    quality_logits_t = torch.stack([
                        round_quality_logits_all[nid] for nid in active_ids
                    ]).mean(dim=1).squeeze(-1)  # [N]
                # C15: quality_logits 优先于 scores（预测质量 head，无 LOO 泄漏、无判别器尺度游戏）
                fused_logits, route_weights = self._confidence_routing_fusion(
                    all_logits, native_list, active_ids, target_domain,
                    scores=scores,
                    quality_logits=quality_logits_t,
                    trust_override=trust_override)  # trust_override: 上界诊断
                weights = route_weights  # [N] batch 平均路由权重（监控）
                # 路由权重由 logits 决定（非参数），balance_loss 梯度会反向干扰路由 → 置 0
                balance_loss = torch.zeros((), device=fused_logits.device)
            else:
                # soft 软加权融合（默认，全可微）
                weights = F.softmax(scores / temperature, dim=0)  # [N]
                fused_logits = torch.einsum('n,nblv->blv', weights, all_logits)
                balance_loss = -(weights * torch.log(weights + 1e-8)).sum()
                # C15: soft 模式也构造 quality_logits（contrastive_loss 监督用）
                quality_logits_t = None
                if round_quality_logits_all:
                    quality_logits_t = torch.stack([
                        round_quality_logits_all[nid] for nid in active_ids
                    ]).mean(dim=1).squeeze(-1)  # [N]

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

        # ── per-neuron NLL（供 C12 共振分对比 + §4.0d Router 对比约束共用）──
        # targets=None 时不计算（向后兼容）
        # C20（2026-08-08）：answer_mask 传入时只对 answer（回复）部分算回合级 NLL——
        # prompt 部分无区分度（输入即答案），回合粒度真实质量 = "谁能生成好这个回复"。
        per_neuron_nll: Optional[torch.Tensor] = None
        if targets is not None and N >= 2:
            # all_logits: [N, B, L, V], targets: [B, L]
            shift_logits = all_logits[:, :, :-1, :].contiguous()  # [N, B, L-1, V]
            shift_targets = targets[:, 1:].contiguous()            # [B, L-1]
            nll_all = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_targets.unsqueeze(0).expand(N, -1, -1).reshape(-1),
                reduction="none",
            ).view(N, -1)  # [N, B*(L-1)]
            if answer_mask is not None:
                # 只对 answer 位置求均值（回合级 NLL）；无 answer 位置时回退全序列
                am_shift = answer_mask[:, 1:].contiguous().reshape(-1).bool()  # [B*(L-1)]
                n_tok = int(am_shift.sum().item())
                if n_tok > 0:
                    per_neuron_nll = nll_all[:, am_shift].mean(dim=1)  # [N]
                else:
                    per_neuron_nll = nll_all.mean(dim=1)
            else:
                per_neuron_nll = nll_all.mean(dim=1)  # [N]（C16d 全序列口径）

        # ── C16b: per-neuron NLL z-score（2026-08-08，修复跨 neuron 不可比）──
        # 绝对 NLL 受 neuron 在 general 空间分布锐度主导（code 域=英文最匹配 →
        # NLL 全局最低 → C15 监督 one-hot 偏 code）。z-score = (NLL-μ)/σ（μ/σ 为
        # 该 neuron 自身 NLL 的 EMA 统计）把监督转为"相对自身水平"：code 在中文
        # 对话上 NLL 远超自身基线 → z 大 → 不再天然获胜；dialogue neuron 在自身
        # 域文本上 NLL 接近基线 → z 小 → 获高权重。warmup（count<WARMUP）用原始
        # NLL（统计未建立前避免噪声）。EMA 更新用 detach（纯统计，不参与梯度）。
        NLL_EMA_ALPHA = 0.05
        NLL_EMA_WARMUP = 20
        nll_z: Optional[torch.Tensor] = None
        if per_neuron_nll is not None:
            zs = []
            for i, nid in enumerate(active_ids):
                v = float(per_neuron_nll[i].detach())
                s = self._nll_ema.get(nid)
                if s is None:
                    s = {"mean": v, "ms": v * v, "count": 1.0}
                    self._nll_ema[nid] = s
                else:
                    s["count"] += 1.0
                    a = min(NLL_EMA_ALPHA, 1.0 / s["count"])  # 前期大步长收敛
                    s["mean"] = (1 - a) * s["mean"] + a * v
                    s["ms"] = (1 - a) * s["ms"] + a * v * v
                if s["count"] >= NLL_EMA_WARMUP:
                    var = max(s["ms"] - s["mean"] ** 2, 1e-4)
                    zs.append((per_neuron_nll[i] - s["mean"]) / (var ** 0.5))
                else:
                    zs.append(per_neuron_nll[i])
            nll_z = torch.stack(zs)  # [N]

        # ── C16d: gate + z-score 质量监督（2026-08-08，最终方案）──
        # C15 绝对 NLL 不可比 → code 独占（general 空间英文主导+code 分布锐利）；
        # C16b 纯 z-score → 转译 dialogue neuron（NLL 基线数百~上万）z 系统性负
        # → 抢英文文本；C16c LOO 融合增益 → base 融合被转译 neuron 抢位，边际贡献全负。
        # 最终：z-score（相对自身水平）解决 code 独占 + 绝对质量 gate（batch 最优
        # ×GATE_FACTOR 排除 NLL 基线巨大的转译 neuron）防止 dialogue 抢英文。
        # 验证（verify_c16b_contrastive，gate=50）：code→code ✓ math→math ✓
        # zh/dialogue→zh_aug ✓；en→math（en 域数据不足致 math 相对提升更大，信号语义正确）。
        GATE_FACTOR = 50.0
        nll_gated_z: Optional[torch.Tensor] = None
        if nll_z is not None and per_neuron_nll is not None:
            min_nll = per_neuron_nll.min()
            gate_ok = per_neuron_nll < min_nll * GATE_FACTOR  # [N] 绝对质量 gate
            nll_gated_z = nll_z.clone()
            nll_gated_z[~gate_ok] = 1e9  # 排除者给极高 z（softmax 权重→0）

        # ── C15: 预测质量对比约束（D 方案，2026-08-08）──
        # 目标：quality_head 输出对齐 per-neuron NLL 排序——"谁能预测好当前文本谁上"。
        # 替代 C12 的 field_score_proj 条件（从未启用：训练时 score_dim=None → contrastive 恒 0）
        # 和 C13/C14 的域标签判别（判别任务不对称 + 输入不可比 + 尺度游戏，三次失败）。
        # NLL 是客观预测质量（训练时可得），quality_logit 推理时直接可用（round 1 独立）。
        # actual 温度 1.0：softmax 分布对齐 ideal；logit 膨胀会让 KL 增大 → 天然防尺度游戏。
        # C16b（2026-08-08）：ideal 用 per-neuron z-score（相对质量）而非绝对 NLL。
        # C16d（2026-08-08）：z-score + 绝对质量 gate（最终方案，见上）。
        if quality_logits_t is not None:
            # actual: 预测质量 head 驱动的权重（温度 1.0）
            actual_weights = F.softmax(quality_logits_t / 1.0, dim=0)  # [N]
        if nll_gated_z is not None and quality_logits_t is not None:
            # ideal: gated z-score 低（相对自身水平预测更好且绝对质量达标）者获高权重
            ideal_weights = F.softmax(-nll_gated_z / 0.5, dim=0)  # tau=0.5 温度
            # KL(actual || ideal) = Σ actual * log(actual/ideal)
            contrastive_loss = (
                actual_weights * (actual_weights.clamp(min=1e-8).log() - ideal_weights.clamp(min=1e-8).log())
            ).sum()
        else:
            contrastive_loss = torch.tensor(0.0, device=weights.device)

        # ── §4.0d: Router 对比约束（让 Router 学"能力路由"）──
        # 无约束的 Router 只反映"响应强弱"（大神经元天然强响应 → 退化回大神经元主导）
        # 对比约束迫使 Router 学"谁在**当前样本**上预测最好"：
        # ideal = softmax(-nll/tau)（NLL 低的神经元获高路由权重）
        # actual = Router soft_weights，KL(actual || ideal) 对齐
        router_contrastive_loss = torch.tensor(0.0, device=weights.device)
        if (
            nll_z is not None
            and self._last_router_result is not None
        ):
            router_soft = self._last_router_result["soft_weights"]  # [B, N]
            actual_router = router_soft.mean(dim=0)  # [N] batch 平均
            router_ideal = F.softmax(-nll_z / 0.5, dim=0)  # [N]
            router_contrastive_loss = (
                actual_router * (actual_router.clamp(min=1e-8).log() - router_ideal.clamp(min=1e-8).log())
            ).sum()

        result: Dict[str, torch.Tensor] = {
            "fused_logits": fused_logits,
            "weights": weights,
            "scores": scores,
            "balance_loss": balance_loss,
            "diversity_loss": diversity_loss,
            "contrastive_loss": contrastive_loss,  # C12
            "router_contrastive_loss": router_contrastive_loss,  # §4.0d
            "per_neuron_nll": per_neuron_nll,  # C16b: 原始 NLL（诊断）
            "per_neuron_nll_z": nll_z,  # C16b: 标准化 z-score（诊断）
            "field_state": field_state_full,
            "n_rounds": n_rounds,
        }

        # C15: 预测质量 logits [N]（round 1 独立前向，batch 平均）
        # 缺失时（旧 ckpt 无 quality_head）为 None，向后兼容 scores 路径
        if round_quality_logits_all:
            ql = torch.stack([
                round_quality_logits_all[nid] for nid in active_ids
            ]).mean(dim=1)  # [N, 1] -> mean over batch -> [N, 1]
            result["quality_logits"] = ql.squeeze(-1)  # [N]
        else:
            result["quality_logits"] = None

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

    def _score_logit_fusion(
        self,
        all_logits: Dict[str, torch.Tensor],
        scores: Dict[str, float],
        result: dict,
        ref: torch.Tensor,
        temperature: float = 1.0,
    ) -> None:
        """缺口 N 修复（2026-08-06）：共振分 softmax 融合（与训练 forward_train 对齐）。

        状态（2026-08-07 收敛）：**主路径（同 vocab 推理）**——fusion_mode="soft"/"score"
        时的默认融合，generate/_generate_p7 默认走这里。

        训练时融合权重 = softmax(scores/temp)（C12 让共振分与 NLL 排序对齐），
        但推理默认 per_position（entropy 路由）是启发式，无视训练学的共振分，
        导致训练-推理不一致（训练分布 PPL 2.2 vs 推理 12.6，生成质量差）。

        此方法用 softmax(scores/temp) 生成融合权重，让推理直接复用训练学到的
        协作权重（side_channels 学到的"谁擅长什么"经 scores 体现）。
        """
        neuron_ids = list(all_logits.keys())
        score_vals = torch.tensor(
            [scores.get(nid, 0.0) for nid in neuron_ids],
            dtype=torch.float32, device=ref.device,
        )
        weights = F.softmax(score_vals / max(temperature, 1e-6), dim=0)  # [N]
        logits_stack = torch.stack([all_logits[nid] for nid in neuron_ids])  # [N, B, L, V]
        fused = torch.einsum('n,nblv->blv', weights, logits_stack)
        result["weighted_logits"] = fused
        result["weights"] = weights.detach().cpu().tolist()
        result["fusion_mode"] = "score"

    def _division_logit_fusion(
        self,
        all_logits: Dict[str, torch.Tensor],
        result: dict,
        ref: torch.Tensor,
        normalize: bool = False,
    ) -> None:
        """统一空间（同 vocab）max-prob 分工路由：per-position 硬路由。

        状态（2026-08-07 收敛）：实验/诊断——仅 fusion_mode="division"/"division_norm"
        显式选择时使用，非主路径（默认 soft → `_score_logit_fusion`；跨 vocab →
        `_confidence_routing_fusion`）。

        背景（2026-08-06）：共享 general lm_head 统一输出空间后，所有 neuron 原生
        输出就在 256K 空间——max-prob 天然尖锐（旧诊断 max-prob≈0.001 是静态稀疏
        投影到 256K 的置信度稀释，已随共享 head 消除）。per-position 硬路由直接把
        每个 token 位置交给 max-prob 最高的 neuron：

        - 域内文本：域 neuron 全位置胜出（≈个体，无伤害）
        - 跨域语义桥接（zh 理解 → code 表达）：中文位置 zh neuron、代码位置 code
          neuron——分工涌现，无需转译投影、无需 min(原生, 投影)（对比
          `_confidence_routing_fusion`，后者是跨 vocab 专用于防御投影极端值）

        已知局限（2026-08-06 验证）：跨 neuron 的 logits 尺度天然不可比（compact
        实测 code 平均 max-prob 0.566 vs zh 0.374）——裸 max-prob 分工让尖锐者抢走
        非自身域位置但预测错误，协作层未训练时 division PPL 劣于最强个体。正确校准
        需协作层训练（C12 对比约束让共振分对齐 NLL 排序 / Sparse Router 学习路由）。

        normalize（默认 False，供诊断）：per-neuron 相对归一化（max-prob ÷ 自身
        batch 平均），实测会过度平坦化导致域内分工对角消失，不作为默认。

        Args:
            all_logits: {nid: [B, L, V]}，全部同 vocab（V 相同，调用前 same_vocab 校验）
            result: 写入 weighted_logits / weights（路由权重监控）/ route_sel
            ref: 参考 tensor（device/dtype）
            normalize: 是否做 per-neuron 相对归一化（消除系统性尖锐度偏差，实验用）
        """
        neuron_ids = list(all_logits.keys())
        N = len(neuron_ids)
        # 逐 neuron max-prob（softmax 后取最大值）→ [N, B, L]
        probs = torch.stack([
            F.softmax(all_logits[nid], dim=-1).max(dim=-1).values
            for nid in neuron_ids
        ])
        if normalize:
            # 相对校准：除以每 neuron 的 batch 平均 max-prob（消除系统性更尖锐者主导）
            scale = probs.mean(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            conf = probs / scale
        else:
            conf = probs
        sel = conf.argmax(dim=0)  # [B, L] 每位置胜出 neuron
        w = F.one_hot(sel, num_classes=N).permute(2, 0, 1).float()  # [N, B, L]
        logits_stack = torch.stack([all_logits[nid] for nid in neuron_ids])  # [N, B, L, V]
        fused = (w.unsqueeze(-1) * logits_stack).sum(dim=0)  # [B, L, V]
        result["weighted_logits"] = fused
        result["weights"] = w.mean(dim=(1, 2)).detach().cpu().tolist()  # [N] 位置占比监控
        result["fusion_mode"] = "division"
        result["route_sel"] = sel.detach().cpu()  # 供诊断：per-position 路由归属
        result["route_nids"] = neuron_ids  # route_sel/weights 的 neuron 顺序（对应关系）

    def _confidence_routing_fusion(
        self,
        all_logits: torch.Tensor,
        native_logits: torch.Tensor,
        active_ids: List[str],
        target_domain: str,
        router_temperature: float = 0.15,
        scores: Optional[torch.Tensor] = None,  # [N] 协作层校准的共振分（per-sample 信任）
        quality_logits: Optional[torch.Tensor] = None,  # [N] C15 预测质量 logit（优先于 scores）
        trust_override: Optional[torch.Tensor] = None,  # [N] 实验：直接指定信任系数（覆盖 scores）
    ):
        """跨 vocab 分工融合：per-position **硬路由**，置信度 = min(原生, 投影后) max-prob。

        状态（2026-08-07 收敛）：**主路径（跨 vocab）**——训练 forward_train 与推理
        forward 的混合阵容（不同词表 neuron）融合统一走这里，含 trust_override 上界诊断。

        诊断（2026-08-06）：
        - 域外 neuron 的 logits 经稀疏转译投影到目标域后幅值极端（zh max|logit|=60828），
          软加权平均必然被垃圾幅值淹没（0.07×60000=4200 >> 原生 ±14）→ 需硬路由。
        - 仅用投影后 max-prob 路由不稳：投影极端值会在个别位置尖峰（max-prob→1.0），
          抢走原生 neuron 的位置 → 协作 loss 崩塌。
        - 仅用原生空间 max-prob 分离弱（code 数据上 code 0.586 vs en 0.407）。
        - **min(原生, 投影后)**：原生项 = 域门（域外 neuron 原生 max-prob 0.31-0.41，
          即使投影尖峰到 1.0 也被原生项封顶 < code 0.586）；投影项 = 目标空间信息量。
          域内 code 文本：code 0.586 全位置胜出 → 协作 ≈ 个体（无伤害）。

        scores 校准（2026-08-06 混合阵容诊断新增）：
        跨空间原生 max-prob 不可比——general 空间 zh neuron 高锐度（原生 max-prob
        0.999 vs code 0.005）系统性抢走 code/math/en 文本位置但预测错误（code 文本
        仅 67.5% 位置给 code neuron → 负 EMERGE）。用 side_channels 学到的共振分
        （scores，反映"协作层认为当前样本谁擅长"）作为 per-neuron 信任系数
        （softmax 归一化），与 per-position max-prob 相乘：信任系数放大协作层
        认可 neuron 的位置置信度 → 抢回正确位置。scores 顺序须与 active_ids 一致。

        梯度只流向被选中 neuron（分工训练：各域 batch 训练各自的原生 neuron）。

        Args:
            all_logits: [N, B, L, V_tgt]（已投影到 target_domain 空间）
            native_logits: List[[B, L, V_i]]（各 neuron 自身 vocab 空间原始 logits，
                逐 neuron 计算置信度，vocab 不同不 stack）
            active_ids: 参与共振的 neuron id
            scores: [N] 协作层共振分（None 时退化为纯 max-prob 路由，向后兼容）

        Returns:
            (fused_logits [B, L, V_tgt], route_weights [N] batch 平均路由权重)
        """
        probs_tgt = F.softmax(all_logits, dim=-1).max(dim=-1).values     # [N, B, L]
        # 原生空间 vocab 不同，逐 neuron 计算后 stack
        probs_nat = torch.stack([
            F.softmax(ln, dim=-1).max(dim=-1).values for ln in native_logits
        ])                                                               # [N, B, L]
        conf = torch.minimum(probs_nat, probs_tgt)                       # [N, B, L]
        if trust_override is not None:
            # 上界诊断：trust 直接给定（如"已知域"硬门控），跳过 scores 软校准
            trust = trust_override.to(conf.dtype).to(conf.device).clamp(min=1e-9)
        elif quality_logits is not None:
            # C15（2026-08-08，优先）：预测质量 head 输出——每个 neuron 学习"我对当前
            # 样本的预测质量"（round 1 独立前向，梯度无跨 neuron 泄漏），监督 = NLL 排序
            # （ensemble.contrastive_loss）。信任系数 = softmax(quality_logits/temp)。
            # 替代 C13/C14 域标签判别（判别任务不对称：math/code 是英文 → en 覆盖；
            # 判别器输入各自 round1 表征不可比；softmax CE 无尺度约束 → logit 膨胀作弊）
            # 和 scores（LOO cosine：梯度经 round 2 场条件化泄漏 + 强 neuron 主导场方向）。
            trust = F.softmax(quality_logits / max(router_temperature, 1e-6), dim=0)  # [N]
        elif scores is not None:
            # 协作层校准：信任系数 = softmax(scores/temp)（归一化 <1，乘后缩小但相对
            # 放大协作层认可 neuron）→ 修正跨空间原生 max-prob 系统性不可比
            trust = F.softmax(scores / max(router_temperature, 1e-6), dim=0)  # [N]
        else:
            trust = None
        if trust is not None:
            conf = conf * trust.view(-1, 1, 1)
        if self.maturity is not None:
            # C17（2026-08-08）静默期：新生 neuron（幼稚态）融合权重按成熟度压低——
            # get_resonance_weight 0.1→1.0 线性 ramp，配合 IntegrateEngine 无缝衔接
            # （人脑"沉默突触"：新生神经元初期不参与输出）。成熟 neuron 返回 1.0 不受影响。
            mw = torch.tensor(
                [self.maturity.get_resonance_weight(nid) for nid in active_ids],
                dtype=conf.dtype, device=conf.device,
            )
            conf = conf * mw.view(-1, 1, 1)
        sel = conf.argmax(dim=0)                                         # [B, L]
        w = F.one_hot(sel, num_classes=all_logits.shape[0]).permute(2, 0, 1).float()  # [N, B, L]
        fused = (w.unsqueeze(-1) * all_logits).sum(dim=0)                # [B, L, V_tgt]
        route_weights = w.mean(dim=(1, 2))                               # [N] 监控
        return fused, route_weights

    def _compute_per_position_weights(
        self,
        all_logits: Dict[str, torch.Tensor],
        vectors: Dict[str, torch.Tensor],
        scores: Dict[str, float],
        result: dict,
        ref: torch.Tensor,
    ) -> None:
        """Per-position routing (v2): logit-entropy weighting + complementarity.

        状态（2026-08-07 收敛）：旧/诊断——entropy 启发式路由，无训练对齐。
        仅 fusion_mode="per_position" 显式选择时使用；默认已改走 `_score_logit_fusion`。

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

    def _consensus_logit_fusion(
        self,
        all_logits: Dict[str, torch.Tensor],
        scores: Dict[str, float],
        result: dict,
        ref: torch.Tensor,
        temperature: float = 1.0,
    ) -> None:
        """R3: 共识投票融合（consensus voting fusion）。

        状态（2026-08-07 收敛）：实验保留——仅 fusion_mode="consensus" 显式选择时使用，
        非主路径（默认 soft → `_score_logit_fusion`）。

        三种融合模式的精神：
        - per_position: 每位置独立选最自信神经元（熵路由，局部决策）
        - residual: 族长主导 + 其他修正（层级预测，强领导）
        - consensus: 多神经元 top-k 预测一致性加权（投票共识，集体智慧）

        机制：
        1. 每个神经元独立预测 top-k token（k=consensus_k，默认 5）
        2. 计算每位置的共识度：多少神经元的 top-k 包含同一 token
        3. 高共识 token 获得权重加成（多神经元同意 → 更可信）
        4. 低共识 token 回退到共振分加权平均（无共识时用传统融合）

        上限提升：
        - 传统加权平均受大神经元主导（即使错误也会被选中）
        - 共识投票让"多神经元都认同的 token"获得优先权
        - 即使小神经元权重低，只要它与其他神经元在某个 token 上达成共识
          就能提升该 token 的最终权重（小神经元的话语权提升）

        与 C12（评分投影）+ R1（共振分数软路由）协同：
        - C12 让分数可比 → 共识投票的权重分配更公平
        - R1 让 top-k 神经元激活 → 共识投票的参与者都是相关神经元
        - R3 让共识 token 获得加成 → 集体智慧浮现

        Args:
            all_logits: {nid: [B, L, V]} 所有激活神经元的 logits（同 vocab）
            scores: {nid: float} 最终共振分
            result: forward() 的 result dict，写入 weighted_logits 和 final_weights
            ref: 参考 tensor（device 信息）
            temperature: softmax 温度
            consensus_k: 每个神经元投票的 top-k token 数（默认 5）
        """
        neuron_ids = list(all_logits.keys())
        n_neurons = len(neuron_ids)
        if n_neurons < 2:
            result["weighted_logits"] = all_logits[neuron_ids[0]]
            result["final_weights"] = {neuron_ids[0]: 1.0}
            return

        # 堆叠 logits: [N, B, L, V]
        logits_stack = torch.stack([all_logits[nid] for nid in neuron_ids], dim=0)
        N, B, L, V = logits_stack.shape

        # 1. 共振分 softmax 权重（基础权重）
        score_tensor = torch.tensor(
            [float(scores.get(nid, 0.0)) for nid in neuron_ids],
            device=ref.device, dtype=logits_stack.dtype,
        )  # [N]
        base_weights = F.softmax(score_tensor / temperature, dim=0)  # [N]

        # 2. 每神经元 top-k 预测（投票）
        consensus_k = getattr(self, "consensus_k", 5)
        consensus_k = min(consensus_k, V)
        # top_k_indices: [N, B, L, k]
        _, top_k_indices = logits_stack.topk(consensus_k, dim=-1)

        # 3. 计算共识度：对每个位置 (B,L)，统计每个 token 被多少神经元的 top-k 包含
        # 用 one-hot 编码: [N, B, L, k] → scatter 到 [N, B, L, V]
        # 然后按 N 维求和得到共识票数 [B, L, V]
        one_hot = torch.zeros(N, B, L, V, device=ref.device, dtype=logits_stack.dtype)
        one_hot.scatter_(-1, top_k_indices, 1.0)
        consensus_votes = one_hot.sum(dim=0)  # [B, L, V]，值域 [0, N]

        # 4. 共识加成因子：共识度高 → 权重提升
        # consensus_factor = 1 + α × (votes / N)，α=0.5（温和加成）
        # - 全员同意（votes=N）→ factor=1.5（权重提升 50%）
        # - 无共识（votes=1）→ factor=1.0（无加成，回退到基础权重）
        # - votes=0（不在任何 top-k）→ factor=1.0（无加成，但不会被完全压制）
        consensus_alpha = getattr(self, "consensus_alpha", 0.5)
        consensus_factor = 1.0 + consensus_alpha * (consensus_votes / N)  # [B, L, V]

        # 5. 基础加权 logits: [B, L, V] = Σ_n w_n × logits_n
        base_fused = torch.einsum('n,nblv->blv', base_weights, logits_stack)

        # 6. 应用共识加成：fused = base_fused × consensus_factor
        # 注意：这是乘性加成（高共识 token 的 logit 被放大）
        # 而非加性（避免低共识 token 被完全压制，保留多样性）
        fused_logits = base_fused * consensus_factor

        result["weighted_logits"] = fused_logits
        result["final_weights"] = {nid: float(w.item()) for nid, w in zip(neuron_ids, base_weights)}
        result["consensus_votes"] = consensus_votes  # [B, L, V]，供调试/可视化
        result["fusion_mode"] = "consensus"

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
