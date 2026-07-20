"""ThalamicRouter - 基于教师模型 hidden state 的输入路由器。

人脑参考：
  皮层定位 + 丘脑闸门 - 信号先到丘脑，丘脑决定送哪个皮层区域处理。
  信号进入 neuron 之前就决定路由，错误 neuron 根本不参与 forward。

核心思想：
  1. 蒸馏后，用教师模型对本域数据 forward 取平均 hidden state -> domain prototype
  2. 推理时，教师模型计算当前输入的 hidden state
  3. 与所有 prototype 算余弦相似度 -> softmax 路由
  4. 只让匹配域的 neuron forward（hard/soft 路由可配置）

与 DomainRouter 的关键区别：
  - DomainRouter 让所有 neuron 都 forward 再用 field_vector 相似度加权
  - ThalamicRouter 在 forward 之前就决定路由，过拟合的 neuron 根本不参与
  - ThalamicRouter 用教师 hidden state（客观），不用 neuron 自产的 field_vector

P6-2 扩展：脱教师推理
  原版依赖 teacher_model forward 算 hidden state，推理时必须跑 1.5B 教师
  P6-2: 用 standalone_embedding 直接 lookup 算 embedding 均值作为输入表征
  原型计算保持不变（仍用 teacher，因为蒸馏时教师可用）
  路由切换为 embedding-based（input_ids -> embedding -> mean pool -> 相似度）

文件：
  - data/distill/thalamic_prototypes.pt: {neuron_id: [hidden_dim] tensor}
  - data/distill/thalamic_prototypes_embed.pt: P6-2 embedding-based prototypes
"""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Callable
from collections import deque


class ThalamicRouter:
    """基于教师 hidden state × domain prototype 相似度的路由器。

    Usage:
        # 1. 蒸馏后计算 prototypes
        router = ThalamicRouter(hidden_dim=2048, temperature=0.1)
        router.compute_prototypes(teacher_model, neurons, domain_data, extract_fn, device)

        # 2. 推理时路由
        weights, top_nids = router.route(input_hidden)
        # weights: {nid: float}, top_nids: List[str]（按权重排序）

        # 3. 只让 top_nids 中的 neuron forward
    """

    def __init__(
        self,
        hidden_dim: int,
        temperature: float = 0.1,
        hard_route_threshold: float = 0.7,
        soft_route_threshold: float = 0.4,
        unknown_buffer_size: int = 50,
        pooling: str = "mean",
    ):
        """
        Args:
            hidden_dim: 教师 hidden state 维度（如 2048）
            temperature: softmax 温度，越低路由越 sharp
            hard_route_threshold: 相似度 > 此值时 hard route 到 top-1
            soft_route_threshold: 相似度 > 此值但 < hard 时 top-2 加权；低于此值触发未知
            unknown_buffer_size: 未知输入累积 buffer 大小（Phase 5.2 用）
            pooling: prototype 计算时的池化方式
                     "mean" = 对所有 token 位置平均（更稳定）
                     "last" = 只取最后 token（默认旧行为）
        """
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.hard_route_threshold = hard_route_threshold
        self.soft_route_threshold = soft_route_threshold
        self.unknown_buffer_size = unknown_buffer_size
        self.pooling = pooling

        # prototypes[nid] = [hidden_dim] 归一化向量（旧路径：teacher hidden state）
        self.prototypes: Dict[str, torch.Tensor] = {}
        # P6-2: prototypes_embed[nid] = [embed_dim] 归一化向量（脱教师路径）
        self.prototypes_embed: Dict[str, torch.Tensor] = {}
        # neuron 元信息（domain 名等，便于日志）
        self.neuron_meta: Dict[str, Dict] = {}
        # 未知输入 buffer（Phase 5.2 用）
        self.unknown_buffer: deque = deque(maxlen=unknown_buffer_size)
        # 新 neuron 学徒期权重（Phase 5.3 用，默认 1.0）
        self.routing_weights: Dict[str, float] = {}

    def register_domain(
        self,
        neuron_id: str,
        prototype: torch.Tensor,
        meta: Optional[Dict] = None,
        routing_weight: float = 1.0,
    ) -> None:
        """注册新 neuron 的 prototype（Phase 5.2 神经新生时调用）。

        Args:
            neuron_id: 神经元 ID
            prototype: [hidden_dim] 教师 hidden state 平均向量（会自动归一化）
            meta: 元信息（如 {'domain': 'zh'}）
            routing_weight: 学徒期权重（Phase 5.3，默认 1.0 即完整权重）
        """
        if prototype.dim() > 1:
            prototype = prototype.mean(dim=0)
        prototype = prototype / (prototype.norm() + 1e-8)
        self.prototypes[neuron_id] = prototype.cpu()
        self.neuron_meta[neuron_id] = meta or {}
        self.routing_weights[neuron_id] = routing_weight
        print(f"[ThalamicRouter] Registered {neuron_id} (meta={meta}, weight={routing_weight})")

    def compute_prototypes(
        self,
        teacher_model,
        neurons: Dict,  # Dict[str, ResonanceNeuron]（只用 nid 和 domain 名）
        domain_data: Dict[str, torch.Tensor],  # {nid: [N, L] token IDs}
        extract_hidden_fn,  # callable(teacher, input_ids) -> [B, L, hidden_dim]
        device: str = "cpu",
        batch_size: int = 4,
        max_samples: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """计算每个 neuron 的 domain prototype（基于教师 hidden state）。

        对每个 neuron：
          1. 取本域数据前 max_samples 条
          2. 用教师模型 forward 取最后一层 hidden state
          3. 对所有样本的 last token hidden state 平均 -> prototype
          4. L2 归一化

        Args:
            teacher_model: 教师模型（已加载到 device）
            neurons: {nid: ResonanceNeuron}（用于获取 domain 信息）
            domain_data: {nid: [N, L] tensor}
            extract_hidden_fn: 输入 (teacher, input_ids) 返回 [B, L, hidden_dim]
            device: 计算设备
            batch_size: 批大小
            max_samples: 每个域最多用多少样本

        Returns:
            {nid: [hidden_dim] normalized tensor}
        """
        self.prototypes = {}
        print(f"[ThalamicRouter] Computing prototypes (hidden_dim={self.hidden_dim})...")

        teacher_model.eval()
        for nid, neuron in neurons.items():
            if nid not in domain_data:
                print(f"  [{nid}] no domain data, skip")
                continue

            data = domain_data[nid]
            if data.shape[0] == 0:
                continue
            n = min(data.shape[0], max_samples)
            data = data[:n].to(device)

            proto_sum = torch.zeros(self.hidden_dim, device=device)
            count = 0

            with torch.no_grad():
                for i in range(0, n, batch_size):
                    batch = data[i:i + batch_size]
                    hidden = extract_hidden_fn(teacher_model, batch)  # [B, L, H]
                    # 池化：mean（对所有位置平均，更稳定）或 last（只取最后 token）
                    if self.pooling == "mean":
                        pooled = hidden.mean(dim=1)  # [B, H]
                    else:  # "last"
                        pooled = hidden[:, -1, :]  # [B, H]
                    proto_sum += pooled.sum(dim=0)
                    count += pooled.shape[0]

            if count > 0:
                prototype = proto_sum / count
                prototype = prototype / (prototype.norm() + 1e-8)
                self.prototypes[nid] = prototype.cpu()
                # 元信息：尝试从 neuron.config 获取 domain
                meta = {}
                if hasattr(neuron, 'config') and hasattr(neuron.config, 'spec'):
                    meta['spec'] = neuron.config.spec
                self.neuron_meta[nid] = meta
                self.routing_weights[nid] = 1.0
                print(f"  [{nid}] prototype from {count} samples, norm={prototype.norm().item():.4f}")

        return self.prototypes

    # ==================== P6-2: embedding-based 路由（脱教师） ====================

    def compute_prototypes_from_embedding(
        self,
        standalone_embedding,
        neurons: Dict,
        domain_data: Dict[str, torch.Tensor],
        device: str = "cpu",
        batch_size: int = 8,
        max_samples: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """P6-2: 用 standalone_embedding 算 prototype（脱教师）。

        与 compute_prototypes 等价，但用 embedding lookup + mean pool 替代
        teacher_model.forward。原型维度 = embed_dim（如 512），不是 hidden_dim。

        保存到 self.prototypes_embed（与 self.prototypes 分离），
        route_by_embedding 用这个；route（旧路径）仍用 self.prototypes。

        Args:
            standalone_embedding: StandaloneEmbedding 实例
            neurons: {nid: ResonanceNeuron}
            domain_data: {nid: [N, L] token IDs}
            device: 计算设备
            batch_size: 批大小
            max_samples: 每个域最多用多少样本

        Returns:
            {nid: [embed_dim] normalized tensor}
        """
        embed_dim = standalone_embedding.embed_dim
        self.prototypes_embed: Dict[str, torch.Tensor] = {}
        print(f"[ThalamicRouter] P6-2 computing embedding-based prototypes "
              f"(embed_dim={embed_dim})...")

        standalone_embedding = standalone_embedding.to(device)
        standalone_embedding.eval()

        for nid, neuron in neurons.items():
            if nid not in domain_data:
                continue
            data = domain_data[nid]
            if data.shape[0] == 0:
                continue
            n = min(data.shape[0], max_samples)
            data = data[:n].to(device)

            proto_sum = torch.zeros(embed_dim, device=device)
            count = 0

            with torch.no_grad():
                for i in range(0, n, batch_size):
                    batch = data[i:i + batch_size]
                    emb = standalone_embedding(batch)  # [B, L, embed_dim]
                    if self.pooling == "mean":
                        pooled = emb.mean(dim=1)  # [B, embed_dim]
                    else:
                        pooled = emb[:, -1, :]
                    proto_sum += pooled.sum(dim=0)
                    count += pooled.shape[0]

            if count > 0:
                prototype = proto_sum / count
                prototype = prototype / (prototype.norm() + 1e-8)
                self.prototypes_embed[nid] = prototype.cpu()
                print(f"  [{nid}] embed prototype from {count} samples, "
                      f"norm={prototype.norm().item():.4f}")

        # 同步元信息（如果 prototypes 还没建过）
        if not self.neuron_meta:
            for nid, neuron in neurons.items():
                meta = {}
                if hasattr(neuron, 'config') and hasattr(neuron.config, 'spec'):
                    meta['spec'] = neuron.config.spec
                self.neuron_meta[nid] = meta
                self.routing_weights.setdefault(nid, 1.0)

        return self.prototypes_embed

    def route_by_embedding(
        self,
        input_ids: torch.Tensor,
        standalone_embedding,
    ) -> Tuple[Dict[str, float], List[str]]:
        """P6-2: 用 standalone_embedding 算输入表征并路由（脱教师）。

        流程：
          1. standalone_embedding(input_ids) -> [B, L, embed_dim]
          2. mean pool over L -> [B, embed_dim]，取 batch_idx=0
          3. L2 归一化
          4. 与 self.prototypes_embed[nid] 算余弦相似度
          5. softmax（带 temperature + routing_weight）-> 路由权重

        Args:
            input_ids: [B, L] token IDs
            standalone_embedding: StandaloneEmbedding 实例

        Returns:
            weights: {nid: float} 归一化权重（和为 1）
            top_nids: List[str] 按权重降序排列
        """
        protos = getattr(self, 'prototypes_embed', None)
        if not protos:
            return {}, []

        device = input_ids.device
        with torch.no_grad():
            emb = standalone_embedding(input_ids.to(device))  # [B, L, embed_dim]
            if self.pooling == "mean":
                pooled = emb.mean(dim=1)  # [B, embed_dim]
            else:
                pooled = emb[:, -1, :]
            input_vec = pooled[0]  # [embed_dim]
            input_vec = input_vec / (input_vec.norm() + 1e-8)

        # 与所有 prototype 算余弦相似度（prototype 已归一化）
        nids = list(protos.keys())
        sim_list = []
        for nid in nids:
            proto = protos[nid].to(device)
            sim = torch.dot(input_vec, proto).item()
            sim_list.append(sim)

        sim_tensor = torch.tensor(sim_list, dtype=torch.float32)

        # 应用学徒期 routing_weight
        weight_scales = torch.tensor(
            [self.routing_weights.get(nid, 1.0) for nid in nids],
            dtype=torch.float32,
        )
        scaled = sim_tensor * weight_scales / self.temperature
        weights_tensor = F.softmax(scaled, dim=0)

        weights = {nid: float(weights_tensor[i].item()) for i, nid in enumerate(nids)}
        top_nids = sorted(nids, key=lambda n: weights[n], reverse=True)

        # 未知域检测：把 input_vec（cpu）入 buffer
        max_sim = max(sim_list) if sim_list else 0.0
        if max_sim < self.soft_route_threshold:
            self.unknown_buffer.append(input_vec.cpu())

        return weights, top_nids

    def route_top_k_by_embedding(
        self,
        input_ids: torch.Tensor,
        standalone_embedding,
        k: int = 2,
    ) -> Tuple[Dict[str, float], List[str]]:
        """P6-2: embedding-based 的 top-K 路由。

        Args:
            input_ids: [B, L] token IDs
            standalone_embedding: StandaloneEmbedding 实例
            k: 返回 top-K 个 neuron

        Returns:
            weights: {nid: float} 仅 top-K 有值，重新归一化
            top_nids: List[str] top-K 个 nid（按权重降序）
        """
        full_weights, all_nids = self.route_by_embedding(input_ids, standalone_embedding)
        if not full_weights:
            return {}, []

        top_nids = all_nids[:k]
        top_weights_sum = sum(full_weights[n] for n in top_nids)
        if top_weights_sum < 1e-8:
            uniform = 1.0 / len(top_nids)
            return {n: uniform for n in top_nids}, top_nids
        weights = {n: full_weights[n] / top_weights_sum for n in top_nids}
        return weights, top_nids

    def get_routing_decision_by_embedding(
        self,
        input_ids: torch.Tensor,
        standalone_embedding,
    ) -> Dict:
        """P6-2: embedding-based 路由决策（用于日志/诊断）。

        Returns:
            {
                'weights': {nid: float},
                'top_nids': List[str],
                'max_sim': float,
                'is_unknown': bool,
                'strategy': 'hard' | 'soft' | 'unknown',
                'similarities': {nid: sim},
            }
        """
        protos = getattr(self, 'prototypes_embed', None)
        if not protos:
            return {
                'weights': {}, 'top_nids': [], 'max_sim': 0.0,
                'is_unknown': True, 'strategy': 'unknown',
                'similarities': {},
            }

        weights, top_nids = self.route_by_embedding(input_ids, standalone_embedding)

        # 重新算相似度（用于判断 strategy）
        device = input_ids.device
        with torch.no_grad():
            emb = standalone_embedding(input_ids.to(device))
            pooled = emb.mean(dim=1) if self.pooling == "mean" else emb[:, -1, :]
            input_vec = pooled[0]
            input_vec = input_vec / (input_vec.norm() + 1e-8)
        sims = {}
        for nid in top_nids:
            proto = protos[nid].to(device)
            sims[nid] = torch.dot(input_vec, proto).item()
        max_sim = max(sims.values()) if sims else 0.0

        if max_sim >= self.hard_route_threshold:
            strategy = 'hard'
            is_unknown = False
        elif max_sim >= self.soft_route_threshold:
            strategy = 'soft'
            is_unknown = False
        else:
            strategy = 'unknown'
            is_unknown = True

        return {
            'weights': weights,
            'top_nids': top_nids,
            'max_sim': max_sim,
            'is_unknown': is_unknown,
            'strategy': strategy,
            'similarities': sims,
        }

    # ─── P0-3: 基于 encoder hidden state 的路由（P6-8 训练成果上线）───

    def route_by_hidden(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[Dict[str, float], List[str]]:
        """P0-3: 用预计算的 hidden state（来自 SharedContextEncoder）路由。

        与 route_by_embedding 的区别：
        - route_by_embedding 接受 input_ids + standalone_embedding，内部 lookup（无上下文）
        - route_by_hidden 接受预计算的 hidden_states（有上下文感知，P6-8 训练成果）

        prototypes_embed 必须也是用同一个 encoder.encode() 算的（语义一致）。

        Args:
            hidden_states: [B, L, embed_dim] 来自 encoder.encode()

        Returns:
            weights: {nid: float} 归一化权重（和为 1）
            top_nids: List[str] 按权重降序排列
        """
        protos = getattr(self, 'prototypes_embed', None)
        if not protos:
            return {}, []

        device = hidden_states.device
        with torch.no_grad():
            if self.pooling == "mean":
                pooled = hidden_states.mean(dim=1)  # [B, embed_dim]
            else:
                pooled = hidden_states[:, -1, :]
            input_vec = pooled[0]  # [embed_dim]
            input_vec = input_vec / (input_vec.norm() + 1e-8)

        # 与所有 prototype 算余弦相似度
        nids = list(protos.keys())
        sim_list = []
        for nid in nids:
            proto = protos[nid].to(device)
            sim = torch.dot(input_vec, proto).item()
            sim_list.append(sim)

        sim_tensor = torch.tensor(sim_list, dtype=torch.float32)

        weight_scales = torch.tensor(
            [self.routing_weights.get(nid, 1.0) for nid in nids],
            dtype=torch.float32,
        )
        scaled = sim_tensor * weight_scales / self.temperature
        weights_tensor = F.softmax(scaled, dim=0)

        weights = {nid: float(weights_tensor[i].item()) for i, nid in enumerate(nids)}
        top_nids = sorted(nids, key=lambda n: weights[n], reverse=True)

        max_sim = max(sim_list) if sim_list else 0.0
        if max_sim < self.soft_route_threshold:
            self.unknown_buffer.append(input_vec.cpu())

        return weights, top_nids

    def route_top_k_by_hidden(
        self,
        hidden_states: torch.Tensor,
        k: int = 2,
    ) -> Tuple[Dict[str, float], List[str]]:
        """P0-3: hidden-state-based 的 top-K 路由."""
        full_weights, all_nids = self.route_by_hidden(hidden_states)
        if not full_weights:
            return {}, []
        top_nids = all_nids[:k]
        top_weights_sum = sum(full_weights[n] for n in top_nids)
        if top_weights_sum < 1e-8:
            uniform = 1.0 / len(top_nids)
            return {n: uniform for n in top_nids}, top_nids
        weights = {n: full_weights[n] / top_weights_sum for n in top_nids}
        return weights, top_nids

    def get_routing_decision_by_hidden(
        self,
        hidden_states: torch.Tensor,
    ) -> Dict:
        """P0-3: hidden-state-based 路由决策（诊断用）."""
        protos = getattr(self, 'prototypes_embed', None)
        if not protos:
            return {
                'weights': {}, 'top_nids': [], 'max_sim': 0.0,
                'is_unknown': True, 'strategy': 'unknown',
                'similarities': {},
            }

        weights, top_nids = self.route_by_hidden(hidden_states)

        device = hidden_states.device
        with torch.no_grad():
            pooled = hidden_states.mean(dim=1) if self.pooling == "mean" else hidden_states[:, -1, :]
            input_vec = pooled[0]
            input_vec = input_vec / (input_vec.norm() + 1e-8)
        sims = {}
        for nid in top_nids:
            proto = protos[nid].to(device)
            sims[nid] = torch.dot(input_vec, proto).item()
        max_sim = max(sims.values()) if sims else 0.0

        if max_sim >= self.hard_route_threshold:
            strategy = 'hard'
            is_unknown = False
        elif max_sim >= self.soft_route_threshold:
            strategy = 'soft'
            is_unknown = False
        else:
            strategy = 'unknown'
            is_unknown = True

        return {
            'weights': weights,
            'top_nids': top_nids,
            'max_sim': max_sim,
            'is_unknown': is_unknown,
            'strategy': strategy,
            'similarities': sims,
        }

    # ==================== 旧路径：teacher hidden state 路由（保留兼容） ====================

    def route(
        self,
        input_hidden: torch.Tensor,
    ) -> Tuple[Dict[str, float], List[str]]:
        """根据当前输入的 hidden state 与 prototype 的相似度计算路由权重。

        路由策略：
          - 相似度 > hard_route_threshold: hard route 到 top-1
          - soft_route_threshold <= 相似度 < hard: top-2 加权
          - 相似度 < soft_route_threshold: 触发未知（权重仍返回但记录到 buffer）

        Args:
            input_hidden: [hidden_dim] 或 [B, hidden_dim] 或 [B, L, hidden_dim]
                          - 3D: 自动 mean pooling over L
                          - 2D: 取 batch_idx=0
                          - 1D: 直接使用

        Returns:
            weights: {nid: float} 归一化权重（和为 1）
            top_nids: List[str] 按权重降序排列（Phase 5.1 ensemble 用 top-K）
        """
        if not self.prototypes:
            return {}, []

        # 处理不同维度的输入
        if input_hidden.dim() == 3:
            # [B, L, H] -> mean pool over L -> [B, H]
            input_hidden = input_hidden.mean(dim=1)
        if input_hidden.dim() == 2:
            input_hidden = input_hidden[0]  # [hidden_dim]
        input_hidden = input_hidden / (input_hidden.norm() + 1e-8)

        # 计算与所有 prototype 的余弦相似度
        nids = list(self.prototypes.keys())
        sim_list = []
        for nid in nids:
            proto = self.prototypes[nid].to(input_hidden.device)
            sim = torch.dot(input_hidden, proto).item()
            sim_list.append(sim)

        sim_tensor = torch.tensor(sim_list, dtype=torch.float32)

        # 应用学徒期 routing_weight（Phase 5.3 用）
        weight_scales = torch.tensor(
            [self.routing_weights.get(nid, 1.0) for nid in nids],
            dtype=torch.float32,
        )
        # softmax with temperature and routing_weight scaling
        scaled = sim_tensor * weight_scales / self.temperature
        weights_tensor = F.softmax(scaled, dim=0)

        weights = {nid: float(weights_tensor[i].item()) for i, nid in enumerate(nids)}
        # 按权重降序
        top_nids = sorted(nids, key=lambda n: weights[n], reverse=True)

        # 检查是否触发未知域（Phase 5.2 用）
        max_sim = max(sim_list) if sim_list else 0.0
        if max_sim < self.soft_route_threshold:
            self.unknown_buffer.append(input_hidden.cpu())
            # 注意：Phase 5.1 仍返回权重，让 ensemble 做最佳猜测
            # Phase 5.2 会检查 buffer 是否累积到阈值，触发神经新生

        return weights, top_nids

    def route_top_k(
        self,
        input_hidden: torch.Tensor,
        k: int = 2,
    ) -> Tuple[Dict[str, float], List[str]]:
        """只返回 top-K 个 neuron 的权重（其余归零，避免污染）。

        Args:
            input_hidden: [hidden_dim] 教师 hidden state
            k: 返回 top-K 个 neuron

        Returns:
            weights: {nid: float} 仅 top-K 有值，重新归一化
            top_nids: List[str] top-K 个 nid（按权重降序）
        """
        full_weights, all_nids = self.route(input_hidden)
        if not full_weights:
            return {}, []

        top_nids = all_nids[:k]
        # 重新归一化 top-K 权重
        top_weights_sum = sum(full_weights[n] for n in top_nids)
        if top_weights_sum < 1e-8:
            # 退化情况：均匀分配
            uniform = 1.0 / len(top_nids)
            return {n: uniform for n in top_nids}, top_nids
        weights = {n: full_weights[n] / top_weights_sum for n in top_nids}
        return weights, top_nids

    def get_routing_decision(
        self,
        input_hidden: torch.Tensor,
    ) -> Dict:
        """获取完整路由决策信息（用于日志/诊断）。

        Returns:
            {
                'weights': {nid: float},
                'top_nids': List[str],
                'max_sim': float,
                'is_unknown': bool,  # 是否触发未知域
                'strategy': 'hard' | 'soft' | 'unknown',
            }
        """
        if not self.prototypes:
            return {
                'weights': {}, 'top_nids': [], 'max_sim': 0.0,
                'is_unknown': True, 'strategy': 'unknown',
            }

        weights, top_nids = self.route(input_hidden)

        # 重新计算 max_sim（因为 route 里已计算过）
        if input_hidden.dim() == 3:
            input_hidden = input_hidden.mean(dim=1)
        if input_hidden.dim() == 2:
            input_hidden = input_hidden[0]
        input_norm = input_hidden / (input_hidden.norm() + 1e-8)
        sims = []
        for nid in top_nids:
            proto = self.prototypes[nid].to(input_hidden.device)
            sims.append(torch.dot(input_norm, proto).item())
        max_sim = max(sims) if sims else 0.0

        if max_sim >= self.hard_route_threshold:
            strategy = 'hard'
            is_unknown = False
        elif max_sim >= self.soft_route_threshold:
            strategy = 'soft'
            is_unknown = False
        else:
            strategy = 'unknown'
            is_unknown = True

        return {
            'weights': weights,
            'top_nids': top_nids,
            'max_sim': max_sim,
            'is_unknown': is_unknown,
            'strategy': strategy,
            'similarities': dict(zip(top_nids, sims)),
        }

    def check_unknown_buffer(self) -> Tuple[bool, int]:
        """检查未知输入 buffer 是否累积到阈值（Phase 5.2 用）。

        Returns:
            (should_trigger_neurogenesis, buffer_size)
        """
        return (len(self.unknown_buffer) >= self.unknown_buffer_size,
                len(self.unknown_buffer))

    def sync_apprentice_weights(self, maturity_tracker) -> Dict[str, float]:
        """Phase 5.3: 从 MaturityTracker 同步学徒期 routing_weight。

        学徒期 routing_weight 从 0.1 (幼稚态) 渐进到 1.0 (成熟态)。
        调用此方法后，router.routing_weights 会反映各 neuron 的当前成熟度。

        Args:
            maturity_tracker: MaturityTracker 实例

        Returns:
            {nid: float} 更新后的 routing_weights
        """
        for nid in list(self.prototypes.keys()):
            # 未注册到 maturity 的 neuron 视为已成熟（get_resonance_weight 返回 1.0）
            self.routing_weights[nid] = maturity_tracker.get_resonance_weight(nid)
        return dict(self.routing_weights)

    def merge_similar_domains(
        self,
        similarity_threshold: float = 0.95,
        keep_strategy: str = "first",
    ) -> List[Tuple[str, str]]:
        """Phase 5.3: 合并相似度 > threshold 的 domain（防爆炸）。

        如果两个 domain prototype 余弦相似度 > similarity_threshold，
        说明它们在教师 hidden space 中几乎重合，可以合并。

        Args:
            similarity_threshold: 合并阈值（默认 0.95）
            keep_strategy: "first" 保留先注册的，"last" 保留后注册的

        Returns:
            [(merged_nid, removed_nid), ...] 合并记录
        """
        nids = list(self.prototypes.keys())
        merged = []

        for i, ni in enumerate(nids):
            if ni not in self.prototypes:
                continue  # 已被合并
            pi = self.prototypes[ni]
            for nj in nids[i+1:]:
                if nj not in self.prototypes:
                    continue
                pj = self.prototypes[nj]
                sim = torch.dot(pi, pj).item()
                if sim >= similarity_threshold:
                    # 合并：根据 keep_strategy 决定保留谁
                    if keep_strategy == "first":
                        keep, remove = ni, nj
                    else:
                        keep, remove = nj, ni
                    self.prototypes.pop(remove, None)
                    self.neuron_meta.pop(remove, None)
                    self.routing_weights.pop(remove, None)
                    merged.append((keep, remove, sim))
                    print(f"[ThalamicRouter] Merged {remove} -> {keep} (sim={sim:.4f})")

        return [(k, r) for k, r, _ in merged]

    def drain_unknown_buffer(self) -> torch.Tensor:
        """清空并返回 unknown_buffer（Phase 5.2 新生域识别时用）。

        Returns:
            [N, hidden_dim] tensor
        """
        if not self.unknown_buffer:
            return torch.empty(0, self.hidden_dim)
        result = torch.stack(list(self.unknown_buffer))
        self.unknown_buffer.clear()
        return result

    def save(self, path: str) -> None:
        """保存 prototypes 到磁盘（同时保存 P6-2 的 prototypes_embed）。"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "hidden_dim": self.hidden_dim,
            "temperature": self.temperature,
            "hard_route_threshold": self.hard_route_threshold,
            "soft_route_threshold": self.soft_route_threshold,
            "unknown_buffer_size": self.unknown_buffer_size,
            "pooling": self.pooling,
            "prototypes": self.prototypes,
            "prototypes_embed": self.prototypes_embed,  # P6-2
            "neuron_meta": self.neuron_meta,
            "routing_weights": self.routing_weights,
        }, path)
        n_embed = len(self.prototypes_embed)
        print(f"[ThalamicRouter] Saved {len(self.prototypes)} prototypes"
              f"{' + ' + str(n_embed) + ' embed prototypes' if n_embed else ''}"
              f" to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ThalamicRouter":
        """从磁盘加载 prototypes。"""
        data = torch.load(path, map_location=device, weights_only=False)
        router = cls(
            hidden_dim=data["hidden_dim"],
            temperature=data["temperature"],
            hard_route_threshold=data.get("hard_route_threshold", 0.7),
            soft_route_threshold=data.get("soft_route_threshold", 0.4),
            unknown_buffer_size=data.get("unknown_buffer_size", 50),
            pooling=data.get("pooling", "last"),
        )
        router.prototypes = {nid: v.to(device) for nid, v in data["prototypes"].items()}
        # P6-2: 加载 embedding-based prototypes（旧 ckpt 没有时容错）
        router.prototypes_embed = {
            nid: v.to(device)
            for nid, v in data.get("prototypes_embed", {}).items()
        }
        router.neuron_meta = data.get("neuron_meta", {})
        router.routing_weights = data.get("routing_weights", {})
        # 修正：routing_weights 的 key 必须都是 str
        router.routing_weights = {str(k): float(v) for k, v in router.routing_weights.items()}
        print(f"[ThalamicRouter] Loaded {len(router.prototypes)} prototypes"
              f"{' + ' + str(len(router.prototypes_embed)) + ' embed prototypes' if router.prototypes_embed else ''}"
              f" from {path}")
        return router

    def get_prototype(self, nid: str) -> Optional[torch.Tensor]:
        return self.prototypes.get(nid)

    def list_domains(self) -> List[str]:
        """列出所有已注册的 neuron ID。"""
        return list(self.prototypes.keys())
