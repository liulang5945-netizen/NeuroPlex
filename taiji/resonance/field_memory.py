"""场记忆库（Field Memory Bank）— C26：睡眠把场状态沉淀为持久记忆。

背景（2026-08-11 架构审视结论）：
态极的共振场（4096-dim，推理时可写的共享状态）已是"可写记忆"的形态，
但缺两样——写无学习信号、写后不固化。本模块补"固化 + 检索 + 持久化"
这一格（"睡眠巩固"的第 0 层），注入编排（记忆条件化生成）由调用方完成。

记忆条目 = (向量, 文本标签, 时间戳)：
- 向量：场状态快照，统一 L2 归一化存储（场空间语义 = 方向即意义）
- 文本标签：生成该场状态的文本摘要/标题，供注入消费（向量暂不直接
  条件化生成——那是 C26 之后"记忆可读进生成"的下一步）
- 去重 = 与现存最近邻 cosine > threshold 视为重复跳过（人脑"突触稳态
  下调"的工程简化：睡眠只保留显著新模式，不重复囤积）

公开接口：
- consolidate(vectors, labels) -> added    固化（去重后追加）
- retrieve(query_vector, top_k) -> [(label, sim)]   跨会话检索
- save(path) / load(path) -> bool         持久化（.pt）
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class WriteGate(nn.Module):
    """可学习写门控（缺口 K：场记忆 vs Titans 最大差距的第 1 步）。

    Titans 的写是梯度驱动的 memory-as-model；态极 C26 第 0 格是朴素场快照 +
    硬阈值去重（cosine_threshold=0.92）。本门控把"是否值得写入"变成可学习的：
    输入 = [场向量, 与既有记忆的最近邻相似度] → 输出 P(值得写入)。

    训练信号 = 检索回报近似：新信息样本（与既有记忆 sim 低）标签 1（值得写），
    冗余样本（sim 高）标签 0（不值得）——门控学"什么值得记"，替代硬阈值。
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, field_vec: torch.Tensor,
                nearest_sim: torch.Tensor) -> torch.Tensor:
        """写门控打分。

        Args:
            field_vec: [..., in_dim] 场向量
            nearest_sim: [...] 与既有记忆的最近邻余弦

        Returns:
            [..., 1] sigmoid 概率（>0.5 = 值得写入）
        """
        x = torch.cat([field_vec, nearest_sim.unsqueeze(-1)], dim=-1)
        return torch.sigmoid(self.net(x))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"in_dim": self.in_dim, "state_dict": self.state_dict()}, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return False
        self.in_dim = int(payload.get("in_dim", self.in_dim))
        # 按产物维度重建网络（构造时传入的 dim 可能与产物不一致，
        # 如 sleep_engine 无 cortex 时用默认 4096——load 自修复维度）
        self.net = nn.Sequential(
            nn.Linear(self.in_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        try:
            self.load_state_dict(payload["state_dict"])
        except Exception:
            return False
        return True


class FieldMemoryBank:
    """持久场记忆库：固化 → 检索 → 保存/加载。"""

    def __init__(self, dim: int = 4096, cosine_threshold: float = 0.92,
                 gate: Optional[WriteGate] = None,
                 projector: Optional[Any] = None):
        self.dim = dim
        # 去重阈值：与现存记忆余弦 > threshold 视为重复（只保留显著新模式）
        self.cosine_threshold = cosine_threshold
        # 缺口 K：可学习写门控（替代/增强硬阈值；None = 纯硬阈值向后兼容）
        self.gate = gate
        # 缺口 L：跨域语义锚点投影（AnchorProjector）——检索在锚点空间进行
        # （跨域语义对齐的共享语义空间，而非原始场空间）；None = 场空间检索
        self.projector = projector
        self.entries: List[Dict] = []

    # ─── 固化 ───────────────────────────────────────────

    def consolidate(self, vectors: List[torch.Tensor],
                    labels: List[str],
                    gate: Optional[WriteGate] = None,
                    texts: Optional[List[Optional[str]]] = None) -> int:
        """固化一批场记忆：L2 归一化 + 去重决策后追加。

        去重决策（二选一）：
        - gate 给定（或 self.gate）：学习门控 P(值得写入) > 0.5 才写入
        - 否则：硬阈值（与既有记忆最近邻 sim > cosine_threshold 跳过）

        Args:
            vectors: 场状态快照列表（每个 [D] 或 [1, D]）
            labels: 与向量一一对应的文本标签
            gate: 本次调用显式指定的写门控（None → 用 self.gate）
            texts: 记忆内容文本（C26 增量三突触沉淀的重放样本来源；
                   None/空 → 回退用 label）

        Returns:
            added: 实际新增的条目数（被门控/阈值拒绝的条目跳过）
        """
        if len(vectors) != len(labels):
            raise ValueError(f"vectors({len(vectors)}) 与 labels({len(labels)}) 数量不一致")
        gate = gate if gate is not None else self.gate
        added = 0
        for i, (vec, label) in enumerate(zip(vectors, labels)):
            v = self._normalize(vec)
            if v is None:
                continue
            text = (texts[i] if texts is not None and i < len(texts) and texts[i]
                    else label)
            sim = self._max_sim(v)
            if gate is not None:
                keep = float(gate(v, torch.tensor(sim, dtype=v.dtype)).item()) > 0.5
            else:
                keep = sim <= self.cosine_threshold
            if not keep:
                continue
            # 缺口 L：若挂了锚点投影，同时保存锚点副本（检索在锚点空间进行）
            anchor = None
            if self.projector is not None:
                with torch.no_grad():
                    anchor = self.projector(v.unsqueeze(0)).squeeze(0).detach()
            self.entries.append({
                "vector": v,
                "anchor": anchor,
                "label": label,
                "text": text,
                "ts": datetime.now().isoformat(timespec="seconds"),
                # C26 增量三（2026-08-14）：海马→皮层两层记忆
                # access_count: 检索命中次数（高频 = 皮层沉淀候选）
                # consolidated: 是否已沉淀进神经元权重（沉淀后防重复重放）
                "access_count": 0,
                "consolidated": False,
            })
            added += 1
        return added

    def _normalize(self, vec: torch.Tensor) -> Optional[torch.Tensor]:
        v = vec.detach().float().flatten()
        if v.numel() != self.dim:
            return None
        norm = v.norm()
        if norm < 1e-8:
            return None
        return v / norm

    def _max_sim(self, v: torch.Tensor) -> float:
        """与既有记忆的最近邻余弦（写门控的关键输入特征）。"""
        if not self.entries:
            return 0.0
        stack = torch.stack([e["vector"] for e in self.entries])
        return float((stack @ v).max().item())

    def _is_duplicate(self, v: torch.Tensor) -> bool:
        return self._max_sim(v) > self.cosine_threshold

    # ─── 检索 ───────────────────────────────────────────

    def retrieve(self, query_vector: torch.Tensor,
                 top_k: int = 1) -> List[Tuple[str, float]]:
        """用查询向量对记忆库做余弦 top-k 检索（仅标签与相似度）。

        检索空间（二选一）：
        - 挂了锚点投影（projector）：记忆锚点副本 + query 投影后在**跨域语义
          锚点空间**余弦（对齐语义而非原始场方向）
        - 否则：原始场空间余弦

        Args:
            query_vector: 查询向量（如新会话的场状态快照）
            top_k: 返回最相似的 k 条记忆

        Returns:
            [(label, sim), ...] 按相似度降序
        """
        return [(label, sim) for label, sim, _ in
                self.retrieve_vectors(query_vector, top_k)]

    def retrieve_vectors(self, query_vector: torch.Tensor,
                         top_k: int = 1) -> List[Tuple[str, float, torch.Tensor]]:
        """带记忆向量的检索——C26 增量二"记忆可读进生成"的向量来源。

        与 retrieve 同检索逻辑，额外返回记忆向量本身（统一场空间快照，
        L2 归一化存储）。调用方拿到向量后可直接写入共振场做条件化
        （如 cortex.generate(memory_vectors=...)），让记忆通过已训练的
        场条件化路径参与生成，而非仅文本标签通道。

        Args:
            query_vector: 查询向量（如新会话的场状态快照）
            top_k: 返回最相似的 k 条记忆

        Returns:
            [(label, sim, vector), ...] 按相似度降序（vector 为 [D] 张量）
        """
        if not self.entries:
            return []
        q_raw = self._normalize(query_vector)
        if q_raw is None:
            return []
        if self.projector is not None:
            with torch.no_grad():
                q = self.projector(q_raw.unsqueeze(0)).squeeze(0)
            anchors = [e["anchor"] for e in self.entries
                       if e.get("anchor") is not None]
            if anchors:
                stack = torch.stack(anchors)
            else:
                # 旧库无锚点副本 → 回退场空间（query 也不投影）
                stack = torch.stack([e["vector"] for e in self.entries])
                q = q_raw
        else:
            q = q_raw
            stack = torch.stack([e["vector"] for e in self.entries])
        sims = stack @ q
        k = min(top_k, len(self.entries))
        idx = torch.topk(sims, k).indices.tolist()
        # C26 增量三：命中即计数（高频记忆 = 皮层突触沉淀候选）
        for i in idx:
            self.entries[i]["access_count"] += 1
        return [(self.entries[i]["label"], float(sims[i].item()),
                 self.entries[i]["vector"]) for i in idx]

    def frequent_entries(self, min_access: int = 2,
                         limit: Optional[int] = None) -> List[Dict]:
        """高频未沉淀条目——海马→皮层沉淀候选（C26 增量三）。

        人脑对应：反复重放（高频检索）的记忆才值得从海马迁移到皮层；
        已沉淀（consolidated）的条目不再重复迁移。

        Args:
            min_access: 访问计数下限（命中次数 ≥ 该值才算高频）
            limit: 最多返回条数（None = 全部）

        Returns:
            [{idx, label, text?}...]——text 需由调用方从 label 恢复；
            返回条目本身（含 vector/label/access_count），idx 供 mark_consolidated。
        """
        cands = [(i, e) for i, e in enumerate(self.entries)
                 if not e.get("consolidated") and e.get("access_count", 0) >= min_access]
        cands.sort(key=lambda ie: ie[1].get("access_count", 0), reverse=True)
        if limit is not None:
            cands = cands[:limit]
        return [dict(e, idx=i) for i, e in cands]

    def mark_consolidated(self, idxs: List[int]) -> int:
        """标记条目已沉淀进皮层（重置访问计数，防重复重放）。

        Returns:
            实际标记条数
        """
        n = 0
        for i in idxs:
            if 0 <= i < len(self.entries):
                self.entries[i]["consolidated"] = True
                self.entries[i]["access_count"] = 0
                n += 1
        return n

    # ─── 持久化 ─────────────────────────────────────────

    def save(self, path: str) -> None:
        payload = {
            "dim": self.dim,
            "cosine_threshold": self.cosine_threshold,
            "entries": [
                {
                    "vector": e["vector"].cpu(),
                    "anchor": e.get("anchor").cpu() if e.get("anchor") is not None else None,
                    "label": e["label"],
                    "text": e.get("text", e["label"]),
                    "ts": e["ts"],
                    "access_count": e.get("access_count", 0),
                    "consolidated": e.get("consolidated", False),
                }
                for e in self.entries
            ],
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(payload, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return False
        self.dim = int(payload.get("dim", self.dim))
        self.cosine_threshold = float(
            payload.get("cosine_threshold", self.cosine_threshold))
        self.entries = []
        for e in payload.get("entries", []):
            v = e["vector"].float().flatten()
            norm = v.norm()
            if norm < 1e-8:
                continue
            a = e.get("anchor")
            self.entries.append({
                "vector": v / norm,
                "anchor": a.float().flatten() if a is not None else None,
                "label": e.get("label", ""),
                "text": e.get("text") or e.get("label", ""),
                "ts": e.get("ts", ""),
                # C26 增量三：旧库无字段 → 默认 0/False（向后兼容）
                "access_count": int(e.get("access_count", 0)),
                "consolidated": bool(e.get("consolidated", False)),
            })
        return True

    # ─── 状态 ───────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.entries)

    def status(self) -> str:
        if not self.entries:
            return "empty"
        return f"{len(self.entries)} 条记忆, dim={self.dim}, 去重阈值={self.cosine_threshold}"
