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
from typing import Dict, List, Optional, Tuple

import torch


class FieldMemoryBank:
    """持久场记忆库：固化 → 检索 → 保存/加载。"""

    def __init__(self, dim: int = 4096, cosine_threshold: float = 0.92):
        self.dim = dim
        # 去重阈值：与现存记忆余弦 > threshold 视为重复（只保留显著新模式）
        self.cosine_threshold = cosine_threshold
        self.entries: List[Dict] = []

    # ─── 固化 ───────────────────────────────────────────

    def consolidate(self, vectors: List[torch.Tensor],
                    labels: List[str]) -> int:
        """固化一批场记忆：L2 归一化 + 余弦去重后追加。

        Args:
            vectors: 场状态快照列表（每个 [D] 或 [1, D]）
            labels: 与向量一一对应的文本标签

        Returns:
            added: 实际新增的条目数（重复条目被跳过）
        """
        if len(vectors) != len(labels):
            raise ValueError(f"vectors({len(vectors)}) 与 labels({len(labels)}) 数量不一致")
        added = 0
        for vec, label in zip(vectors, labels):
            v = self._normalize(vec)
            if v is None:
                continue
            if self._is_duplicate(v):
                continue
            self.entries.append({
                "vector": v,
                "label": label,
                "ts": datetime.now().isoformat(timespec="seconds"),
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

    def _is_duplicate(self, v: torch.Tensor) -> bool:
        if not self.entries:
            return False
        stack = torch.stack([e["vector"] for e in self.entries])
        sims = stack @ v
        return float(sims.max().item()) > self.cosine_threshold

    # ─── 检索 ───────────────────────────────────────────

    def retrieve(self, query_vector: torch.Tensor,
                 top_k: int = 1) -> List[Tuple[str, float]]:
        """用查询向量对记忆库做余弦 top-k 检索。

        Args:
            query_vector: 查询向量（如新会话的场状态快照）
            top_k: 返回最相似的 k 条记忆

        Returns:
            [(label, sim), ...] 按相似度降序
        """
        if not self.entries:
            return []
        q = self._normalize(query_vector)
        if q is None:
            return []
        stack = torch.stack([e["vector"] for e in self.entries])
        sims = stack @ q
        k = min(top_k, len(self.entries))
        idx = torch.topk(sims, k).indices.tolist()
        return [(self.entries[i]["label"], float(sims[i].item())) for i in idx]

    # ─── 持久化 ─────────────────────────────────────────

    def save(self, path: str) -> None:
        payload = {
            "dim": self.dim,
            "cosine_threshold": self.cosine_threshold,
            "entries": [
                {"vector": e["vector"].cpu(), "label": e["label"], "ts": e["ts"]}
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
            self.entries.append({
                "vector": v / norm,
                "label": e.get("label", ""),
                "ts": e.get("ts", ""),
            })
        return True

    # ─── 状态 ───────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.entries)

    def status(self) -> str:
        if not self.entries:
            return "empty"
        return f"{len(self.entries)} 条记忆, dim={self.dim}, 去重阈值={self.cosine_threshold}"
