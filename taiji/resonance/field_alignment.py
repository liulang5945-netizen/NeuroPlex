"""跨域语义锚点投影（AnchorProjector）— 缺口 L 落地（2026-08-11）。

背景（对齐冒烟 verify_c26_cross_domain_align.py 4/4 PASS）：
共振场未自发对齐跨域语义（同义对 -0.022），但场向量**蕴含可提取的跨域
语义结构**——可学投影 P 对比训练后同义 0.728 vs 错配 0.181（+0.547）。
结论：缺口 L 走「轻量锚点投影 + 场级对比约束」，无需 hub neuron 全套。

本模块把冒烟组件 AlignProj 正式化为产品组件：
- `AnchorProjector`：field_dim → proj_dim（默认 128）2 层 MLP，输出 L2 归一化
  （余弦空间 = 跨域语义锚点空间）
- `train_anchor_projector(vectors_dict, pairs, ...)`：内置对比 margin loss
  训练函数（同义跨域对拉近、异义对推开），供验证/训练脚本复用
- 持久化：save/load（.pt，含 in_dim 校验）

设计边界（C23-C4 教训）：
- 只作用在**场读出侧**（field_state → 锚点投影），不改变场写入/判定/生成路径
- 默认装配可选（cortex.set_anchor_projector），未装配时 field_state 原样返回
- live 神经元权重零改动（投影是独立组件，冻结场向量训练）
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class AnchorProjector(nn.Module):
    """跨域语义锚点投影：把场向量投影到可对齐的低维语义空间。

    输出空间语义 = 跨域语义锚点空间：同义跨域输入（zh"函数"↔en"function"）
    的投影余弦显著高于异义对——这是 hub neuron 设计中 field_vector 锚点角色
    的轻量实现（无需整套 hub 架构）。
    """

    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.proj_dim = proj_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """投影 + L2 归一化（余弦空间）。

        Args:
            x: [..., in_dim] 场向量（建议先 L2 归一化输入）

        Returns:
            [..., proj_dim] 归一化锚点向量
        """
        y = self.net(x)
        return y / (y.norm(dim=-1, keepdim=True) + 1e-8)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "in_dim": self.in_dim,
            "proj_dim": self.proj_dim,
            "state_dict": self.state_dict(),
        }, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return False
        self.in_dim = int(payload.get("in_dim", self.in_dim))
        self.proj_dim = int(payload.get("proj_dim", self.proj_dim))
        self.load_state_dict(payload["state_dict"])
        return True


def train_anchor_projector(
    vectors: Dict[str, torch.Tensor],
    pos_pairs: List[Tuple[str, str]],
    neg_pairs: List[Tuple[str, str]],
    proj_dim: int = 128,
    steps: int = 300,
    lr: float = 1e-3,
    margin: float = 0.5,
    seed: int = 42,
) -> AnchorProjector:
    """训练跨域语义锚点投影（对比 margin loss，冻结输入向量）。

    Args:
        vectors: {文本: 冻结场向量 [in_dim]}
        pos_pairs: 同义跨域对 [(zh_text, en_text), ...]
        neg_pairs: 异义跨域对
        steps: 训练步数（默认 300，与冒烟一致）
        lr / margin: 优化与对比边界

    Returns:
        训练好的 AnchorProjector（in_dim 从 vectors 推断）
    """
    in_dim = next(iter(vectors.values())).shape[-1]
    torch.manual_seed(seed)
    proj = AnchorProjector(in_dim, proj_dim)

    A = torch.stack([vectors[a] for a, _ in pos_pairs])
    P = torch.stack([vectors[b] for _, b in pos_pairs])
    N = torch.stack([vectors[b] for _, b in neg_pairs])

    opt = torch.optim.Adam(proj.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pa, pp, pn = proj(A), proj(P), proj(N)
        cos_pos = (pa * pp).sum(-1)
        cos_neg = (pa.unsqueeze(1) * pn.unsqueeze(0)).sum(-1).max(-1).values
        loss = torch.clamp(margin - (cos_pos - cos_neg), min=0).mean()
        loss.backward()
        opt.step()
    return proj


def evaluate_alignment(
    proj: AnchorProjector,
    vectors: Dict[str, torch.Tensor],
    pos_pairs: List[Tuple[str, str]],
    neg_pairs: List[Tuple[str, str]],
) -> Tuple[float, float, float]:
    """评估投影空间的跨域对齐质量。

    Returns:
        (同义对余弦均值, 错配对余弦均值, 对齐幅度)
    """
    with torch.no_grad():
        pa = torch.stack([proj(vectors[a]) for a, _ in pos_pairs])
        pp = torch.stack([proj(vectors[b]) for _, b in pos_pairs])
        p_syn = torch.stack([pa[i] @ pp[i] for i in range(len(pa))])
        ma = torch.stack([proj(vectors[a]) for a, _ in neg_pairs])
        mp = torch.stack([proj(vectors[b]) for _, b in neg_pairs])
        p_mis = torch.stack([ma[i] @ mp[i] for i in range(len(ma))])
    syn_mean = float(p_syn.mean().item())
    mis_mean = float(p_mis.mean().item())
    return syn_mean, mis_mean, syn_mean - mis_mean
