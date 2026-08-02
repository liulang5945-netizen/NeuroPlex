"""R7: 代际迁移蒸馏（teacher → student）三联 loss。

设计哲学（人脑启发：代际学习 / 髓鞘化）：
- 老神经元（teacher）已成熟，其输出分布、表示空间、注意力模式蕴含领域知识
- 新神经元（student）通过模仿 teacher 快速获得基础能力，再个性化发展
- 与"小神经元协作匹配大模型"的核心理念一致：先继承、再创新

三联蒸馏（用户选定方案 D = A + B + C）：
A. Logits 蒸馏：KL(student_logits/T, teacher_logits/T) * T^2
   - student 直接模仿 teacher 的输出分布（温度软化，保留类别间相对信息）
   - 支持 vocab 对齐：student/teacher vocab 不同时，仅在共享 token 子集上计算
B. 中间层对齐：per-layer 可学习投影后 cosine/MSE
   - student 每层 hidden 投影到 teacher 维度后对齐
   - 层映射：均匀采样（student n 层 ↔ teacher m 层）
   - 残差 + 零初始化投影头：初始不改变 student 表示（零破坏）
C. 注意力转移：student attention 模式匹配 teacher
   - 每层 attention weights [B, heads, L, L] 对齐
   - head 数不同时：可学习 head 投影（student_heads -> teacher_heads）
   - 或 mode="mean"：聚合所有 heads 后对齐（无额外参数）

全部可微，可端到端训练。兼容 CE loss 联合优化（权重可调）。

用法：
    distill = DistillationLoss(
        student_hidden=768, teacher_hidden=1024,
        student_layers=6, teacher_layers=14,
        student_heads=12, teacher_heads=16,
        vocab_alignment={stu_id: tea_id, ...},  # None = 相同 vocab
    )
    losses = distill(
        student_logits, teacher_logits,
        student_hiddens, teacher_hiddens,
        student_attns, teacher_attns,
        mask=mask,  # [B, L] bool, True = 有效 token
    )
    total = (logits_w * losses["kl"] + hidden_w * losses["hidden"]
             + attn_w * losses["attn"])
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_layer_map(student_layers: int, teacher_layers: int) -> List[Tuple[int, int]]:
    """构建 student 层 → teacher 层映射（均匀采样）。

    保证每一对 (s_idx, t_idx) 覆盖从浅到深的结构位置：
    - student_layers <= teacher_layers: 每个 student 层映射到一个均匀采样的 teacher 层
    - student_layers > teacher_layers: 每个 teacher 层映射到一个均匀采样的 student 层
    """
    if student_layers == teacher_layers:
        return [(i, i) for i in range(student_layers)]
    if student_layers < teacher_layers:
        teacher_indices = [
            int(round(i * (teacher_layers - 1) / (student_layers - 1)))
            for i in range(student_layers)
        ]
        return [(i, teacher_indices[i]) for i in range(student_layers)]
    student_indices = [
        int(round(i * (student_layers - 1) / (teacher_layers - 1)))
        for i in range(teacher_layers)
    ]
    return [(student_indices[i], i) for i in range(teacher_layers)]


class HiddenProjector(nn.Module):
    """student hidden → teacher hidden 的可学习投影（残差 + 零初始化）。

    零初始化保证初始投影输出 = 0，不改变 student 表示（零破坏升级）。
    student_dim == teacher_dim 时为 Identity（零参数）。
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        if student_dim == teacher_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(student_dim, teacher_dim, bias=False)
            nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class HeadProjector(nn.Module):
    """student attention heads → teacher heads 的可学习投影。

    输入 [B, s_heads, L, L]，输出 [B, t_heads, L, L]。
    s_heads == t_heads 时为 Identity。
    """

    def __init__(self, student_heads: int, teacher_heads: int):
        super().__init__()
        if student_heads == teacher_heads:
            self.proj = nn.Identity()
        else:
            # 展平 head 维：Linear(s_heads, t_heads) 作用在 head 维
            self.proj = nn.Linear(student_heads, teacher_heads, bias=False)
            nn.init.zeros_(self.proj.weight)

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        # attn: [B, s_heads, L, L] -> [B, L, L, s_heads] -> Linear -> [B, L, L, t_heads] -> [B, t_heads, L, L]
        if isinstance(self.proj, nn.Identity):
            return attn
        return self.proj(attn.transpose(1, 3)).transpose(1, 3)


class DistillationLoss(nn.Module):
    """R7: 代际迁移蒸馏三联 loss（KL + hidden + attention）。"""

    def __init__(
        self,
        student_hidden: int,
        teacher_hidden: int,
        student_layers: int,
        teacher_layers: int,
        student_heads: int,
        teacher_heads: int,
        temperature: float = 4.0,  # 蒸馏温度（软化分布，保留类间关系）
        vocab_alignment: Optional[Dict[int, int]] = None,  # {student_id: teacher_id}, None=相同 vocab
        attn_align_mode: str = "mean",  # "mean"=聚合 heads, "proj"=可学习 head 投影
    ):
        """Args:
            student_hidden: student hidden_size
            teacher_hidden: teacher hidden_size
            student_layers: student Transformer 层数
            teacher_layers: teacher Transformer 层数
            student_heads: student attention heads 数
            teacher_heads: teacher attention heads 数
            temperature: KL 蒸馏温度 T（软化 logits）
            vocab_alignment: student token id -> teacher token id 映射（vocab 不同时）
            attn_align_mode: "mean"=所有 heads 平均后对齐（零参数）,
                             "proj"=可学习 head 投影（student_heads -> teacher_heads）
        """
        super().__init__()
        self.temperature = temperature
        self.vocab_alignment = vocab_alignment
        self.attn_align_mode = attn_align_mode

        # B. 层映射 + hidden 投影头
        self.layer_map: List[Tuple[int, int]] = build_layer_map(student_layers, teacher_layers)
        self.hidden_projectors = nn.ModuleList([
            HiddenProjector(student_hidden, teacher_hidden)
            for _ in self.layer_map
        ])

        # C. head 投影
        if attn_align_mode == "proj":
            self.head_projector = HeadProjector(student_heads, teacher_heads)
        else:
            self.head_projector = nn.Identity()

        # 预计算 vocab 对齐索引（CUDA 加速用）
        self._aligned_stu_ids: Optional[torch.Tensor] = None
        self._aligned_tea_ids: Optional[torch.Tensor] = None

    def _get_alignment_indices(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (student_ids, teacher_ids) 用于 gather 对齐 logits。"""
        if self.vocab_alignment is None:
            return None, None
        if self._aligned_stu_ids is None:
            pairs = sorted(self.vocab_alignment.items())
            self._aligned_stu_ids = torch.tensor([s for s, _ in pairs], dtype=torch.long)
            self._aligned_tea_ids = torch.tensor([t for _, t in pairs], dtype=torch.long)
        return (
            self._aligned_stu_ids.to(device),
            self._aligned_tea_ids.to(device),
        )

    def kl_loss(
        self,
        student_logits: torch.Tensor,  # [B, L, V_s]
        teacher_logits: torch.Tensor,  # [B, L, V_t]
        mask: Optional[torch.Tensor] = None,  # [B, L] bool, True=有效
    ) -> torch.Tensor:
        """A. Logits 蒸馏：KL(student/T || teacher/T) * T^2。

        vocab 相同时直接计算；vocab 不同时按对齐表只算共享 token 子集。
        """
        T = self.temperature
        s = student_logits.float() / T
        t = teacher_logits.float() / T

        if self.vocab_alignment is not None:
            # 只取共享 token 子集 [B, L, V_aligned]
            stu_ids, tea_ids = self._get_alignment_indices(s.device)
            s = s.index_select(-1, stu_ids)
            t = t.index_select(-1, tea_ids)

        log_softmax_s = F.log_softmax(s, dim=-1)
        log_softmax_t = F.log_softmax(t, dim=-1)
        softmax_t = torch.exp(log_softmax_t)
        # KL(teacher || student) per-token = Σ softmax(t)·(log_softmax(t) − log_softmax(s))
        per_token_kl = (softmax_t * (log_softmax_t - log_softmax_s)).sum(dim=-1)  # [B, L]

        if mask is not None:
            per_token_kl = per_token_kl * mask.float()
            n = mask.float().sum().clamp(min=1.0)
        else:
            n = per_token_kl.numel()
        loss = per_token_kl.sum() / n * (T * T)
        return loss

    def hidden_loss(
        self,
        student_hiddens: torch.Tensor,  # [B, s_layers, L, s_hidden]
        teacher_hiddens: torch.Tensor,  # [B, t_layers, L, t_hidden]
        mask: Optional[torch.Tensor] = None,
        mode: str = "cosine",  # "cosine" / "mse"
    ) -> torch.Tensor:
        """B. 中间层对齐：投影后 cosine/MSE。

        层映射由 self.layer_map 决定。mask 覆盖位置不参与（取均值向量）。
        """
        losses = []
        for proj, (s_idx, t_idx) in zip(self.hidden_projectors, self.layer_map):
            s = student_hiddens[:, s_idx]  # [B, L, s_hidden]
            t = teacher_hiddens[:, t_idx]  # [B, L, t_hidden]
            s_proj = proj(s)  # [B, L, t_hidden]

            if mask is not None:
                # 有效 token 上的平均表示（去 mask）
                mask_f = mask.float().unsqueeze(-1)  # [B, L, 1]
                s_mean = (s_proj * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)  # [B, t_hidden]
                t_mean = (t * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
            else:
                s_mean = s_proj.mean(dim=1)
                t_mean = t.mean(dim=1)

            if mode == "cosine":
                losses.append(1.0 - F.cosine_similarity(s_mean, t_mean, dim=-1).mean())
            else:
                losses.append(F.mse_loss(s_mean, t_mean))
        if not losses:
            return torch.tensor(0.0, device=student_hiddens.device)
        return torch.stack(losses).mean()

    def attn_loss(
        self,
        student_attns: torch.Tensor,  # [B, s_layers, s_heads, L, L]
        teacher_attns: torch.Tensor,  # [B, t_layers, t_heads, L, L]
    ) -> torch.Tensor:
        """C. 注意力转移：student attention 模式匹配 teacher。

        - mode="mean": 每层所有 heads 平均 → [B, L, L]，MSE 对齐
        - mode="proj": 每层用可学习 HeadProjector 投影后 MSE 对齐
        """
        losses = []
        for s_idx, t_idx in self.layer_map:
            s = student_attns[:, s_idx]  # [B, s_heads, L, L]
            t = teacher_attns[:, t_idx]  # [B, t_heads, L, L]

            if self.attn_align_mode == "proj":
                s_aligned = self.head_projector(s)  # [B, t_heads, L, L]
                s_aligned = F.normalize(s_aligned.reshape(s_aligned.shape[0], -1), dim=-1)
                t_norm = F.normalize(t.reshape(t.shape[0], -1), dim=-1)
                losses.append(F.mse_loss(s_aligned, t_norm))
            else:
                # mean: 聚合 heads
                s_mean = s.mean(dim=1)  # [B, L, L]
                t_mean = t.mean(dim=1)
                s_norm = F.normalize(s_mean.reshape(s_mean.shape[0], -1), dim=-1)
                t_norm = F.normalize(t_mean.reshape(t_mean.shape[0], -1), dim=-1)
                losses.append(F.mse_loss(s_norm, t_norm))
        if not losses:
            return torch.tensor(0.0, device=student_attns.device)
        return torch.stack(losses).mean()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_hiddens: torch.Tensor,
        teacher_hiddens: torch.Tensor,
        student_attns: Optional[torch.Tensor] = None,
        teacher_attns: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        hidden_mode: str = "cosine",
    ) -> Dict[str, torch.Tensor]:
        """计算三联蒸馏 loss。

        Args:
            student_logits: [B, L, V_s]
            teacher_logits: [B, L, V_t]（teacher 输出，detach 由调用方处理）
            student_hiddens: [B, s_layers, L, s_hidden]
            teacher_hiddens: [B, t_layers, L, t_hidden]
            student_attns: [B, s_layers, s_heads, L, L]（可选）
            teacher_attns: [B, t_layers, t_heads, L, L]（可选）
            mask: [B, L] bool, True=有效 token
            hidden_mode: "cosine" / "mse"

        Returns:
            {"kl": ..., "hidden": ..., "attn": ...}
        """
        results = {}
        results["kl"] = self.kl_loss(student_logits, teacher_logits, mask)
        results["hidden"] = self.hidden_loss(student_hiddens, teacher_hiddens, mask, hidden_mode)
        if student_attns is not None and teacher_attns is not None:
            results["attn"] = self.attn_loss(student_attns, teacher_attns)
        else:
            results["attn"] = torch.tensor(0.0, device=student_logits.device)
        return results
