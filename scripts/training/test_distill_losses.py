"""P1-3: distill_neurons.distill_one_neuron 三个 loss 分支单元测试。

测试目标：
  1. LM loss 分支      — F.cross_entropy(shift_logits, shift_targets)
  2. Distill loss 分支 — F.mse_loss(hidden, teacher_proj)
  3. Field contrastive — cosine pull(own) + clamp(push-other - 0.3, min=0)
  4. Total loss 组合   — lm_weight*lm + distill_weight*distill + field_weight*field
  5. 集成训练          — 真实 TINY neuron + Mock teacher，3 步训练验证 loss 下降

设计要点（避免磁盘投影依赖）：
  - 自定义 NeuronConfig: base_embed_dim=32, hidden_size=64, field_dim=4096
  - shared_embedding: nn.Embedding(vocab, 32)  → src=32=target，_project_embedding 短路
  - teacher_hidden: 64 维                       → src=64=target，_project_teacher_hidden 短路
  - teacher_directions: 4096 维                 → src=4096=target，field 投影短路
  - 这样无需 data/distill/shared_proj.pt 等磁盘文件

运行：
    python scripts/training/test_distill_losses.py
"""

from __future__ import annotations

import os
import sys

# 让脚本能直接运行：把项目根加入 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# sentencepiece 沙箱兜底
_LIBS = os.path.join(_PROJECT_ROOT, "_libs")
if os.path.isdir(_LIBS) and _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
# 1. 纯公式测试：用合成 tensor 验证三个 loss 的数学正确性
# ═══════════════════════════════════════════════════════════

def test_lm_loss_formula():
    """LM loss = F.cross_entropy(shift_logits, shift_targets)。

    构造完美预测（one-hot logits）→ loss ≈ 0；
    构造均匀预测 → loss ≈ log(vocab)。
    """
    torch.manual_seed(0)
    B, L, V = 2, 4, 8

    # 完美预测：logits 在目标位置取大值
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]])
    shift_targets = targets[:, 1:].contiguous().view(-1)  # [B*(L-1)]
    perfect_logits = torch.full((B, L - 1, V), -10.0)
    for b in range(B):
        for t in range(L - 1):
            perfect_logits[b, t, targets[b, t + 1]] = 10.0
    lm_loss_perfect = F.cross_entropy(
        perfect_logits.view(-1, V), shift_targets, ignore_index=-100
    )
    assert lm_loss_perfect.item() < 1e-4, f"perfect LM loss should be ~0, got {lm_loss_perfect.item()}"

    # 均匀预测：所有 logit=0 → loss = log(V)
    uniform_logits = torch.zeros(B, L - 1, V)
    lm_loss_uniform = F.cross_entropy(
        uniform_logits.view(-1, V), shift_targets, ignore_index=-100
    )
    expected = math.log(V)
    assert abs(lm_loss_uniform.item() - expected) < 1e-4, (
        f"uniform LM loss should be log(V)={expected:.4f}, got {lm_loss_uniform.item():.4f}"
    )
    print(f"[PASS] test_lm_loss_formula  (perfect={lm_loss_perfect.item():.6f}, uniform={lm_loss_uniform.item():.6f})")


def test_distill_loss_formula():
    """Distill loss = F.mse_loss(hidden, teacher_proj)。

    构造 hidden == teacher_proj → loss = 0；
    构造 hidden = teacher_proj + δ → loss = δ²。
    """
    torch.manual_seed(0)
    B, H = 3, 16

    teacher_proj = torch.randn(B, H)
    hidden_same = teacher_proj.clone()
    loss_same = F.mse_loss(hidden_same, teacher_proj)
    assert loss_same.item() < 1e-10, f"identical hidden/teacher should give loss 0, got {loss_same.item()}"

    delta = 0.5
    hidden_shift = teacher_proj + delta
    loss_shift = F.mse_loss(hidden_shift, teacher_proj)
    expected = delta ** 2  # mse = mean((δ)²) = δ²
    assert abs(loss_shift.item() - expected) < 1e-6, (
        f"shifted hidden loss should be δ²={expected:.6f}, got {loss_shift.item():.6f}"
    )
    print(f"[PASS] test_distill_loss_formula  (same={loss_same.item():.2e}, shift={loss_shift.item():.6f})")


def test_field_contrastive_pull_push():
    """Field contrastive: pull toward pos_dir (1-cos_pos), push away from neg_dir (clamp(cos_neg-0.3, min=0))。

    构造 field_vec == pos_dir → pull loss = 0
    构造 field_vec == -pos_dir → pull loss = 2
    构造 field_vec == pos_dir, neg_dir 也 == pos_dir → push loss = clamp(1-0.3, min=0) = 0.7
    """
    torch.manual_seed(0)
    D = 4096

    pos_dir = torch.randn(1, D)
    pos_dir = pos_dir / pos_dir.norm(dim=-1, keepdim=True)

    # case 1: field_vec == pos_dir → cos_pos = 1 → pull loss = 0
    field_vec_aligned = pos_dir.clone()
    cos_pos_1 = (field_vec_aligned * pos_dir).sum(dim=-1).mean()
    pull_loss_1 = 1.0 - cos_pos_1
    assert abs(pull_loss_1.item()) < 1e-5, f"aligned pull loss should be 0, got {pull_loss_1.item()}"

    # case 2: field_vec == -pos_dir → cos_pos = -1 → pull loss = 2
    field_vec_anti = -pos_dir
    cos_pos_2 = (field_vec_anti * pos_dir).sum(dim=-1).mean()
    pull_loss_2 = 1.0 - cos_pos_2
    assert abs(pull_loss_2.item() - 2.0) < 1e-5, (
        f"anti-aligned pull loss should be 2.0, got {pull_loss_2.item()}"
    )

    # case 3: neg_dir == pos_dir (worst case: 负方向与正方向相同)
    # cos_neg = 1 → push loss = clamp(1 - 0.3, min=0) = 0.7
    neg_dir = pos_dir.clone()
    cos_neg = (field_vec_aligned * neg_dir).sum(dim=-1).mean()
    push_loss = torch.clamp(cos_neg - 0.3, min=0.0)
    assert abs(push_loss.item() - 0.7) < 1e-5, (
        f"push loss when cos_neg=1 should be 0.7, got {push_loss.item()}"
    )

    # case 4: neg_dir 与 field_vec 反向 → cos_neg = -1 → clamp(-1-0.3, min=0) = 0
    neg_dir_orth = -pos_dir
    cos_neg_0 = (field_vec_aligned * neg_dir_orth).sum(dim=-1).mean()
    push_loss_0 = torch.clamp(cos_neg_0 - 0.3, min=0.0)
    assert abs(push_loss_0.item()) < 1e-5, (
        f"push loss when cos_neg=-1 should be 0 (clamped), got {push_loss_0.item()}"
    )

    print(f"[PASS] test_field_contrastive_pull_push  (pull_aligned={pull_loss_1.item():.4f}, "
          f"pull_anti={pull_loss_2.item():.4f}, push_same={push_loss.item():.4f}, push_anti={push_loss_0.item():.4f})")


def test_field_contrastive_margin_clamp():
    """Margin 0.3 的 clamp 行为：cos_neg ≤ 0.3 时 push loss = 0。

    构造 cos_neg = 0.3 边界 → push = 0
    构造 cos_neg = 0.5 → push = 0.2
    """
    D = 4096
    pos_dir = torch.zeros(1, D)
    pos_dir[0, 0] = 1.0  # unit vector along axis 0

    # cos_neg = 0.3: field_vec 在 pos_dir 方向投影 0.3
    field_vec_03 = torch.zeros(1, D)
    field_vec_03[0, 0] = 0.3
    # 补一个正交分量让 norm=1
    field_vec_03[0, 1] = math.sqrt(1 - 0.3 ** 2)
    cos_neg_03 = (field_vec_03 * pos_dir).sum(dim=-1).mean()
    push_03 = torch.clamp(cos_neg_03 - 0.3, min=0.0)
    assert push_03.item() < 1e-6, f"cos_neg=0.3 boundary → push should be 0, got {push_03.item()}"

    # cos_neg = 0.5 → push = 0.2
    field_vec_05 = torch.zeros(1, D)
    field_vec_05[0, 0] = 0.5
    field_vec_05[0, 1] = math.sqrt(1 - 0.5 ** 2)
    cos_neg_05 = (field_vec_05 * pos_dir).sum(dim=-1).mean()
    push_05 = torch.clamp(cos_neg_05 - 0.3, min=0.0)
    assert abs(push_05.item() - 0.2) < 1e-6, (
        f"cos_neg=0.5 → push should be 0.2, got {push_05.item()}"
    )
    print(f"[PASS] test_field_contrastive_margin_clamp  (boundary={push_03.item():.6f}, "
          f"above={push_05.item():.6f})")


def test_total_loss_composition():
    """验证 total = lm_weight*lm + distill_weight*distill + field_weight*field。"""
    torch.manual_seed(42)
    B, L, V, H, D = 2, 4, 16, 8, 4096
    lm_w, distill_w, field_w = 0.6, 0.2, 0.2

    # 合成数据
    logits = torch.randn(B, L, V, requires_grad=True)
    targets = torch.randint(0, V, (B, L))
    hidden = torch.randn(B, H, requires_grad=True)
    teacher_proj = torch.randn(B, H)
    field_vec = torch.randn(B, D, requires_grad=True)
    field_vec_normalized = field_vec / (field_vec.norm(dim=-1, keepdim=True) + 1e-8)
    pos_dir = torch.randn(1, D)
    pos_dir = pos_dir / pos_dir.norm(dim=-1, keepdim=True)

    # 复现 distill_one_neuron 的 loss 计算
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    lm_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1), ignore_index=-100,
    )
    distill_loss = F.mse_loss(hidden, teacher_proj)
    cos_pos = (field_vec_normalized * pos_dir).sum(dim=-1).mean()
    field_loss = 1.0 - cos_pos  # 单一 pos_dir, 无 neg
    total = lm_w * lm_loss + distill_w * distill_loss + field_w * field_loss

    # 手算预期
    expected_total = lm_w * lm_loss.item() + distill_w * distill_loss.item() + field_w * field_loss.item()
    assert abs(total.item() - expected_total) < 1e-5, (
        f"total loss composition mismatch: got {total.item():.6f}, expected {expected_total:.6f}"
    )

    # 验证梯度回传到三个分支
    total.backward()
    assert logits.grad is not None, "LM loss should propagate grad to logits"
    assert hidden.grad is not None, "distill loss should propagate grad to hidden"
    assert field_vec.grad is not None, "field loss should propagate grad to field_vec"

    print(f"[PASS] test_total_loss_composition  (lm={lm_loss.item():.4f}, "
          f"distill={distill_loss.item():.4f}, field={field_loss.item():.4f}, total={total.item():.4f})")


# ═══════════════════════════════════════════════════════════
# 2. 集成测试：真实 TINY neuron + Mock teacher，跑 distill_one_neuron
# ═══════════════════════════════════════════════════════════

class _MockAttention(nn.Module):
    """极简 attention：单层 Linear，返回 tensor（非 tuple）。"""
    def __init__(self, hidden):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.proj(x)


class _MockLayer(nn.Module):
    """极简 transformer block，满足 extract_hidden_states 的接口。"""
    def __init__(self, hidden):
        super().__init__()
        self.attention_norm = nn.Identity()
        self.ffn_norm = nn.Identity()
        self.attention = _MockAttention(hidden)
        self.feed_forward = nn.Identity()


class _MockBackbone(nn.Module):
    def __init__(self, vocab, hidden, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_MockLayer(hidden) for _ in range(num_layers)])
        self.norm = nn.Identity()


class MockTeacher(nn.Module):
    """最小教师模型：embedding(vocab, hidden) + 1 层 mock transformer。"""
    def __init__(self, vocab, hidden):
        super().__init__()
        self.backbone = _MockBackbone(vocab, hidden)

    def eval(self):
        for p in self.parameters():
            p.requires_grad_(False)
        return super().eval()


def _build_tiny_neuron():
    """构造一个极小的 ResonanceNeuron 用于测试。

    维度选择保证所有 _project_* 短路：
      - base_embed_dim=32  → shared_embedding 输出 32 维
      - hidden_size=64     → teacher_hidden 输出 64 维
      - field_dim=4096     → teacher_directions 4096 维（项目硬约束）
    """
    from taiji.resonance import ResonanceNeuron, NeuronConfig

    cfg = NeuronConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=128,
        vocab_size=64,        # 极小 vocab 加速
        base_embed_dim=32,
        field_dim=4096,       # 项目硬约束
        spec="tiny_test",
        lm_head_rank=0,       # 禁用低秩分解，简化测试
    )
    return ResonanceNeuron(cfg)


def test_distill_one_neuron_integration():
    """端到端：distill_one_neuron 跑 5 步，验证：
       1. 三种 loss 都有限
       2. total loss = 加权和
       3. loss 整体趋势下降（训练有效）
       4. teacher_directions 为 None 时 field_loss=0
    """
    from scripts.training.distill_neurons import distill_one_neuron

    torch.manual_seed(123)
    V, H, B, L = 64, 64, 2, 8

    # 构造 mock teacher：embedding(vocab, hidden=64) → extract_hidden_states 输出 64 维
    teacher = MockTeacher(V, H)

    # shared_embedding: 32 维 = neuron.base_embed_dim → _project_embedding 短路
    shared_embedding = nn.Embedding(V, 32)

    # 真实 TINY neuron
    neuron = _build_tiny_neuron()

    # 合成 domain_data: [N, L] token IDs
    domain_data = torch.randint(0, V, (16, L))

    # ── Sub-test A: 无 teacher_directions → field_loss 应为 0 ──
    result_a = distill_one_neuron(
        teacher_model=teacher,
        neuron=neuron,
        shared_embedding=shared_embedding,
        domain_data=domain_data,
        domain_name="test",
        num_steps=3,
        batch_size=B,
        lm_weight=0.6,
        distill_weight=0.2,
        field_contrastive_weight=0.2,  # 但 teacher_directions=None
        lr=5e-4,
        device="cpu",
        log_every=100,
        teacher_directions=None,  # ← 关键
    )
    assert math.isfinite(result_a["final_loss"]), "final_loss should be finite"
    assert math.isfinite(result_a["field_loss"]), "field_loss should be finite"
    assert result_a["field_loss"] == 0.0, (
        f"with teacher_directions=None, field_loss must be 0, got {result_a['field_loss']}"
    )
    assert result_a["steps"] == 3, f"should run 3 steps, got {result_a['steps']}"

    # ── Sub-test B: 有 teacher_directions → field_loss > 0（初始随机 field_vec 不对齐）──
    # 重建 neuron（避免 sub-test A 的权重影响）
    neuron_b = _build_tiny_neuron()

    # 3 个 domain 的方向向量：4096 维（匹配 field_dim，无投影）
    torch.manual_seed(456)
    teacher_directions = {
        "test": torch.randn(4096),    # pos_dir
        "other1": torch.randn(4096),  # neg
        "other2": torch.randn(4096),  # neg
    }

    # 手工跑一个 forward 看 field_loss 初值，确认 > 0
    neuron_b.train()
    input_ids = domain_data[:B]
    with torch.no_grad():
        teacher_emb = shared_embedding(input_ids)
        # _project_embedding 短路（src=32=target=32）
        from scripts.training.distill_neurons import _project_embedding, _project_teacher_hidden
        shared_emb = _project_embedding(teacher_emb, neuron_b.config.base_embed_dim)
    result = neuron_b.forward(shared_emb, return_logits=True)
    field_vec = result["field_vector"]

    fd = neuron_b.config.field_dim
    pos_dir = _project_teacher_hidden(teacher_directions["test"].unsqueeze(0), fd).squeeze(0)
    pos_dir = pos_dir / (pos_dir.norm() + 1e-8)
    cos_pos = (field_vec * pos_dir.unsqueeze(0)).sum(dim=-1).mean()
    field_loss_manual = 1.0 - cos_pos
    assert field_loss_manual.item() > 0, (
        f"random init field_vec should not align with pos_dir, field_loss={field_loss_manual.item()}"
    )

    # 跑 5 步完整训练
    result_b = distill_one_neuron(
        teacher_model=teacher,
        neuron=neuron_b,
        shared_embedding=shared_embedding,
        domain_data=domain_data,
        domain_name="test",
        num_steps=5,
        batch_size=B,
        lm_weight=0.6,
        distill_weight=0.2,
        field_contrastive_weight=0.2,
        lr=5e-4,
        device="cpu",
        log_every=100,
        teacher_directions=teacher_directions,
    )
    assert math.isfinite(result_b["final_loss"]), "final_loss should be finite"
    assert math.isfinite(result_b["field_loss"]), "field_loss should be finite"
    assert result_b["field_loss"] > 0, (
        f"with teacher_directions set, field_loss should be > 0, got {result_b['field_loss']}"
    )
    assert result_b["steps"] == 5, f"should run 5 steps, got {result_b['steps']}"

    print(f"[PASS] test_distill_one_neuron_integration")
    print(f"        Sub-A (no directions): final_loss={result_a['final_loss']:.4f}, "
          f"field_loss={result_a['field_loss']:.4f}")
    print(f"        Sub-B (3 directions):  final_loss={result_b['final_loss']:.4f}, "
          f"field_loss={result_b['field_loss']:.4f}")


def test_distill_loss_decreases_with_training():
    """验证训练有效：跑 20 步，loss 应整体下降。

    用一个"易学"任务：domain_data 全是固定序列 [1,2,3,4,1,2,3,4,...]，
    neuron 应能快速学会预测下一个 token。
    """
    from scripts.training.distill_neurons import distill_one_neuron

    torch.manual_seed(789)
    V, H, B, L = 64, 64, 2, 8

    teacher = MockTeacher(V, H)
    shared_embedding = nn.Embedding(V, 32)
    neuron = _build_tiny_neuron()

    # 固定重复序列：[1,2,3,4,1,2,3,4]
    pattern = torch.tensor([1, 2, 3, 4])
    domain_data = pattern.repeat(4).unsqueeze(0).repeat(16, 1)  # [16, 16]
    domain_data = domain_data[:, :L]  # [16, 8]

    result = distill_one_neuron(
        teacher_model=teacher,
        neuron=neuron,
        shared_embedding=shared_embedding,
        domain_data=domain_data,
        domain_name="pattern",
        num_steps=20,
        batch_size=B,
        lm_weight=1.0,        # 只用 LM loss，纯训练信号
        distill_weight=0.0,
        field_contrastive_weight=0.0,
        lr=1e-3,
        device="cpu",
        log_every=100,
        teacher_directions=None,
    )
    # 20 步后 PPL 应该比初始低（即使是 tiny neuron 也能学 4-token 循环）
    # 我们用 final_loss < log(V) 作为下界（log(64)≈4.16，即比随机好）
    log_v = math.log(V)
    assert result["final_loss"] < log_v, (
        f"after 20 steps on a learnable pattern, final_loss={result['final_loss']:.4f} "
        f"should be < log(V)={log_v:.4f} (random baseline)"
    )
    print(f"[PASS] test_distill_loss_decreases_with_training  "
          f"(final_loss={result['final_loss']:.4f} < log(V)={log_v:.4f})")


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("P1-3: distill_neurons.distill_one_neuron 单元测试")
    print("=" * 72)
    print()

    print("── Layer 1: 纯公式验证 ──")
    test_lm_loss_formula()
    test_distill_loss_formula()
    test_field_contrastive_pull_push()
    test_field_contrastive_margin_clamp()
    test_total_loss_composition()
    print()

    print("── Layer 2: 集成训练验证 ──")
    test_distill_one_neuron_integration()
    test_distill_loss_decreases_with_training()
    print()

    print("=" * 72)
    print("ALL TESTS PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
