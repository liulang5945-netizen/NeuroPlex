"""R7 代际迁移蒸馏 smoke test.

验证三联蒸馏（KL + Hidden + Attention）的正确性：
1. 层映射构建正确（不同层数组合）
2. KL 蒸馏：相同 logits → 0，不同 → >0，温度缩放正确
3. vocab 对齐：不同 vocab 下按对齐表计算 KL
4. hidden 对齐：相同 hidden → 0；零初始化投影头初始不改变 student 表示
5. attention 转移：mean/proj 两种模式
6. neuron forward return_intermediate：形状正确
7. 向后兼容：return_intermediate=False 不影响正常 forward
8. 端到端：student 蒸馏后 logits 接近 teacher
9. checkpoint round-trip：student + distill 参数完整保存/恢复

用 TINY_TEST 规格避免加载真实模型。
"""
from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from taiji.resonance.config import TINY_TEST
from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.distillation import DistillationLoss, build_layer_map


def make_neuron(seed: int, vocab_size: int = 100, num_layers: int = 2,
                num_heads: int = 4, hidden_size: int = 256) -> ResonanceNeuron:
    cfg = copy.deepcopy(TINY_TEST)
    cfg.vocab_size = vocab_size
    cfg.neuron_id = f"n_{seed}"
    cfg.num_hidden_layers = num_layers
    cfg.num_attention_heads = num_heads
    cfg.hidden_size = hidden_size
    cfg.intermediate_size = hidden_size * 2
    torch.manual_seed(seed)
    return ResonanceNeuron(cfg)


def test_layer_map():
    """[1] 层映射构建正确。"""
    print("\n[1] 层映射构建")
    m = build_layer_map(2, 4)
    assert m == [(0, 0), (1, 3)], f"2→4 层映射应均匀采样, got {m}"
    m = build_layer_map(4, 4)
    assert m == [(0, 0), (1, 1), (2, 2), (3, 3)], f"4→4 应为逐层对应, got {m}"
    m = build_layer_map(4, 2)
    assert m == [(0, 0), (3, 1)], f"4→2 应从 teacher 角度均匀采样, got {m}"
    m = build_layer_map(6, 14)
    assert len(m) == 6, f"6→14 应产生 6 对映射, got {len(m)}"
    assert m[-1] == (5, 13), f"最后一对应映射到 teacher 最后一层, got {m[-1]}"
    print(f"  PASS: 层映射正确 (2→4: {build_layer_map(2,4)}, 6→14 末对: {build_layer_map(6,14)[-1]})")


def test_kl_loss():
    """[2] KL 蒸馏：相同 → 0，不同 → >0，温度缩放。"""
    print("\n[2] KL 蒸馏 loss")
    torch.manual_seed(42)
    d = DistillationLoss(256, 256, 2, 2, 4, 4)
    B, L, V = 2, 8, 100
    logits = torch.randn(B, L, V)

    # 相同 logits → KL ≈ 0
    kl_same = d.kl_loss(logits, logits.clone(), mask=None)
    assert abs(kl_same.item()) < 1e-4, f"相同 logits KL 应≈0, got {kl_same.item():.4f}"

    # 不同 logits → KL > 0
    logits2 = torch.randn(B, L, V)
    kl_diff = d.kl_loss(logits, logits2, mask=None)
    assert kl_diff.item() > 1e-3, f"不同 logits KL 应>0, got {kl_diff.item():.4f}"

    # 温度缩放：T 大 → 平滑 KL/T² 更小（分布更平缓），乘 T² 后总体不单调
    d1 = DistillationLoss(256, 256, 2, 2, 4, 4, temperature=1.0)
    d8 = DistillationLoss(256, 256, 2, 2, 4, 4, temperature=8.0)
    kl_t1 = d1.kl_loss(logits, logits2, mask=None)
    kl_t8 = d8.kl_loss(logits, logits2, mask=None)
    # 平滑 KL（= KL/T²）应随 T 增大而减小
    assert kl_t8 / (8.0 * 8.0) < kl_t1, f"平滑 KL 应随 T 减小, T=1: {kl_t1:.4f}, T=8 平滑: {kl_t8/64:.4f}"

    # mask 生效：全 False → 0
    mask_false = torch.zeros(B, L, dtype=torch.bool)
    kl_masked = d.kl_loss(logits, logits2, mask=mask_false)
    assert abs(kl_masked.item()) < 1e-6, f"全 False mask KL 应≈0, got {kl_masked.item()}"

    print(f"  PASS: KL 蒸馏正确 (same={kl_same.item():.2e}, diff={kl_diff.item():.4f}, "
          f"T1={kl_t1.item():.4f}, T8={kl_t8.item():.4f})")


def test_vocab_alignment():
    """[3] vocab 对齐：不同 vocab 按对齐表计算 KL。"""
    print("\n[3] vocab 对齐")
    torch.manual_seed(42)
    # student vocab 100, teacher vocab 200, 共享 token 50 个
    alignment = {i: i * 2 for i in range(50)}
    d = DistillationLoss(256, 256, 2, 2, 4, 4, vocab_alignment=alignment)
    B, L = 2, 8
    s_logits = torch.randn(B, L, 100)
    t_logits = torch.randn(B, L, 200)

    # 对齐后 student logits 子集 == teacher logits 对应子集 → KL = 0
    s_aligned = torch.randn(B, L, 100)
    t = torch.randn(B, L, 200)
    with torch.no_grad():
        s_copy = s_aligned.clone()
        for stu_id, tea_id in alignment.items():
            t[:, :, tea_id] = s_copy[:, :, stu_id]
    kl_zero = d.kl_loss(s_aligned, t, mask=None)
    assert abs(kl_zero.item()) < 1e-3, f"对齐子集相同 KL 应≈0, got {kl_zero.item():.4f}"

    # 不对齐 → KL > 0
    kl_rand = d.kl_loss(torch.randn(B, L, 100), torch.randn(B, L, 200), mask=None)
    assert kl_rand.item() > 1e-3
    print(f"  PASS: vocab 对齐正确 (aligned={kl_zero.item():.2e}, random={kl_rand.item():.4f})")


def test_hidden_loss():
    """[4] hidden 对齐：相同 → 0；零初始化投影头初始不改变 student 表示。"""
    print("\n[4] hidden 对齐")
    torch.manual_seed(42)
    # student hidden=256, teacher hidden=384（维度不同 → 需要投影）
    d = DistillationLoss(256, 384, 2, 3, 4, 6)
    B, L = 2, 8
    s_h = torch.randn(B, 2, L, 256)
    t_h = torch.randn(B, 3, L, 384)

    # 随机 hidden → loss > 0
    hl = d.hidden_loss(s_h, t_h, mask=None)
    assert hl.item() > 0, f"随机 hidden loss 应>0, got {hl.item()}"

    # 零初始化投影头初始输出 = 0，cosine loss = 1.0（无信息）
    proj0 = d.hidden_projectors[0]
    assert isinstance(proj0.proj, nn.Linear), "维度不同时应创建 Linear 投影"
    out = proj0(s_h[:, 0])
    assert out.abs().max().item() < 1e-6, f"零初始化投影输出应≈0, got {out.abs().max().item():.4f}"

    # 同维度 + 相同 hidden → cosine loss ≈ 0
    d_same = DistillationLoss(256, 256, 2, 2, 4, 4)
    h = torch.randn(B, 2, L, 256)
    hl_same = d_same.hidden_loss(h, h.clone(), mask=None, mode="cosine")
    assert abs(hl_same.item()) < 1e-5, f"相同 hidden cosine loss 应≈0, got {hl_same.item():.4f}"

    print(f"  PASS: hidden 对齐正确 (random={hl.item():.4f}, same={hl_same.item():.2e})")


def test_attn_loss():
    """[5] attention 转移：mean/proj 两种模式。"""
    print("\n[5] attention 转移")
    torch.manual_seed(42)
    B, L = 2, 8

    # mean 模式（同 heads）
    d_mean = DistillationLoss(256, 256, 2, 2, 4, 4, attn_align_mode="mean")
    s_attn = torch.randn(B, 2, 4, L, L)
    t_attn = torch.randn(B, 2, 4, L, L)
    al = d_mean.attn_loss(s_attn, t_attn)
    assert al.item() > 0

    # 相同 attn → 0
    al_same = d_mean.attn_loss(s_attn, s_attn.clone())
    assert abs(al_same.item()) < 1e-5, f"相同 attn loss 应≈0, got {al_same.item():.4f}"

    # proj 模式（student 4 heads → teacher 8 heads）
    d_proj = DistillationLoss(256, 256, 2, 2, 4, 8, attn_align_mode="proj")
    t_attn8 = torch.randn(B, 2, 8, L, L)
    al_proj = d_proj.attn_loss(s_attn, t_attn8)
    assert al_proj.item() > 0
    assert al_proj.shape == torch.Size([]), "loss 应为标量"

    print(f"  PASS: attention 转移正确 (mean={al.item():.4f}, same={al_same.item():.2e}, "
          f"proj(4→8 heads)={al_proj.item():.4f})")


def test_neuron_intermediate():
    """[6] neuron forward return_intermediate：形状正确。"""
    print("\n[6] neuron forward return_intermediate")
    torch.manual_seed(42)
    neuron = make_neuron(0)
    neuron.eval()
    B, L = 2, 16
    shared_emb = torch.randn(B, L, 512)  # base_embed_dim=512

    with torch.no_grad():
        result = neuron.forward(shared_emb, return_logits=True, return_intermediate=True)

    assert "intermediate_hidden" in result, "应返回 intermediate_hidden"
    assert "attn_weights" in result, "应返回 attn_weights"
    ih = result["intermediate_hidden"]
    aw = result["attn_weights"]
    assert ih.shape == (B, 2, L, 256), f"intermediate_hidden 形状 {ih.shape} != (B,2,L,256)"
    assert aw.shape == (B, 2, 4, L, L), f"attn_weights 形状 {aw.shape} != (B,2,4,L,L)"

    # logits 与不 return_intermediate 时一致（零破坏）
    with torch.no_grad():
        result_no = neuron.forward(shared_emb, return_logits=True)
    diff = (result["logits"] - result_no["logits"]).abs().max().item()
    assert diff < 1e-6, f"return_intermediate 不应改变 logits, diff={diff}"

    print(f"  PASS: intermediate_hidden={tuple(ih.shape)}, attn_weights={tuple(aw.shape)}, "
          f"logits diff={diff:.2e}")


def test_backward_compat():
    """[7] 向后兼容：return_intermediate=False（默认）不影响。"""
    print("\n[7] 向后兼容")
    torch.manual_seed(42)
    neuron = make_neuron(1)
    neuron.eval()
    shared_emb = torch.randn(1, 8, 512)
    with torch.no_grad():
        result = neuron.forward(shared_emb, return_logits=True)
    assert "intermediate_hidden" not in result, "默认不应返回 intermediate_hidden"
    assert "attn_weights" not in result, "默认不应返回 attn_weights"
    assert "logits" in result
    print(f"  PASS: 默认 forward 完全向后兼容")


def test_end_to_end():
    """[8] 端到端：student 蒸馏后 logits 接近 teacher。"""
    print("\n[8] 端到端蒸馏")
    torch.manual_seed(42)
    teacher = make_neuron(10)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = make_neuron(11)
    student.train()

    # 同规格（隐藏/层/head 相同），无需投影
    d = DistillationLoss(256, 256, 2, 2, 4, 4, temperature=2.0)
    B, L = 1, 16
    shared_emb = torch.randn(B, L, 512)

    # 初始 KL 大
    with torch.no_grad():
        t_result = teacher.forward(shared_emb, return_logits=True, return_intermediate=True)
        s_result = student.forward(shared_emb, return_logits=True, return_intermediate=True)
    kl_before = d.kl_loss(s_result["logits"], t_result["logits"], mask=None).item()

    # 单步优化 student 逼近 teacher logits
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    for step in range(30):
        s_result = student.forward(shared_emb, return_logits=True, return_intermediate=True)
        loss = d.kl_loss(s_result["logits"], t_result["logits"], mask=None)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        s_result = student.forward(shared_emb, return_logits=True, return_intermediate=True)
    kl_after = d.kl_loss(s_result["logits"], t_result["logits"], mask=None).item()
    assert kl_after < kl_before, f"蒸馏后 KL 应下降, before={kl_before:.4f}, after={kl_after:.4f}"
    print(f"  PASS: 蒸馏收敛 (KL: {kl_before:.4f} → {kl_after:.4f})")


def test_checkpoint_roundtrip():
    """[9] checkpoint round-trip：student + distill 参数完整保存/恢复。"""
    print("\n[9] checkpoint round-trip")
    torch.manual_seed(42)
    student = make_neuron(20)
    d = DistillationLoss(256, 384, 2, 3, 4, 6, attn_align_mode="proj")

    # 训练几步让参数变化
    B, L = 1, 8
    x = torch.randn(B, L, 512)
    teacher_h = torch.randn(B, 3, L, 384)
    t_logits = torch.randn(B, L, 100)
    opt = torch.optim.Adam(list(student.parameters()) + list(d.parameters()), lr=1e-3)
    s_result = student.forward(x, return_logits=True, return_intermediate=True)
    loss = d.kl_loss(s_result["logits"], t_logits, mask=None) + \
           d.hidden_loss(s_result["intermediate_hidden"], teacher_h, mask=None) + \
           d.attn_loss(s_result["attn_weights"], teacher_h.new_zeros(B, 3, 6, L, L))
    opt.zero_grad()
    loss.backward()
    opt.step()

    # 保存
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    ckpt = {"student_state": student.state_dict(), "distill_state": d.state_dict()}
    torch.save(ckpt, tmp_path)

    # 恢复
    student2 = make_neuron(21)
    d2 = DistillationLoss(256, 384, 2, 3, 4, 6, attn_align_mode="proj")
    ckpt2 = torch.load(tmp_path, map_location="cpu", weights_only=False)
    student2.load_state_dict(ckpt2["student_state"])
    d2.load_state_dict(ckpt2["distill_state"])
    os.remove(tmp_path)

    # 0 mismatch
    s1, s2 = student.state_dict(), student2.state_dict()
    mismatches = [k for k in s1 if not torch.equal(s1[k], s2[k])]
    assert not mismatches, f"student 参数不匹配: {mismatches[:3]}"
    d1, d2 = d.state_dict(), d2.state_dict()
    mismatches_d = [k for k in d1 if not torch.equal(d1[k], d2[k])]
    assert not mismatches_d, f"distill 参数不匹配: {mismatches_d[:3]}"
    print(f"  PASS: checkpoint round-trip 0 mismatch "
          f"(student={len(s1)} params, distill={len(d1)} params)")


def main():
    print("=" * 70)
    print("R7 代际迁移蒸馏 smoke test")
    print("=" * 70)

    test_layer_map()
    test_kl_loss()
    test_vocab_alignment()
    test_hidden_loss()
    test_attn_loss()
    test_neuron_intermediate()
    test_backward_compat()
    test_end_to_end()
    test_checkpoint_roundtrip()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
