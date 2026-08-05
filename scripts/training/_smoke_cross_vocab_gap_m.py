"""缺口 M smoke test：跨 vocab 联合训练 + 词库转译矩阵。

验证项：
1. 词库转译 build_domain_to_domain_alignment：source token → target token 映射
   （byte fallback 正确处理，空映射 pad 兜底）
2. build_logits_alignment_matrix：稀疏矩阵形状 [V_src, V_tgt]、行归一化、缓存复用、
   指纹失效（tokenizer 热插拔自动重建）
3. forward_train 跨 vocab：vocab 不一致时融合到 target 域空间，fused_logits 形状
   [B, L, V_tgt]，CE 可算，梯度可流回 side_channels
4. 向后兼容：vocab 一致路径不受影响（不传 target_domain 也可运行）

Usage:
    python -u scripts/training/_smoke_cross_vocab_gap_m.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from taiji.resonance import ResonanceNeuron, ResonanceField, ResonanceEnsemble
from taiji.resonance.config import NeuronConfig
from taiji.resonance.translator import (
    TokenizerHub, build_domain_to_domain_alignment,
    build_logits_alignment_matrix, tokenizer_fingerprint,
)


# ── 最小 SentencePiece 接口 mock（id_to_piece / encode / decode / pad_id）──
class MockSP:
    def __init__(self, vocab_size: int, domain: str):
        self._size = vocab_size
        self._domain = domain
        # 用可逆伪 piece 保证转译有真实映射

    def GetPieceSize(self) -> int:
        return self._size

    def id_to_piece(self, i: int) -> str:
        if i == 0:
            return "<pad>"
        if i == 1:
            return "<s>"
        if i == 2:
            return "</s>"
        if i == 3:
            return "<unk>"
        if i == 4:
            return "<0x0A>"  # byte fallback 样例（换行）
        # 其余 piece：域内词元，源/目标域共享词汇以便对齐
        return f"tok_{i}"

    def piece_to_id(self, piece: str) -> int:
        return int(piece.split("_")[1]) if piece.startswith("tok_") else -1

    def encode(self, text: str) -> list:
        """文本 → id 列表：模拟子词切分（按空格拆，tok_ 前缀）。"""
        if text == "\n":
            return [4]
        if text in ("<pad>", "<s>", "</s>", "<unk>"):
            return [[0, 1, 2, 3][["<pad>", "<s>", "</s>", "<unk>"].index(text)]]
        ids = []
        for tok in text.split(" "):
            tok = tok.strip()
            if not tok:
                continue
            pid = self.piece_to_id(tok)
            if pid >= 0 and pid < self._size:
                ids.append(pid)
        return ids

    def decode(self, ids: list) -> str:
        if len(ids) == 1 and ids[0] == 4:
            return "\n"
        return " ".join(self.id_to_piece(i).removeprefix("tok_") for i in ids)

    def pad_id(self) -> int:
        return 0


def make_tiny_neuron(nid: str, field_dim: int, vocab_size: int) -> ResonanceNeuron:
    cfg = NeuronConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=vocab_size,
        base_embed_dim=32,
        field_dim=field_dim,
        spec="test",
        neuron_id=nid,
    )
    cfg.unified_field_dim = None
    return ResonanceNeuron(cfg)


def test_alignment_build():
    print("\n=== Test 1: 词库转译 build_domain_to_domain_alignment ===")
    src = MockSP(64, "code")
    tgt = MockSP(32, "zh")
    alignment, src_vocab = build_domain_to_domain_alignment(src, tgt)
    assert src_vocab == 64, f"src_vocab={src_vocab}"
    assert len(alignment) == 64
    # byte fallback: <0x0A> → 换行 → target encode → [4]
    assert alignment[4] == [4], f"byte fallback 转译错误: {alignment[4]}"
    # 普通 token：tok_10 → "10" → [10]
    assert 10 in alignment[10] or alignment[10] == [10], f"普通 token 转译错误: {alignment[10]}"
    # 空映射兜底（若 encode 失败 → [pad_id]）
    nonempty = all(len(v) >= 1 for v in alignment)
    assert nonempty, "存在空映射未兜底"
    print(f"  ✅ alignment 构建通过: {src_vocab} entries, byte fallback 正确")
    return True


def test_matrix_and_hotswap():
    print("\n=== Test 2: 稀疏矩阵 + 缓存 + 指纹失效（热插拔）===")
    src = MockSP(64, "code")
    tgt = MockSP(32, "zh")
    cache = {}

    m1 = build_logits_alignment_matrix(src, tgt, "code", "zh", cache)
    assert m1.shape[0] == 64 and m1.shape[1] == 32, f"矩阵形状错误: {m1.shape}"
    assert m1._nnz() > 0, "矩阵全零"

    # 行归一化：每行非零权重和 = 1（logits 尺度守恒）
    dense = m1.to_dense()
    row_sums = dense.sum(dim=1)
    nonempty_rows = row_sums[row_sums > 0]
    assert torch.allclose(nonempty_rows, torch.ones_like(nonempty_rows), atol=1e-5), \
        f"行归一化失败: {nonempty_rows[:5]}"

    # 缓存复用：同一 tokenizer 实例 → 返回同一对象
    m2 = build_logits_alignment_matrix(src, tgt, "code", "zh", cache)
    assert m1 is m2, "缓存未命中（同一指纹应复用）"

    # 指纹失效：替换 target tokenizer（热插拔）→ 自动重建
    tgt_new = MockSP(64, "zh")  # vocab 变化
    m3 = build_logits_alignment_matrix(src, tgt_new, "code", "zh", cache)
    assert m3.shape[1] == 64, f"热插拔后矩阵未重建: {m3.shape}"
    print(f"  ✅ 矩阵构建/归一化/缓存/热插拔失效全部通过")
    return True


def test_forward_train_cross_vocab():
    print("\n=== Test 3: forward_train 跨 vocab 联合训练 ===")
    torch.manual_seed(42)
    src = MockSP(64, "code")
    tgt = MockSP(128, "zh")
    hub = TokenizerHub()
    hub.register_domain("code", src)
    hub.register_domain("zh", tgt)

    neurons = {
        "code": make_tiny_neuron("code", field_dim=128, vocab_size=64),
        "zh_a": make_tiny_neuron("zh_a", field_dim=128, vocab_size=128),
        "zh_b": make_tiny_neuron("zh_b", field_dim=128, vocab_size=128),
    }
    for post_id in neurons:
        for pre_id in neurons:
            if post_id == pre_id:
                continue
            neurons[post_id].establish_side_channel(pre_id, neurons[pre_id], channel_type="excite")
    # 冻结核心，仅 side_channels 可训练（验证梯度流）
    for neuron in neurons.values():
        for p in neuron.parameters():
            p.requires_grad = False
        for ch in neuron.excite_channels.values():
            for p in ch.parameters():
                p.requires_grad = True
        neuron.train()

    field = ResonanceField(dim=128)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2)
    ensemble.set_tokenizer_hub(hub)

    shared_emb = torch.randn(2, 8, 32)
    result = ensemble.forward_train(
        shared_embeddings=shared_emb,
        n_rounds=2,
        fusion_mode="soft",
        target_domain="zh",
    )
    fused = result["fused_logits"]
    assert fused.shape == (2, 8, 128), f"fused_logits 形状错误: {fused.shape}"
    assert torch.isfinite(fused).all(), "fused_logits 含 NaN/Inf"

    # CE + 反向传播（验证可微）
    targets = torch.randint(0, 128, (2, 8), dtype=torch.long)
    loss = F.cross_entropy(fused.view(-1, 128), targets.view(-1))
    loss = loss + 0.01 * result["balance_loss"] + 0.05 * result["diversity_loss"]
    loss.backward()

    grad_ok = grad_total = 0
    for neuron in neurons.values():
        for ch in neuron.excite_channels.values():
            for p in ch.parameters():
                grad_total += 1
                if p.grad is not None and p.grad.abs().sum().item() > 0:
                    grad_ok += 1
    assert grad_ok == grad_total, f"梯度只到 {grad_ok}/{grad_total}"
    print(f"  ✅ 跨 vocab 融合通过: fused={tuple(fused.shape)}, "
          f"side_channels 梯度 {grad_ok}/{grad_total}, loss={loss.item():.4f}")
    return True


def test_forward_train_same_vocab_backward_compat():
    print("\n=== Test 4: vocab 一致向后兼容（不传 target_domain）===")
    torch.manual_seed(42)
    neurons = {
        "zh_a": make_tiny_neuron("zh_a", field_dim=128, vocab_size=128),
        "zh_b": make_tiny_neuron("zh_b", field_dim=128, vocab_size=128),
    }
    for post_id in neurons:
        for pre_id in neurons:
            if post_id == pre_id:
                continue
            neurons[post_id].establish_side_channel(pre_id, neurons[pre_id], channel_type="excite")
    field = ResonanceField(dim=128)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2)
    shared_emb = torch.randn(2, 8, 32)
    result = ensemble.forward_train(
        shared_embeddings=shared_emb, n_rounds=2, fusion_mode="soft",
    )
    assert result["fused_logits"].shape == (2, 8, 128)
    print(f"  ✅ 向后兼容通过: fused={tuple(result['fused_logits'].shape)}")
    return True


def test_fingerprint():
    print("\n=== Test 5: tokenizer_fingerprint 区分 ===")
    a = MockSP(64, "code")
    b = MockSP(128, "zh")
    assert tokenizer_fingerprint(a) != tokenizer_fingerprint(b)
    assert tokenizer_fingerprint(a) == tokenizer_fingerprint(MockSP(64, "code"))
    print("  ✅ 指纹可区分不同 tokenizer、同构复现")
    return True


if __name__ == "__main__":
    tests = [
        test_alignment_build,
        test_matrix_and_hotswap,
        test_forward_train_cross_vocab,
        test_forward_train_same_vocab_backward_compat,
        test_fingerprint,
    ]
    passed = 0
    for t in tests:
        try:
            if t():
                passed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ❌ {t.__name__} 失败: {e}")
    print(f"\n{'=' * 60}\n结果: {passed}/{len(tests)} 通过")
    sys.exit(0 if passed == len(tests) else 1)
