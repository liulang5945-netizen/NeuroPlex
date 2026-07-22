"""BioOSS（p/o 双神经元模型）验证脚本。

验证 BioOSS 机制：
1. ResonanceField.write_inhibit 乘法衰减掩码
2. get_effective_state 返回 state ⊙ inhibitory_mask
3. _leave_one_out_state 同时撤销 excitatory 和 inhibitory 贡献
4. ResonanceNeuron.is_inhibitory 属性
5. ResonanceEnsemble 按 is_inhibitory 分流（write vs write_inhibit）
6. Cortex.add_neuron 按 ~20% 比例生成 inhibitory

不修改现有 ckpt（保留 excitatory 基线），通过临时创建 inhibitory neuron 验证。

Usage:
    $env:TAIJI_TEST_MODE="1"
    python scripts/training/verify_biooss.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

os.environ.setdefault("TAIJI_TEST_MODE", "1")

from taiji.resonance.field import ResonanceField
from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.config import NeuronConfig, get_domain_neuron_config
from taiji.resonance.ensemble import ResonanceEnsemble
from taiji.resonance.tribal import CoactivationTracker


def _make_neuron(nid: str, neuron_type: str = "excitatory") -> ResonanceNeuron:
    cfg = get_domain_neuron_config("zh")
    cfg.neuron_id = nid
    cfg.neuron_type = neuron_type
    return ResonanceNeuron(cfg)


passed = 0
failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


print("=" * 60)
print("BioOSS 验证 — p/o 双神经元模型")
print("=" * 60)

# ── Test 1: write_inhibit 乘法衰减 ──
print("\n=== Test 1: write_inhibit 乘法衰减 ===")
field = ResonanceField(dim=64)
field.reset(batch_size=1)

# 先写入 excitatory 贡献
v_exc = torch.randn(64)
v_exc = v_exc / v_exc.norm()
field.write("exc_1", v_exc, scale=1.0)
state_after_exc = field.state.clone()
mask_after_exc = field.inhibitory_mask.clone()
check("write 后 state 非零", state_after_exc.norm() > 0.5)
check("write 后 mask 仍全 1", torch.allclose(mask_after_exc, torch.ones(64)))

# 写入 inhibitory 贡献
v_inh = torch.randn(64)
v_inh = v_inh / v_inh.norm()
weight_inh = 0.5
field.write_inhibit("inh_1", v_inh, weight=weight_inh)
mask_after_inh = field.inhibitory_mask.clone()
# inhibitory 不改变 state，只改变 mask
check("write_inhibit 不改 state", torch.allclose(field.state, state_after_exc, atol=1e-6))
# mask 衰减公式：decay_i = 1 - weight * |v_i| / |v_abs|.norm()
# v_inh 已 L2 归一化，|v_inh|.norm()=1，所以 decay_i = 1 - weight * |v_inh_i|
v_abs = v_inh.abs()
expected_decay = 1.0 - weight_inh * v_abs
expected_mask = expected_decay.clamp(min=0.0, max=1.0)
check(
    "write_inhibit 衰减 mask",
    torch.allclose(mask_after_inh, expected_mask, atol=1e-6),
    f"expected mean ~{expected_mask.mean().item():.3f}, got {mask_after_inh.mean().item():.3f}",
)

# ── Test 2: get_effective_state = state ⊙ mask ──
print("\n=== Test 2: get_effective_state = state ⊙ mask ===")
effective = field.get_effective_state()
manual_effective = field.state * field.inhibitory_mask
manual_norm = manual_effective / (manual_effective.norm() + 1e-8)
check(
    "effective_state = normalize(state ⊙ mask)",
    torch.allclose(effective, manual_norm, atol=1e-6),
    f"effective[:5]={effective[:5].tolist()}, manual[:5]={manual_norm[:5].tolist()}",
)

# ── Test 3: _leave_one_out_state 撤销 inhibitory 贡献 ──
print("\n=== Test 3: _leave_one_out_state 撤销 inhibitory 贡献 ===")
# 撤销 excitatory 贡献：只有 exc_1 时撤销后 state=0，effective=0
# 改为验证返回的是有限 tensor（不 NaN）
loo_exc = field._leave_one_out_state("exc_1")
check("loo(exc_1) 返回有限 tensor", torch.isfinite(loo_exc).all())

# 撤销 inhibitory 贡献：mask 应恢复接近 1（无抑制）
# loo_inh 是归一化的 effective state
loo_inh = field._leave_one_out_state("inh_1")
check("loo(inh_1) 返回有限 tensor", torch.isfinite(loo_inh).all())

# 关键验证：loo(inh_1) 的 effective state 应不同于 loo(unknown)
# 因为 loo(inh_1) 撤销了 mask 衰减，而 loo(unknown) 没有
loo_unknown = field._leave_one_out_state("nonexistent")
full_effective = field.get_effective_state()
check(
    "loo(unknown) = full effective_state",
    torch.allclose(loo_unknown, full_effective, atol=1e-6),
    f"loo_unknown[:5]={loo_unknown[:5].tolist()}, full[:5]={full_effective[:5].tolist()}",
)

# loo(inh_1) 应该 != full_effective（因为撤销了 inhibitory 衰减，mask 恢复到 1）
# effective = state ⊙ mask_loo，mask_loo(inh_1) ≈ 1，mask_full < 1
# 所以 loo(inh_1) 的 effective 应该更接近 state 本身（归一化后）
loo_inh_differs = not torch.allclose(loo_inh, full_effective, atol=1e-4)
check("loo(inh_1) != full effective（撤销了 inhibitory）", loo_inh_differs)

# ── Test 4: ResonanceNeuron.is_inhibitory 属性 ──
print("\n=== Test 4: ResonanceNeuron.is_inhibitory 属性 ===")
exc_neuron = _make_neuron("exc_1", "excitatory")
inh_neuron = _make_neuron("inh_1", "inhibitory")
check("excitatory neuron", exc_neuron.neuron_type == "excitatory")
check("inhibitory neuron", inh_neuron.neuron_type == "inhibitory")
check("is_inhibitory=False (excitatory)", exc_neuron.is_inhibitory is False)
check("is_inhibitory=True (inhibitory)", inh_neuron.is_inhibitory is True)

# ── Test 5: ensemble 按 is_inhibitory 分流 ──
print("\n=== Test 5: ensemble 按 is_inhibitory 分流 ===")
neurons = {"exc_1": exc_neuron, "inh_1": inh_neuron}
field2 = ResonanceField(dim=exc_neuron.config.field_dim)
coaction = CoactivationTracker()
ensemble = ResonanceEnsemble(neurons, field2, max_rounds=2, coaction=coaction)

# 构造 shared embedding 输入
B, L = 2, 4
base_embed_dim = exc_neuron.config.base_embed_dim
shared_emb = torch.randn(B, L, base_embed_dim)

# forward 前 mask 应全 1
check("forward 前 mask 全 1", torch.allclose(field2.inhibitory_mask, torch.ones_like(field2.inhibitory_mask)))

result = ensemble.forward(shared_embeddings=shared_emb, return_logits=False)

# forward 后，inh_1 应写入 _inhibit_contributions
has_inh_contrib = "inh_1" in field2._inhibit_contributions
has_exc_contrib = "exc_1" in field2._contributions
check("exc_1 贡献在 _contributions", has_exc_contrib)
check("inh_1 贡献在 _inhibit_contributions", has_inh_contrib)

# mask 应被 inhibitory 衰减（不再是全 1）
mask_changed = not torch.allclose(field2.inhibitory_mask, torch.ones_like(field2.inhibitory_mask))
check("forward 后 mask 被 inhibitory 衰减", mask_changed)

# state 应有 excitatory 贡献（inh 不写入 state）
state_nonzero = field2.state.norm() > 0.1
check("forward 后 state 有 excitatory 贡献", state_nonzero)

# ── Test 6: ensemble forward 不崩溃 + 生成 logits ──
print("\n=== Test 6: ensemble forward with logits (inhibitory + excitatory) ===")
result = ensemble.forward(shared_embeddings=shared_emb, return_logits=True)
check("forward 返回结果", result is not None)
check("final_scores 包含两 neuron", set(result.get("final_scores", {}).keys()) == {"exc_1", "inh_1"})
check("weighted_logits 存在", "weighted_logits" in result)

# ── Test 7: Cortex.add_neuron 按 ~20% 生成 inhibitory ──
print("\n=== Test 7: Cortex.add_neuron 按 ~20% 生成 inhibitory ===")
from taiji.brain.cortex import Cortex
from taiji.loader import assemble_cortex

cortex = assemble_cortex(
    neurons_dir="data/neurons",
    device="cpu",
    max_rounds=2,
    wire_bio_modules=False,
)
# assemble_cortex 返回 (cortex, tokenizer, modules) tuple
if isinstance(cortex, tuple):
    cortex = cortex[0]

# 现有 5 个 neuron 全是 excitatory
initial_inhibitory = sum(1 for n in cortex.neurons.values() if n.is_inhibitory)
check("初始 inhibitory 数=0", initial_inhibitory == 0)

# 临时切换 neurons_dir 避免污染 data/neurons
import tempfile
tmp_dir = tempfile.mkdtemp()
cortex.neurons_dir = tmp_dir

# 新建第 1 个 zh neuron（域内首 neuron，应 excitatory）
nid1 = cortex.add_neuron("zh")
n1_type = cortex.neurons[nid1].neuron_type
check("域内首 neuron = excitatory", n1_type == "excitatory", f"got {n1_type}")

# 新建第 2 个 zh neuron（inhibitory_ratio=0/1=0 < 0.2，应 inhibitory）
nid2 = cortex.add_neuron("zh")
n2_type = cortex.neurons[nid2].neuron_type
check("域内第 2 neuron = inhibitory (0% < 20%)", n2_type == "inhibitory", f"got {n2_type}")

# 新建第 3 个 zh neuron（inhibitory_ratio=1/2=0.5 >= 0.2，应 excitatory）
nid3 = cortex.add_neuron("zh")
n3_type = cortex.neurons[nid3].neuron_type
check("域内第 3 neuron = excitatory (50% >= 20%)", n3_type == "excitatory", f"got {n3_type}")

# 新建第 4 个 zh neuron（inhibitory_ratio=1/3=0.33 >= 0.2，应 excitatory）
nid4 = cortex.add_neuron("zh")
n4_type = cortex.neurons[nid4].neuron_type
check("域内第 4 neuron = excitatory (33% >= 20%)", n4_type == "excitatory", f"got {n4_type}")

# 新建第 5 个 zh neuron（inhibitory_ratio=1/4=0.25 >= 0.2，应 excitatory）
nid5 = cortex.add_neuron("zh")
n5_type = cortex.neurons[nid5].neuron_type
check("域内第 5 neuron = excitatory (25% >= 20%)", n5_type == "excitatory", f"got {n5_type}")

# 新建第 6 个 zh neuron（inhibitory_ratio=1/5=0.2 >= 0.2，应 excitatory）
nid6 = cortex.add_neuron("zh")
n6_type = cortex.neurons[nid6].neuron_type
check("域内第 6 neuron = excitatory (20% >= 20%)", n6_type == "excitatory", f"got {n6_type}")

# 新建第 7 个 zh neuron（inhibitory_ratio=1/6≈0.17 < 0.2，应 inhibitory）
nid7 = cortex.add_neuron("zh")
n7_type = cortex.neurons[nid7].neuron_type
check("域内第 7 neuron = inhibitory (17% < 20%)", n7_type == "inhibitory", f"got {n7_type}")

# 统计 7 个新 neuron 的 inhibitory 比例
new_nids = [nid1, nid2, nid3, nid4, nid5, nid6, nid7]
new_inh = sum(1 for n in new_nids if cortex.neurons[n].is_inhibitory)
check("7 个新 neuron 中 2 个 inhibitory (29%)", new_inh == 2, f"got {new_inh}")

# 清理临时 ckpt
import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)

# ── Summary ──
print("\n" + "=" * 60)
total = passed + failed
print(f"BioOSS 验证: {passed}/{total} PASSED")
if failed:
    print(f"FAILED: {failed}")
    sys.exit(1)
print("ALL CHECKS PASSED — BioOSS 机制验证通过")
print("=" * 60)
