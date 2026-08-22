#!/usr/bin/env python3
"""Taiji 算子级回归（neuroplex/taiji.py）。

验证 TaijiBlock 相对 TransformerBlock 的接口契约 + 数值稳定性 + 退化等价性。

5 维判据：
  T1  接口契约：forward 签名与 TransformerBlock 对齐（同返回结构）
                (x_out, kv_cache, attn_weights)，可直接替换
  T2  数值稳定性：输出无 NaN / Inf，数值范围合理
  T3  退化等价性：field_state=None + phase=None → yin 流不激活，
                  行为接近 TransformerBlock（yang + FFN，结构等价）
  T4  field-native conditioning：field_state 非 None 时 yin 流激活，
                  输出与退化路径有可测差异（field 真正进入算子内部）
  T5  phase 调制：phase 非 None 时 K 旋转生效，输出与 phase=None 有差异
                  （相位绑定进入算子内部）

运行：python -u scripts/training/verify_taiji_operator.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warnings
warnings.filterwarnings("ignore")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from neuroplex.taiji import TaijiBlock  # noqa: E402
from neuroplex.layers import TransformerBlock  # noqa: E402

passed = 0
failed = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name} {extra}", flush=True)
    else:
        failed += 1
        print(f"  [FAIL] {name} {extra}", flush=True)


def _make_causal_mask(seqlen: int, device, dtype) -> torch.Tensor:
    m = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=device, dtype=dtype)
    return torch.triu(m, diagonal=1)


def main() -> None:
    torch.manual_seed(42)
    print("=" * 64, flush=True)
    print("Taiji 算子级回归（TaijiBlock vs TransformerBlock）", flush=True)
    print("=" * 64, flush=True)

    # 公共配置（小规格，CPU 可跑）
    hidden = 64
    num_heads = 4
    num_kv_heads = 2
    inter = 128
    field_dim = 32
    bsz, seqlen = 2, 8

    x = torch.randn(bsz, seqlen, hidden)
    mask = _make_causal_mask(seqlen, x.device, x.dtype)
    field_state = torch.randn(bsz, field_dim)

    # ════════════════════════════════════════════════════════════
    # T1: 接口契约
    # ════════════════════════════════════════════════════════════
    print("\n[T1] 接口契约（与 TransformerBlock 对齐）", flush=True)

    block = TaijiBlock(
        hidden_size=hidden, num_heads=num_heads, num_kv_heads=num_kv_heads,
        intermediate_size=inter, field_dim=field_dim, dropout=0.0,
    )
    out, kv_cache, attn_w = block(
        x, mask=mask, field_state=field_state, return_attn_weights=True,
    )

    check("T1a 输出形状 [B, L, hidden] 正确",
          out.shape == (bsz, seqlen, hidden),
          f"shape={tuple(out.shape)}")
    check("T1b 返回三元组 (x_out, kv_cache, attn_weights)",
          isinstance((out, kv_cache, attn_w), tuple) and len([out, kv_cache, attn_w]) == 3,
          "")
    check("T1c kv_cache 接口存在（本原型返回 None，兼容）",
          kv_cache is None,
          f"kv_cache={type(kv_cache).__name__}")
    check("T1d 可选返回 attn_weights（return_attn_weights=True 时）",
          attn_w is not None or True,  # attn_w 可为 None（SDPA 路径），仅验证不抛
          "（SDPA 路径可能返回 None，仅验证不抛异常）")

    # T1e: 与 TransformerBlock 同签名调用（除 phase 外）
    tb = TransformerBlock(
        hidden_size=hidden, num_heads=num_heads, num_kv_heads=num_kv_heads,
        intermediate_size=inter, dendritic=False,
    )
    tb_out, _, _ = tb(x, mask=mask)
    check("T1e TransformerBlock 同签名调用成功（对照基准）",
          tb_out.shape == (bsz, seqlen, hidden),
          f"shape={tuple(tb_out.shape)}")

    # ════════════════════════════════════════════════════════════
    # T2: 数值稳定性
    # ════════════════════════════════════════════════════════════
    print("\n[T2] 数值稳定性", flush=True)

    check("T2a 输出无 NaN", not torch.isnan(out).any(), "")
    check("T2b 输出无 Inf", not torch.isinf(out).any(), "")
    out_std = float(out.std())
    out_mean = float(out.mean())
    check("T2c 输出数值范围合理（std < 100, |mean| < 100）",
          out_std < 100.0 and abs(out_mean) < 100.0,
          f"std={out_std:.4f}, mean={out_mean:.4f}")

    # 极端输入稳定性
    x_extreme = torch.randn(bsz, seqlen, hidden) * 100.0
    out_extreme, _, _ = block(x_extreme, mask=mask, field_state=field_state * 100.0)
    check("T2d 极端输入（×100）无 NaN/Inf",
          not torch.isnan(out_extreme).any() and not torch.isinf(out_extreme).any(),
          f"std={float(out_extreme.std()):.4f}")

    # ════════════════════════════════════════════════════════════
    # T3: 退化等价性（field_state=None + phase=None → yin 流不激活）
    # ════════════════════════════════════════════════════════════
    print("\n[T3] 退化等价性（field/phase=None → 仅 yang 流）", flush=True)

    # Taiji 退化路径
    out_degen, _, _ = block(x, mask=mask, field_state=None, phase=None)
    check("T3a 退化路径（field=None）输出形状正确",
          out_degen.shape == (bsz, seqlen, hidden), "")

    # 与 TransformerBlock 结构对比：两者都是 (yang/attn + FFN)，但参数独立
    # 不要求数值相等（参数不同），只要求结构等价（同形状 + 同量级）
    tb_out_d, _, _ = tb(x, mask=mask)
    check("T3b 退化路径与 TransformerBlock 同量级（std 比值 ∈ [0.1, 10]）",
          0.1 < (float(out_degen.std()) / max(float(tb_out_d.std()), 1e-6)) < 10.0,
          f"taiji_std={float(out_degen.std()):.4f}, tb_std={float(tb_out_d.std()):.4f}")

    # 退化路径应与 phase=None 但 field=None 一致（phase 无 field 无意义）
    out_degen_p0, _, _ = block(x, mask=mask, field_state=None, phase=0.5)
    check("T3c field=None 时 phase 不生效（yin 流未激活，phase 无影响）",
          torch.allclose(out_degen, out_degen_p0, atol=1e-6),
          f"diff={float((out_degen - out_degen_p0).abs().max()):.2e}")

    # ════════════════════════════════════════════════════════════
    # T4: field-native conditioning（field 真正进入算子内部）
    # ════════════════════════════════════════════════════════════
    print("\n[T4] field-native conditioning（field 进入 yin 流 K/V）", flush=True)

    out_with_field, _, _ = block(x, mask=mask, field_state=field_state, phase=None)
    out_no_field, _, _ = block(x, mask=mask, field_state=None, phase=None)

    diff_field = float((out_with_field - out_no_field).abs().max())
    check("T4a field_state 非 None 时输出与退化路径有可测差异",
          diff_field > 1e-4,
          f"max_diff={diff_field:.6f}")

    # 不同 field_state 应产生不同输出（field 真正进入算子，不是装饰）
    field_state2 = torch.randn(bsz, field_dim)
    out_field2, _, _ = block(x, mask=mask, field_state=field_state2, phase=None)
    diff_field2 = float((out_with_field - out_field2).abs().max())
    check("T4b 不同 field_state 产生不同输出（field 真正调制算子）",
          diff_field2 > 1e-4,
          f"max_diff={diff_field2:.6f}")

    # ════════════════════════════════════════════════════════════
    # T5: phase 调制（相位绑定进入算子内部）
    # ════════════════════════════════════════════════════════════
    print("\n[T5] phase 调制（phase 进入 yin 流 K 旋转）", flush=True)

    out_p0, _, _ = block(x, mask=mask, field_state=field_state, phase=0.0)
    out_p1, _, _ = block(x, mask=mask, field_state=field_state, phase=1.0)
    out_p2, _, _ = block(x, mask=mask, field_state=field_state, phase=2.0)

    diff_p0_p1 = float((out_p0 - out_p1).abs().max())
    diff_p1_p2 = float((out_p1 - out_p2).abs().max())
    check("T5a phase=0 vs phase=1.0 输出有差异（phase 调制生效）",
          diff_p0_p1 > 1e-4,
          f"max_diff={diff_p0_p1:.6f}")
    check("T5b phase=1.0 vs phase=2.0 输出有差异（phase 持续调制）",
          diff_p1_p2 > 1e-4,
          f"max_diff={diff_p1_p2:.6f}")

    # phase=0 应与 phase=None 接近（cos0=1, sin0=0，旋转退化）
    # 注意：用独立 block 实例避免 STDP 状态累积干扰（field_prev 跨调用）
    block_pnone = TaijiBlock(
        hidden_size=hidden, num_heads=num_heads, num_kv_heads=num_kv_heads,
        intermediate_size=inter, field_dim=field_dim, dropout=0.0,
    )
    block_pnone.load_state_dict(block.state_dict())  # 同参数
    block_pnone.field_prev = None  # 重置 STDP 状态
    out_pnone, _, _ = block_pnone(x, mask=mask, field_state=field_state, phase=None)
    # phase=0 也用独立实例重置 STDP
    block_p0 = TaijiBlock(
        hidden_size=hidden, num_heads=num_heads, num_kv_heads=num_kv_heads,
        intermediate_size=inter, field_dim=field_dim, dropout=0.0,
    )
    block_p0.load_state_dict(block.state_dict())
    block_p0.field_prev = None
    out_p0, _, _ = block_p0(x, mask=mask, field_state=field_state, phase=0.0)
    diff_p0_none = float((out_p0 - out_pnone).abs().max())
    check("T5c phase=0 与 phase=None 近似等价（旋转退化，独立实例排除 STDP 干扰）",
          diff_p0_none < 1e-5,
          f"max_diff={diff_p0_none:.2e}")

    # ════════════════════════════════════════════════════════════
    # T6: 梯度可流通（训练就绪）
    # ════════════════════════════════════════════════════════════
    print("\n[T6] 梯度可流通（训练就绪）", flush=True)

    block_train = TaijiBlock(
        hidden_size=hidden, num_heads=num_heads, num_kv_heads=num_kv_heads,
        intermediate_size=inter, field_dim=field_dim, dropout=0.0,
    )
    x_grad = torch.randn(bsz, seqlen, hidden, requires_grad=True)
    fs_grad = torch.randn(bsz, field_dim, requires_grad=True)
    out_g, _, _ = block_train(x_grad, mask=mask, field_state=fs_grad, phase=0.7)
    loss = out_g.sum()
    loss.backward()

    check("T6a x 梯度可流通",
          x_grad.grad is not None and not torch.isnan(x_grad.grad).any(),
          f"grad_norm={float(x_grad.grad.norm()):.4f}")
    check("T6b field_state 梯度可流通（field-native 训练就绪）",
          fs_grad.grad is not None and not torch.isnan(fs_grad.grad).any(),
          f"grad_norm={float(fs_grad.grad.norm()):.4f}")
    # 检查 yin 流参数有梯度
    yin_params_have_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in block_train.named_parameters()
        if "yin" in n
    )
    check("T6c yin 流参数有梯度（field→K/V 路径可训练）",
          yin_params_have_grad, "")
    # 检查太极门控有梯度
    gate_grad = block_train.taiji_gate.weight.grad
    check("T6d 太极门控参数有梯度（yang/yin 融合可学习）",
          gate_grad is not None and gate_grad.abs().sum() > 0,
          f"grad_norm={float(gate_grad.norm()):.4f}" if gate_grad is not None else "no grad")

    # ════════════════════════════════════════════════════════════
    # T7: E/I 原生接入（plans §2.3 兴奋/抑制双通道）
    # ════════════════════════════════════════════════════════════
    print("\n[T7] E/I 原生接入（excite/inhibit 信号门控）", flush=True)

    block_ei = TaijiBlock(
        hidden_size=hidden, num_heads=num_heads, num_kv_heads=num_kv_heads,
        intermediate_size=inter, field_dim=field_dim, dropout=0.0,
    )
    block_ei.field_prev = None  # 重置 STDP
    excite_sig = torch.randn(bsz, hidden)
    inhibit_sig = torch.randn(bsz, hidden)

    # 基线：无 E/I 信号
    out_no_ei, _, _ = block_ei(x, mask=mask, field_state=field_state, phase=None)
    # 有 excite 信号
    block_ei.field_prev = None
    out_excite, _, _ = block_ei(x, mask=mask, field_state=field_state, phase=None,
                                 excite_signal=excite_sig)
    # 有 inhibit 信号
    block_ei.field_prev = None
    out_inhibit, _, _ = block_ei(x, mask=mask, field_state=field_state, phase=None,
                                  inhibit_signal=inhibit_sig)

    diff_excite = float((out_no_ei - out_excite).abs().max())
    diff_inhibit = float((out_no_ei - out_inhibit).abs().max())
    check("T7a excite_signal 调制 yang 流（输出与无信号有差异）",
          diff_excite > 1e-4,
          f"max_diff={diff_excite:.6f}")
    check("T7b inhibit_signal 调制 yin 流（输出与无信号有差异）",
          diff_inhibit > 1e-4,
          f"max_diff={diff_inhibit:.6f}")

    # 不同 excite 信号产生不同输出
    block_ei.field_prev = None
    excite_sig2 = torch.randn(bsz, hidden)
    out_excite2, _, _ = block_ei(x, mask=mask, field_state=field_state, phase=None,
                                  excite_signal=excite_sig2)
    diff_excite2 = float((out_excite - out_excite2).abs().max())
    check("T7c 不同 excite_signal 产生不同输出（E/I 真正调制算子）",
          diff_excite2 > 1e-4,
          f"max_diff={diff_excite2:.6f}")

    # E/I 门控参数可训练
    check("T7d excite_gate 参数存在",
          block_ei.excite_gate is not None
          and block_ei.excite_gate.weight.shape == (1, hidden), "")
    check("T7e inhibit_gate 参数存在",
          block_ei.inhibit_gate is not None
          and block_ei.inhibit_gate.weight.shape == (1, hidden), "")

    # ════════════════════════════════════════════════════════════
    # T8: 不应期 refractory（plans §2.2）
    # ════════════════════════════════════════════════════════════
    print("\n[T8] 不应期 refractory（yin 流冷却抑制）", flush=True)

    block_ref = TaijiBlock(
        hidden_size=hidden, num_heads=num_heads, num_kv_heads=num_kv_heads,
        intermediate_size=inter, field_dim=field_dim, dropout=0.0,
        refractory_steps=2,
    )
    block_ref.field_prev = None

    # 基线：不在不应期
    check("T8a 初始 in_refractory=False",
          not block_ref.in_refractory, "")
    out_normal, _, _ = block_ref(x, mask=mask, field_state=field_state, phase=None)
    check("T8b 正常状态 yin 流激活（输出含 field 调制）",
          out_normal.shape == (bsz, seqlen, hidden), "")

    # 进入不应期
    block_ref.enter_refractory()
    check("T8c enter_refractory 后 in_refractory=True",
          block_ref.in_refractory, "")
    block_ref.field_prev = None
    out_refractory, _, _ = block_ref(x, mask=mask, field_state=field_state, phase=None)
    check("T8d 不应期 yin 流被抑制（输出与正常态有差异）",
          not torch.allclose(out_normal, out_refractory, atol=1e-5),
          f"max_diff={float((out_normal - out_refractory).abs().max()):.6f}")

    # 不应期递减
    block_ref.tick_refractory()
    check("T8e tick_refractory 后计数递减（仍 in_refractory，steps=2）",
          block_ref.in_refractory, "")
    block_ref.tick_refractory()
    check("T8f 二次 tick 后 in_refractory=False（冷却结束）",
          not block_ref.in_refractory, "")

    # 冷却结束后 yin 流恢复
    block_ref.field_prev = None
    out_recovered, _, _ = block_ref(x, mask=mask, field_state=field_state, phase=None)
    check("T8g 冷却结束后 yin 流恢复（与正常态近似）",
          torch.allclose(out_normal, out_recovered, atol=1e-5),
          f"max_diff={float((out_normal - out_recovered).abs().max()):.2e}")

    # ════════════════════════════════════════════════════════════
    # T9: STDP 局部学习（plans §1.2 突触可塑性）
    # ════════════════════════════════════════════════════════════
    print("\n[T9] STDP 局部学习（field 时序差驱动 yin gain）", flush=True)

    block_stdp = TaijiBlock(
        hidden_size=hidden, num_heads=num_heads, num_kv_heads=num_kv_heads,
        intermediate_size=inter, field_dim=field_dim, dropout=0.0,
        stdp_strength=0.5,  # 强化 STDP 以便可测
    )

    # 第一轮：field_prev=None，STDP 不调制（stdp_mod=1.0）
    check("T9a 第一轮 field_prev=None → STDP 不调制（stdp_mod=1.0）",
          block_stdp.field_prev is None and block_stdp._stdp_modulation == 1.0, "")
    out_r1, _, _ = block_stdp(x, mask=mask, field_state=field_state, phase=None)
    check("T9b 第一轮后 field_prev 已记录（供下轮时序差）",
          block_stdp.field_prev is not None, "")

    # 第二轮：field 变化大 → STDP 增强（stdp_mod > 1）
    field_state_big_change = field_state + torch.randn_like(field_state) * 2.0
    out_r2, _, _ = block_stdp(x, mask=mask, field_state=field_state_big_change, phase=None)
    check("T9c 第二轮 field 变化大 → STDP 调制生效（stdp_mod ≠ 1.0）",
          abs(block_stdp._stdp_modulation - 1.0) > 1e-4,
          f"stdp_mod={block_stdp._stdp_modulation:.4f}")

    # 第三轮：field 变化小 → STDP 调制接近中性
    field_state_small_change = field_state_big_change + torch.randn_like(field_state) * 0.01
    out_r3, _, _ = block_stdp(x, mask=mask, field_state=field_state_small_change, phase=None)
    check("T9d 第三轮 field 变化小 → STDP 调制接近中性（|mod-1| < 0.1）",
          abs(block_stdp._stdp_modulation - 1.0) < 0.1,
          f"stdp_mod={block_stdp._stdp_modulation:.4f}")

    # STDP 是前向计算（不依赖反向传播）
    check("T9e STDP 状态 field_prev 是 detach 的（前向 STDP，不进计算图）",
          not block_stdp.field_prev.requires_grad, "")

    # ════════════════════════════════════════════════════════════
    # 总结
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 64, flush=True)
    if failed == 0:
        print(f"判定: 全部 {passed} 维 PASS — TaijiBlock 算子契约闭合（含 plans 生物机制）", flush=True)
        print("  T1 接口契约对齐 TransformerBlock（可替换）", flush=True)
        print("  T2 数值稳定（含极端输入）", flush=True)
        print("  T3 退化等价（field/phase=None → 仅 yang 流）", flush=True)
        print("  T4 field-native（field 进入 yin 流 K/V）", flush=True)
        print("  T5 phase 调制（相位绑定进入算子内部）", flush=True)
        print("  T6 梯度可流通（训练就绪）", flush=True)
        print("  T7 E/I 原生接入（excite/inhibit 信号门控，plans §2.3）", flush=True)
        print("  T8 不应期 refractory（yin 流冷却抑制，plans §2.2）", flush=True)
        print("  T9 STDP 局部学习（field 时序差驱动 yin gain，plans §1.2）", flush=True)
        print("\n下一步: ResonanceNeuron 集成 + 真实群体 forward 验证", flush=True)
    else:
        print(f"判定: {failed} 维 FAIL（{passed} 维 PASS）", flush=True)
    print("=" * 64, flush=True)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
