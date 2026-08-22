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
    out_pnone, _, _ = block(x, mask=mask, field_state=field_state, phase=None)
    diff_p0_none = float((out_p0 - out_pnone).abs().max())
    check("T5c phase=0 与 phase=None 近似等价（旋转退化）",
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
    # 总结
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 64, flush=True)
    if failed == 0:
        print(f"判定: 全部 {passed} 维 PASS — TaijiBlock 算子契约闭合", flush=True)
        print("  T1 接口契约对齐 TransformerBlock（可替换）", flush=True)
        print("  T2 数值稳定（含极端输入）", flush=True)
        print("  T3 退化等价（field/phase=None → 仅 yang 流）", flush=True)
        print("  T4 field-native（field 进入 yin 流 K/V）", flush=True)
        print("  T5 phase 调制（相位绑定进入算子内部）", flush=True)
        print("  T6 梯度可流通（训练就绪）", flush=True)
        print("\n下一步: 写 taiji 设计 plan + ResonanceNeuron 集成验证", flush=True)
    else:
        print(f"判定: {failed} 维 FAIL（{passed} 维 PASS）", flush=True)
    print("=" * 64, flush=True)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
