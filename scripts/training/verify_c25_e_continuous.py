#!/usr/bin/env python3
"""C25-E 连续时间共振冒烟验证（2026-08-11）。

对比文档 2.11（刻意简化）"离散共振轮次替代连续动力学"——C25-E 修复：
- 离散轮次（round1 全量 → 不应期硬门 → max_rounds）的连续化替代
- 激活强度 a_i(t) = σ(β·binding_i(t)) 连续驱动"谁参与、权重多少"
  （同相强参与、异相退场）——替代不应期硬门的信息轮替
- 场随时间积分：F(t+dt) = F(t) + dt·Σ a_i·project(v_i)·conf_i
- 融合权重 = 时间平均激活（替代 final scores softmax）
- 收敛 = 相位绑定分布稳定（相位锁定）

安全性边界（C23 同款）：executive 判定（judge NLL 主信号）只消费 t=0
判定信号；连续激活不进入判定路径。

运行：python -u scripts/training/verify_c25_e_continuous.py
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataclasses import replace
from taiji.resonance.continuous import ContinuousResonance
from taiji.resonance.phasor import PhasorDynamics
from taiji.resonance.ensemble import ResonanceEnsemble
from taiji.resonance.field import ResonanceField
from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.config import TINY_TEST

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


# ─────────────────── Part 1：ContinuousResonance 核心单元 ───────────────────

def test_1_activation_monotonic_range():
    """activation 单调 + sigmoid 范围 (0,1)。"""
    ct = ContinuousResonance()
    b = torch.linspace(-1.0, 1.0, 21)
    a = ct.activation(b)
    mono = bool((a[1:] >= a[:-1]).all())  # binding↑ → activ↑
    in_range = bool((a > 0).all() and (a < 1).all())
    check("activation 单调递增", mono, f"a[-1]={a[-1]:.3f} a[0]={a[0]:.3f}")
    check("activation 范围 (0,1)", in_range, f"min={a.min():.4f} max={a.max():.4f}")


def test_2_activation_neutral_offset():
    """binding=act_offset → a=0.5（中性约定）。"""
    ct = ContinuousResonance(act_offset=0.2)
    a = ct.activation(torch.tensor([0.2]))
    check("binding=b0 → a≈0.5", abs(a.item() - 0.5) < 1e-5, f"a={a.item():.5f}")


def test_3_activation_continuous():
    """激活连续（无硬跳变）：Δb 小 → Δa 小。"""
    ct = ContinuousResonance(act_temp=2.0)  # 高温度接近硬门，仍应连续
    b = torch.linspace(-1.0, 1.0, 1001)
    a = ct.activation(b)
    da = (a[1:] - a[:-1]).abs().max()
    check("激活无硬跳变（Δa 有界）", da.item() < 0.05, f"max Δa={da.item():.4f}")


def test_4_weights_accum_integral():
    """权重累积 = 时间积分 Σdt·a·conf。"""
    ct = ContinuousResonance(dt=0.1)
    w = torch.zeros(3)
    activ = torch.tensor([1.0, 0.5, 0.0])
    conf = torch.tensor([1.0, 1.0, 1.0])
    for _ in range(10):
        w = ct.weights_accum(w, activ, conf, ct.dt)
    check("权重 = Σdt·a·conf", torch.allclose(w, torch.tensor([1.0, 0.5, 0.0]), atol=1e-5),
          f"w={w.tolist()}")


def test_5_converged_judge():
    """收敛判据：绑定 std 稳定 → True；波动 → False。"""
    ct = ContinuousResonance(conv_tol=0.05)
    stable = [torch.tensor([0.5, 0.4, 0.3]), torch.tensor([0.51, 0.41, 0.3])]
    volatile = [torch.tensor([0.9, 0.0, -0.9]), torch.tensor([0.5, 0.0, -0.5])]
    check("std 稳定 → 收敛", ct.converged(stable), f"Δstd={(stable[1].std()-stable[0].std()).abs():.4f}")
    check("std 波动 → 未收敛", not ct.converged(volatile),
          f"Δstd={(volatile[1].std()-volatile[0].std()).abs():.4f}")


def test_6_step_phase_activation_loop():
    """step：相位演化 → 激活联动（强耦合同频 → 绑定上升 → 激活上升）。"""
    ct = ContinuousResonance()
    ph = PhasorDynamics(omega_init=math.pi / 16, coupling_init=0.3, dt=0.1)
    ph.register_neurons(["a", "b"], phases=[0.0, 1.5])

    class FakeCoact:
        def get_coactivation(self, i, j):
            return 0.9

    b0 = ph.binding_tensor(["a", "b"], coactivation=FakeCoact())
    a0 = ct.activation(b0)
    for _ in range(40):
        ph.kuramoto_step(coupling_strength=0.3, active_ids=["a", "b"],
                         coactivation=FakeCoact(), dt=0.1)
    b1 = ph.binding_tensor(["a", "b"], coactivation=FakeCoact())
    a1 = ct.activation(b1)
    check("Kuramoto 牵引 → 绑定上升", b1[0] > b0[0], f"b0={b0[0]:.3f} → b1={b1[0]:.3f}")
    check("绑定上升 → 激活上升", a1[0] > a0[0], f"a0={a0[0]:.3f} → a1={a1[0]:.3f}")


# ─────────────────── Part 2：ensemble 集成（迷你装配） ───────────────────

def build_mini_ensemble():
    """4 神经元迷你 ensemble：n1/n2/n3 同相（0）、n4 异相（π）。"""
    base = replace(TINY_TEST, vocab_size=512, neuron_id="n1")
    cfgs = [replace(base, neuron_id=f"n{i}") for i in (1, 2, 3, 4)]
    neurons = {c.neuron_id: ResonanceNeuron(c) for c in cfgs}
    # judge_lm_head（general 判定头，C24 双头）——n1 挂上验证判定信号保留
    neurons["n1"].judge_lm_head = nn.Linear(cfgs[0].hidden_size, 512, bias=False)
    field = ResonanceField(dim=TINY_TEST.field_dim)
    ph = PhasorDynamics(omega_init=math.pi / 16, coupling_init=0.05, dt=0.1, binding_scale=0.3)
    # 直接 register_neurons（assign_phase 仅暂存，不构建 phasors 参数）
    ph.register_neurons(["n1", "n2", "n3", "n4"], phases=[0.0, 0.0, 0.0, math.pi])
    ens = ResonanceEnsemble(
        neurons=neurons,
        field=field,
        max_rounds=3,
        gamma_oscillator=ph,
    )
    return ens


def test_7_output_structure():
    """continuous_forward 输出结构完整 + 无 NaN。"""
    ens = build_mini_ensemble()
    emb = torch.randn(1, 8, TINY_TEST.base_embed_dim)
    r = ens.continuous_forward(shared_embeddings=emb, return_logits=True,
                               return_judge_logits=True)
    need = ["field_state", "final_scores", "continuous_weights", "n_steps",
            "phase_locked", "neuron_logits", "round1_judge_logits"]
    check("输出结构完整", all(k in r for k in need), f"keys={sorted(r.keys())}")
    nan = any(torch.isnan(v).any() for v in r["neuron_logits"].values()) if r.get("neuron_logits") else True
    check("logits 无 NaN", not nan)
    fs = r["field_state"]
    check("field_state 非空", fs is not None and float(fs.detach().norm()) > 0,
          f"norm={float(fs.detach().norm()):.4f}")
    check("n_steps ∈ [1,T]", 1 <= r["n_steps"] <= 8, f"n_steps={r['n_steps']} locked={r['phase_locked']}")


def test_8_weights_favor_same_phase():
    """融合权重 = 时间平均激活：同相群体（n1/n2/n3）权重大于异相（n4）。"""
    ens = build_mini_ensemble()
    emb = torch.randn(1, 8, TINY_TEST.base_embed_dim)
    r = ens.continuous_forward(shared_embeddings=emb)
    w = r["continuous_weights"]
    same_avg = (w.get("n1", 0.0) + w.get("n2", 0.0) + w.get("n3", 0.0)) / 3
    check("同相群体权重 > 异相", same_avg > w.get("n4", 0.0),
          f"same_avg={same_avg:.4f} n4={w.get('n4', 0.0):.4f} w={ {k: round(v,4) for k,v in w.items()} }")


def test_9_field_integrates():
    """场随时间积分：多次 forward 场有写入（非零贡献）。"""
    ens = build_mini_ensemble()
    emb = torch.randn(2, 8, TINY_TEST.base_embed_dim)
    r = ens.continuous_forward(shared_embeddings=emb)
    fs = r["field_state"]
    check("场状态 [B,D] 且 norm>0", fs.ndim == 2 and float(fs.detach().norm()) > 0,
          f"shape={tuple(fs.shape)} norm={float(fs.detach().norm()):.3f}")


def test_10_judge_signal_preserved():
    """判定信号保留：t=0 judge logits（C20v2 信号链）与离散 round1 一致收集。"""
    ens = build_mini_ensemble()
    emb = torch.randn(1, 8, TINY_TEST.base_embed_dim)
    r = ens.continuous_forward(shared_embeddings=emb, return_judge_logits=True)
    j = r.get("round1_judge_logits")
    check("n1 判定头 logits 收集", j is not None and "n1" in j,
          f"keys={list(j.keys()) if j else None}")
    if j and "n1" in j:
        lg = j["n1"]
        check("judge logits [B,L,512]", tuple(lg.shape) == (1, 8, 512),
              f"shape={tuple(lg.shape)}")


def test_11_forward_no_regression():
    """接入不破坏原 forward：离散路径仍正常输出。"""
    ens = build_mini_ensemble()
    emb = torch.randn(1, 8, TINY_TEST.base_embed_dim)
    r = ens.forward(shared_embeddings=emb, return_logits=True)
    check("forward 正常返回", r.get("field_state") is not None and r.get("round1_logits"),
          f"keys={sorted(r.keys())}")
    check("forward round1_logits 非空", bool(r.get("round1_logits")),
          f"n={len(r.get('round1_logits') or {})}")


def test_12_determinism():
    """同输入两次运行权重一致（连续积分确定性）。"""
    ens = build_mini_ensemble()
    emb = torch.randn(1, 8, TINY_TEST.base_embed_dim)
    r1 = ens.continuous_forward(shared_embeddings=emb)
    r2 = ens.continuous_forward(shared_embeddings=emb)
    w1, w2 = r1["continuous_weights"], r2["continuous_weights"]
    same = all(abs(w1[k] - w2[k]) < 1e-6 for k in w1)
    check("连续路径确定性（两次一致）", same, f"w1={ {k: round(v,4) for k,v in w1.items()} }")


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("C25-E 连续时间共振冒烟验证（Continuous-Time Resonance）", flush=True)
    print("=" * 60, flush=True)
    for fn in [test_1_activation_monotonic_range, test_2_activation_neutral_offset,
               test_3_activation_continuous, test_4_weights_accum_integral,
               test_5_converged_judge, test_6_step_phase_activation_loop,
               test_7_output_structure, test_8_weights_favor_same_phase,
               test_9_field_integrates, test_10_judge_signal_preserved,
               test_11_forward_no_regression, test_12_determinism]:
        fn()
    print("=" * 60, flush=True)
    print(f"结果: {passed}/{passed + failed} PASS", flush=True)
    sys.exit(0 if failed == 0 else 1)
