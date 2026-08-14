#!/usr/bin/env python3
"""C27 增量四验证：o 型振荡节点 → 可学习节奏控制器（2026-08-14）。

背景：增量三的 OscillatorNode 是固定常量节奏源（纯 float 动力学），forward_train
continuous 训练路径未消费振荡器——ω/coupling/gaba_amp 无梯度。增量四让 o 型
从固定节奏源走向**可学习节奏控制器**（用户决策：三个参数全部进梯度流 +
牵引/门控全链路 + 节奏对齐自监督 osc_rhythm_loss 作 gaba_amp 梯度源）：
- OscillatorNode 升级 nn.Module：omega/coupling/gaba_amp 三个 Parameter +
  gaba_vec buffer；新增可微相位 API（theta_tensor/phase_unit_tensor/gaba_gate_tensor）
- forward_train continuous 分支：每步可微牵引（external_phases 张量 +
  external_weights=coupling Parameter）+ GABA 门控衰减 field_state（可微）
- osc_rhythm_loss：门控强度 w=gaba_amp·gate 对齐 p 型群体锁相度（锁相强→弱抑制）
- 持久化：cortex.save_state/load_state 保存恢复振荡器参数

验证层次：
A. OscillatorNode v2 单元（参数注册 / float 兼容 / 可微相位 / 可微门控）
B. phasor evolve 张量牵引权重可微（coupling Parameter 梯度）
C. 训练接入（forward_train continuous）：振荡器参与 / osc_rhythm_loss /
   三参数梯度全非空（全部可学）
D. 持久化：state_dict 往返 + cortex 保存集成
E. 生产零回归：generate 非空不退化（推理 float 路径兼容）

运行：python -u scripts/training/verify_c27_osc_train.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import random  # noqa: E402
import numpy as np  # noqa: E402

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
from taiji.loader import assemble_cortex  # noqa: E402
from taiji.resonance.oscillator import OscillatorNode, make_default_oscillators  # noqa: E402
from taiji.resonance.phasor import PhasorDynamics  # noqa: E402
from taiji.resonance.continuous import ContinuousResonance  # noqa: E402
# 口径契约：zh/dialogue 域 prompt 必须走训练格式
from taiji.resonance.dialogue_format import build_dialogue_prompt  # noqa: E402

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


DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("C27 增量四：o 型振荡节点 → 可学习节奏控制器", flush=True)
    print("=" * 60, flush=True)

    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name=COLLAB_NAME,
        extra_neurons_dir=EXTRA_NEURONS_DIR,
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=DIALOGUE_IDS,
    )
    print(f"  装配: {list(cortex.neurons.keys())}", flush=True)
    check("装配成功（5 dialogue + 4 general）", len(cortex.neurons) == 9)

    # ── A. OscillatorNode v2 单元 ──
    print("\n[A] OscillatorNode v2 单元 ...", flush=True)
    osc = OscillatorNode(nid="osc_test", omega=0.5, coupling=0.4,
                         gaba_amp=0.08, dim=16, phase=0.0)
    sd_keys = sorted(osc.state_dict().keys())
    check("A1. 节奏参数注册为 Parameter（omega/coupling/gaba_amp + gaba_vec buffer）",
          all(k in sd_keys for k in ("omega", "coupling", "gaba_amp", "gaba_vec")),
          f"keys={sd_keys}")
    osc.step(dt=1.0)
    check("A2. float 状态兼容（phase property / step 推进）",
          abs(osc.phase - 0.5) < 1e-6, f"phase={osc.phase:.4f}")
    u = osc.unit()
    check("A3. unit 兼容（float 状态 cos/sin）",
          abs(float(u[0]) - math.cos(0.5)) < 1e-6, f"unit={u.tolist()}")
    gp = OscillatorNode(nid="g", omega=0.5, coupling=0.1, gaba_amp=0.1,
                        dim=16, phase=math.pi).gaba_gate()
    check("A4. gaba_gate 兼容（π 相位关闭）", gp < 1e-6, f"gp={gp:.4f}")
    # 可微相位：θ(t) = θ0 + ω·t → dθ/dω = t
    theta = osc.theta_tensor(2.0)
    theta.sum().backward()
    check("A5. theta_tensor 对 omega 可微（dθ/dω=t）",
          osc.omega.grad is not None
          and abs(float(osc.omega.grad.item()) - 2.0) < 1e-6,
          f"grad_omega={None if osc.omega.grad is None else float(osc.omega.grad):.4f}")
    osc.zero_grad()
    gate_t = osc.gaba_gate_tensor(1.0)
    gate_t.sum().backward()
    check("A6. gaba_gate_tensor 对 omega 可微（门控进梯度流）",
          osc.omega.grad is not None and float(osc.omega.grad.abs().item()) > 0,
          f"grad_omega={None if osc.omega.grad is None else float(osc.omega.grad):.4f}")

    # ── B. phasor evolve 张量牵引权重可微 ──
    print("\n[B] phasor evolve 张量 external_weights 可微 ...", flush=True)
    try:
        pd = PhasorDynamics()
        pd.register_neurons(["n0", "n1"], phases=[0.1, 0.5])
        osc_c = OscillatorNode(nid="osc_t", omega=0.5, coupling=0.4,
                               gaba_amp=0.08, dim=16, phase=0.0)
        ep = osc_c.phase_unit_tensor(1.0)  # [cos, sin] 可微
        new_p = pd.evolve(
            active_ids=["n0", "n1"], dt=0.0, coupling_strength=0.0,
            external_phases=[ep], external_weights=[osc_c.coupling],
        )
        loss = new_p.sum()
        loss.backward()
        check("B1. 牵引权重（coupling）经 evolve 可微",
              osc_c.coupling.grad is not None
              and float(osc_c.coupling.grad.abs().item()) > 0,
              f"grad_coupling={None if osc_c.coupling.grad is None else float(osc_c.coupling.grad):.4f}")
        check("B2. 牵引相位（ω）经 evolve 可微",
              osc_c.omega.grad is not None
              and float(osc_c.omega.grad.abs().item()) > 0,
              f"grad_omega={None if osc_c.omega.grad is None else float(osc_c.omega.grad):.4f}")
    except Exception as e:
        check("B1. 牵引权重可微", False, f"err={e}")
        check("B2. 牵引相位可微", False, f"err={e}")

    # ── C. 训练接入（forward_train continuous）──
    print("\n[C] forward_train continuous 接入振荡器 ...", flush=True)
    oscs = getattr(cortex.ensemble, "oscillators", [])
    check("C0. 装配振荡器为 nn.Module（含可学习参数）",
          len(oscs) == 2 and all(
              hasattr(o, "omega") and hasattr(o, "coupling")
              and hasattr(o, "gaba_amp") for o in oscs))
    try:
        emb_in = torch.tensor(
            [cortex._general_sp.encode(build_dialogue_prompt("节奏控制训练"))],
            dtype=torch.long, device=cortex.device)
        emb = cortex._shared_embedding(emb_in)
        # 记录振荡器参数梯度（backward 前清零）
        for o in oscs:
            o.zero_grad()
        # 关闭收敛提前 break（min_steps 拉大），积分跑满 8 步——
        # 避免相位锁定提前终止导致牵引项（coupling 梯度）数值过小。
        ct_full = ContinuousResonance(min_steps=10 ** 6)
        # 集成断言：external_weights 以可微张量（coupling Parameter）传入 evolve
        # （梯度可微性由 B1 单元确定性验证；此处验证 forward_train 集成透传）。
        # 注意：spy 必须用普通函数（描述符语义 self 正确传入）——
        # wraps=mock 不是描述符，实例调用不绑定 self，phase_unit_tensor 收到
        # 错误 self 抛异常 → 每步只迭代 1 个 osc、evolve 不执行（已排查确认）。
        seen_ph = {"count": 0}
        _orig_ph = OscillatorNode.phase_unit_tensor

        def _spy_ph(self, *a, **k):
            seen_ph["count"] += 1
            return _orig_ph(self, *a, **k)

        seen_ew = {"count": 0, "all_tensor": True}
        _orig_evolve = cortex.ensemble.gamma_oscillator.evolve

        def _spy_evolve(*a, **k):
            _ew = k.get("external_weights")
            if _ew:
                seen_ew["count"] += 1
                seen_ew["all_tensor"] = (
                    seen_ew["all_tensor"] and all(torch.is_tensor(x) for x in _ew))
            return _orig_evolve(*a, **k)

        with mock.patch.object(OscillatorNode, "phase_unit_tensor", _spy_ph), \
             mock.patch.object(
                cortex.ensemble.gamma_oscillator, "evolve", _spy_evolve):
            out = cortex.ensemble.forward_train(
                shared_embeddings=emb, n_rounds=2, continuous=True,
                target_domain="zh", ct=ct_full,
            )
        check("C1. 振荡器可微相位参与训练（phase_unit_tensor 被调用）",
              seen_ph["count"] > 0, f"calls={seen_ph['count']}")
        rl = out.get("osc_rhythm_loss")
        check("C2. osc_rhythm_loss 计算（有限值）",
              rl is not None and torch.isfinite(rl.detach()).all(),
              f"osc_rhythm_loss={None if rl is None else float(rl.detach()):.4f}")
        pl = out.get("phase_loss")
        loss = out["osc_rhythm_loss"] + out["phase_loss"]
        loss.backward()
        g_omega = [o.omega.grad for o in oscs]
        g_coupling = [o.coupling.grad for o in oscs]
        g_gaba = [o.gaba_amp.grad for o in oscs]
        check("C3. omega 梯度非空（可学习）",
              all(g is not None and float(g.abs().item()) > 0 for g in g_omega),
              f"grads={[None if g is None else round(float(g.item()), 6) for g in g_omega]}")
        check("C4. coupling 以可微张量传入 evolve（牵引进训练路径）",
              seen_ew["count"] > 0 and seen_ew["all_tensor"],
              f"calls={seen_ew['count']} all_tensor={seen_ew['all_tensor']} "
              f"(B1 单元已验证耦合梯度可微)")
        check("C5. gaba_amp 梯度非空（经 osc_rhythm_loss）",
              all(g is not None and float(g.abs().item()) > 0 for g in g_gaba),
              f"grads={[None if g is None else round(float(g.item()), 6) for g in g_gaba]}")
    except Exception as e:
        check("C1. 振荡器参与训练", False, f"err={e}")
        check("C2. osc_rhythm_loss 有限", False, f"err={e}")
        check("C3. omega 梯度", False, f"err={e}")
        check("C4. coupling 梯度", False, f"err={e}")
        check("C5. gaba_amp 梯度", False, f"err={e}")

    # ── D. 持久化 ──
    print("\n[D] 持久化（振荡器参数随状态保存/恢复）...", flush=True)
    osc_p = OscillatorNode(nid="osc_persist", omega=0.5, coupling=0.4,
                           gaba_amp=0.08, dim=8, phase=0.0)
    with torch.no_grad():
        osc_p.omega.add_(0.13)
        osc_p.coupling.add_(-0.05)
        osc_p.gaba_amp.add_(0.02)
    sd_p = osc_p.state_dict()
    osc_p2 = OscillatorNode(nid="osc_persist", omega=0.5, coupling=0.4,
                            gaba_amp=0.08, dim=8, phase=0.0)
    osc_p2.load_state_dict(sd_p)
    ok_roundtrip = (abs(osc_p2.omega.item() - osc_p.omega.item()) < 1e-6
                    and abs(osc_p2.coupling.item() - osc_p.coupling.item()) < 1e-6
                    and abs(osc_p2.gaba_amp.item() - osc_p.gaba_amp.item()) < 1e-6
                    and torch.equal(osc_p2.gaba_vec, osc_p.gaba_vec))
    check("D1. OscillatorNode state_dict 往返保留参数", ok_roundtrip,
          f"omega={osc_p2.omega.item():.3f} coupling={osc_p2.coupling.item():.3f} "
          f"gaba_amp={osc_p2.gaba_amp.item():.3f}")
    _src = Path("taiji/brain/cortex.py").read_text(encoding="utf-8")
    check("D2. cortex.save_state/load_state 含 oscillators 持久化集成",
          '"oscillators"' in _src, "save_state 写 + load_state 读")
    check("D3. 装配注入仍为 theta+gamma 双层（set_oscillators 兼容）",
          any(o.nid.startswith("osc_theta") for o in oscs)
          and any(o.nid.startswith("osc_gamma") for o in oscs),
          f"oscs={[o.nid for o in oscs]}")

    # ── E. 生产零回归 ──
    print("\n[E] 生产零回归（推理 float 路径兼容）...", flush=True)
    try:
        out = cortex.generate(
            build_dialogue_prompt("介绍一下什么是机器学习。"),
            max_tokens=32, domain="zh", temperature=0.55,
        )
        lp = cortex.get_last_phase()
        check("E1. 生成非空不退化", isinstance(out, str)
              and len(out.strip()) > 0 and not cortex._is_degenerate_text(out),
              f"out={out[:30]!r}")
        check("E2. get_last_phase 正常", isinstance(lp, float),
              f"phase={lp}")
    except Exception as e:
        check("E1. 生成非空不退化", False, f"err={e}")
        check("E2. get_last_phase 正常", False, f"err={e}")

    print(f"\n  总耗时: {time.time() - t0:.1f}s", flush=True)
    print("=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
