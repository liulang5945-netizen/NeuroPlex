#!/usr/bin/env python3
"""C27 增量三验证：BioOSS p/o 双神经元模型（2026-08-14）。

背景：态极 neuron 已有 excitatory/inhibitory 亚型（单维标记）。BioOSS 把
角色分工正式化——p（投射型，内容生成，现有全部 neuron）+ o（振荡型，节奏
源，轻量合成 OscillatorNode 无需训练）：
- o 型相位推进（theta 慢 + gamma 快双层，用户决策）
- o 型对 p 型 Kuramoto 相位牵引（驱动锁相）
- o 型 GABA 式节奏门控（write_inhibit 半周期窗口，用户决策）
- KoPE phase_code 纳入振荡段（节奏中心）

验证层次：
A. OscillatorNode 单元：相位推进 / unit 向量 / gaba_gate 半周期窗口
B. 装配注入：cortex.ensemble.oscillators 非空（theta+gamma 双层）
C. 相位牵引：PhasorDynamics.evolve 带 external_phases 牵引生效（相位偏转）
D. GABA 门控：continuous_forward 后 field.inhibitory_mask 被振荡门控调制
E. phase_code 含振荡段：维度 = 2N + 2M（节奏中心进表征）
F. 生产零破坏：generate 非空、get_last_phase 正常

运行：python -u scripts/training/verify_c27_biooss.py
"""

from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import random  # noqa: E402
import numpy as np  # noqa: E402

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
from neuroplex.loader import assemble_cortex  # noqa: E402
from neuroplex.resonance.oscillator import OscillatorNode, make_default_oscillators  # noqa: E402
from neuroplex.resonance.phasor import PhasorDynamics  # noqa: E402
# 口径契约：zh/dialogue 域 prompt 必须走训练格式
from neuroplex.resonance.dialogue_format import build_dialogue_prompt  # noqa: E402

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
    print("C27 增量三：BioOSS p/o 双神经元模型", flush=True)
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

    # ── A. OscillatorNode 单元 ──
    print("\n[A] OscillatorNode 单元 ...", flush=True)
    osc = OscillatorNode(nid="osc_test", omega=0.5, coupling=0.4,
                         gaba_amp=0.08, dim=16, phase=0.0)
    osc.step(dt=1.0)
    check("A1. 相位推进（θ += ω·dt）", abs(osc.phase - 0.5) < 1e-6,
          f"phase={osc.phase:.4f}")
    u = osc.unit()
    check("A2. unit 单位向量（cos/sin）",
          abs(float(u[0]) - math.cos(0.5)) < 1e-6
          and abs(float(u[1]) - math.sin(0.5)) < 1e-6,
          f"unit={u.tolist()}")
    g0 = OscillatorNode(nid="g", omega=0.5, coupling=0.1, gaba_amp=0.1,
                        dim=16, phase=0.0).gaba_gate()
    gp = OscillatorNode(nid="g", omega=0.5, coupling=0.1, gaba_amp=0.1,
                        dim=16, phase=math.pi).gaba_gate()
    check("A3. GABA 半周期窗口（cos>0 激活，π 相位关闭）",
          abs(g0 - 1.0) < 1e-6 and gp < 1e-6,
          f"g0={g0:.3f} gp={gp:.3f}")

    # ── B. 装配注入 ──
    print("\n[B] 装配注入（o 型振荡节点）...", flush=True)
    oscs = getattr(cortex.ensemble, "oscillators", [])
    check("B1. ensemble 装配注入振荡节点（theta+gamma 双层）",
          len(oscs) == 2
          and any(o.nid.startswith("osc_theta") for o in oscs)
          and any(o.nid.startswith("osc_gamma") for o in oscs),
          f"oscs={[o.nid for o in oscs]}")
    check("B2. 振荡节点含 GABA 门控方向（gaba_vec 维度 = 场维度）",
          all(o.gaba_vec.numel() == int(getattr(cortex.field, "dim", 0))
              for o in oscs),
          f"dims={[int(o.gaba_vec.numel()) for o in oscs]}")

    # ── C. 相位牵引（PhasorDynamics evolve 外部振荡器）──
    print("\n[C] 相位牵引（o 型驱动 p 型锁相）...", flush=True)
    try:
        pd = PhasorDynamics()
        pd.register_neurons(["n0", "n1"], phases=[0.1, 0.5])
        # dt=0 且 K=0 → 无内部演化，仅外部牵引项可见
        base = pd.evolve(active_ids=["n0", "n1"], dt=0.0, coupling_strength=0.0)
        pulled = pd.evolve(
            active_ids=["n0", "n1"], dt=0.0, coupling_strength=0.0,
            external_phases=[[1.0, 0.0]],  # osc 相位 0（θ=0）
            external_weights=[0.5],
        )
        diff = float((pulled - base).abs().sum().item())
        check("C1. 外部振荡器牵引改变相位演化", diff > 1e-4,
              f"diff={diff:.5f}")
        # 方向验证：n0 相位 0.1 > osc 0 → sin(0-0.1)<0 → 相位应逆时针减小
        import math as _m
        p0_base = _m.atan2(float(base[0, 1].detach()), float(base[0, 0].detach()))
        p0_pull = _m.atan2(float(pulled[0, 1].detach()), float(pulled[0, 0].detach()))
        delta = (p0_pull - p0_base + _m.pi) % (2 * _m.pi) - _m.pi
        check("C2. 牵引方向正确（朝振荡器相位拉动）", delta < 0,
              f"delta={delta:.4f} (base={p0_base:.3f}→pull={p0_pull:.3f})")
    except Exception as e:
        check("C1. 外部振荡器牵引改变相位演化", False, f"err={e}")
        check("C2. 牵引方向正确", False, f"err={e}")

    # ── D. GABA 门控（场节奏窗口）──
    print("\n[D] GABA 式节奏门控 ...", flush=True)
    try:
        emb_in = torch.tensor(
            [cortex._general_sp.encode(build_dialogue_prompt("节奏门控测试"))],
            dtype=torch.long, device=cortex.device)
        emb = cortex._shared_embedding(emb_in)
        zh_domain_all = ["zh_aug0_dialogue", "zh_aug1_dialogue",
                         "zh_aug2_dialogue", "zh_aug3_dialogue",
                         "zh_std0_dialogue", "zh"]
        # 注意：continuous_forward 写 thread-local task_field（多线程推理隔离），
        # 须经 _get_task_field 读取同一对象（与 continuous_forward 内部一致）。
        tf = cortex.ensemble._get_task_field()
        tf.reset(batch_size=1)
        mask0 = tf.inhibitory_mask.detach().clone()
        cortex.ensemble.continuous_forward(
            shared_embeddings=emb, active_nids=zh_domain_all,
        )
        mask1 = tf.inhibitory_mask.detach()
        gated = bool((mask1 < mask0 - 1e-6).any())
        check("D1. 振荡门控调制 inhibitory_mask（GABA 窗口）",
              gated, f"min_mask={float(mask1.min()):.4f}")
        # 门控幅度受 gaba_amp 限制（不污染内容场：轻微调制）
        check("D2. 门控幅度温和（≥0.9，防内容场过衰减）",
              float(mask1.min()) >= 0.9 - 1e-6,
              f"min_mask={float(mask1.min()):.4f}")
    except Exception as e:
        check("D1. 振荡门控调制 inhibitory_mask", False, f"err={e}")
        check("D2. 门控幅度温和", False, f"err={e}")

    # ── E. phase_code 含振荡段（节奏中心）──
    print("\n[E] KoPE phase_code 纳入振荡段 ...", flush=True)
    try:
        res = cortex.think(emb, active_nids=zh_domain_all,
                           collab_mode="continuous", fusion_mode="soft")
        pc = res.get("phase_code")
        M = len(cortex.ensemble.oscillators)
        expect = 2 * len(zh_domain_all) + 2 * M
        check("E1. phase_code 维度含振荡段（2N+2M）",
              pc is not None and pc.numel() == expect,
              f"dim={None if pc is None else tuple(pc.shape)} expect={expect}")
        # 振荡段数值 = 振荡节点相位
        osc_seg = pc[-2 * M:] if pc is not None else None
        osc_mean_ph = (float(osc_seg[1]) if M == 1 else
                       float(osc_seg[-1]) if M >= 1 else 0.0)
        check("E2. 振荡段为节奏中心（有限值）",
              osc_seg is not None and all(v == v for v in osc_seg.tolist()),
              f"osc_seg={None if osc_seg is None else osc_seg.tolist()[:4]}")
    except Exception as e:
        check("E1. phase_code 维度含振荡段", False, f"err={e}")
        check("E2. 振荡段为节奏中心", False, f"err={e}")

    # ── F. 生产零破坏 ──
    print("\n[F] 生产零破坏（生成 / get_last_phase）...", flush=True)
    try:
        out = cortex.generate(
            build_dialogue_prompt("介绍一下什么是注意力机制。"),
            max_tokens=32, domain="zh", temperature=0.55,
        )
        lp = cortex.get_last_phase()
        check("F1. 生成非空不退化", isinstance(out, str)
              and len(out.strip()) > 0 and not cortex._is_degenerate_text(out),
              f"out={out[:30]!r}")
        check("F2. get_last_phase 正常（含节奏中心相位均值）",
              isinstance(lp, float), f"phase={lp}")
    except Exception as e:
        check("F1. 生成非空不退化", False, f"err={e}")
        check("F2. get_last_phase 正常", False, f"err={e}")

    print(f"\n  总耗时: {time.time() - t0:.1f}s", flush=True)
    print("=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
