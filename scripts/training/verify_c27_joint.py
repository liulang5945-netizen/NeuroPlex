#!/usr/bin/env python3
"""C27 三增量联合端到端验证（2026-08-14）。

背景：C27 三个增量独立验证全绿（增量一 SMCS 实例级路由 14/14、增量二 KoPE
相位编码 13/13、增量三 BioOSS p/o 双模型 14/14）。本脚本在**真实长生成**
中同时开启三者，验证协同不退化：
- 增量一（SMCS）：instance_routing=True（生产默认）chunk 边界演化
- 增量二（KoPE）：phase_code 含振荡段（维度 2N+2M）、phase_mean/phase_lock
- 增量三（BioOSS）：o 型振荡节点（相位牵引 + GABA 门控）在生成中推进

验证层次：
A. 三机制联合装配（振荡节点 / gamma 振荡器 / 实例路由默认开）
B. KoPE 联合合法性（真实 continuous 长生成途中 phase_code 维度含振荡段、
   phase_mean 有限、phase_lock ∈ [0,1]）
C. 记忆注入 × 振荡协同（带 phase 记忆 3 元组按记忆相位对齐 theta；
   记忆写入场成功；GABA 门控温和不抑制记忆写路径）
D. 实例路由 × 振荡协同（真实长生成 chunk 边界触发演化 ≥1；输出不退化；
   振荡节点相位在生成期间实际推进）
E. 三机制同时开启的多轮生成稳定性（3 次长生成非空不退化 + 相位编码合法）

运行：python -u scripts/training/verify_c27_joint.py
"""

from __future__ import annotations

import inspect
import os
import sys
import time
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import random  # noqa: E402
import numpy as np  # noqa: E402

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
from taiji.loader import assemble_cortex  # noqa: E402
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
    print("C27 三增量联合端到端验证（SMCS × KoPE × BioOSS）", flush=True)
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

    # judge EMA 预热（executive 判定主信号，与增量一脚本同口径）
    warm = ["你好", "帮我写代码", "解一道数学题", "What is this?", "写一首诗"]
    for _ in range(30):
        for wp in warm:
            cortex._executive_route(wp)
    print("  judge EMA 预热完成", flush=True)

    zh_domain_all = ["zh_aug0_dialogue", "zh_aug1_dialogue",
                     "zh_aug2_dialogue", "zh_aug3_dialogue",
                     "zh_std0_dialogue", "zh"]
    field_dim = int(cortex.field.dim)

    # ── A. 三机制联合装配 ──
    print("\n[A] 三机制联合装配 ...", flush=True)
    oscs = getattr(cortex.ensemble, "oscillators", [])
    check("A1. BioOSS o 型振荡节点已注入（theta+gamma 双层）",
          len(oscs) == 2
          and any(o.nid.startswith("osc_theta") for o in oscs)
          and any(o.nid.startswith("osc_gamma") for o in oscs),
          f"oscs={[o.nid for o in oscs]}")
    _sig = inspect.signature(cortex.generate)
    _ir_default = (_sig.parameters.get("instance_routing").default
                   if "instance_routing" in _sig.parameters else None)
    check("A2. KoPE 载体 gamma 振荡器 + SMCS 实例路由默认开",
          cortex.ensemble.gamma_oscillator is not None
          and _ir_default is True,
          f"gamma={cortex.ensemble.gamma_oscillator is not None} "
          f"ir_default={_ir_default}")
    check("A3. 振荡节点门控方向维度 = 场维度（编码载体统一）",
          all(int(o.gaba_vec.numel()) == field_dim for o in oscs),
          f"field_dim={field_dim}")

    # ── B. KoPE 联合合法性（真实 continuous 长生成途中）──
    print("\n[B] KoPE 相位编码联合合法性 ...", flush=True)
    try:
        emb_in = torch.tensor(
            [cortex._general_sp.encode(build_dialogue_prompt("什么是机器学习？"))],
            dtype=torch.long, device=cortex.device)
        emb = cortex._shared_embedding(emb_in)
        res = cortex.think(emb, active_nids=zh_domain_all,
                           collab_mode="continuous", fusion_mode="soft")
        pc = res.get("phase_code")
        N = len(zh_domain_all)
        M = len(cortex.ensemble.oscillators)
        expect = 2 * N + 2 * M
        check("B1. phase_code 维度含振荡段（2N+2M，维度不退化）",
              pc is not None and pc.numel() == expect,
              f"dim={None if pc is None else tuple(pc.shape)} expect={expect}")
        pm = res.get("phase_mean")
        check("B2. phase_mean 有限值（记忆对齐目标合法）",
              isinstance(pm, float) and pm == pm and abs(pm) <= 2 * 3.1416,
              f"phase_mean={pm}")
        pl = res.get("phase_lock")
        check("B3. phase_lock 锁相度 ∈ [0,1]",
              isinstance(pl, float) and 0.0 <= pl <= 1.0,
              f"phase_lock={pl}")
    except Exception as e:
        check("B1. phase_code 维度含振荡段", False, f"err={e}")
        check("B2. phase_mean 有限值", False, f"err={e}")
        check("B3. phase_lock ∈ [0,1]", False, f"err={e}")

    # ── C. 记忆注入 × 振荡协同（KoPE 相位归属 × BioOSS 节奏场）──
    print("\n[C] 记忆注入 × 振荡协同（GABA 门控不抑制记忆写路径）...", flush=True)
    try:
        mem_vec = torch.randn(field_dim)
        mem_vec = mem_vec / mem_vec.norm()
        seen_targets: list = []
        orig_entrain = ContinuousResonance.entrain_memory

        def _spy_entrain(self, target_phase: float = 0.0):
            seen_targets.append(float(target_phase))
            return orig_entrain(self, target_phase)

        with mock.patch.object(ContinuousResonance, "entrain_memory",
                               _spy_entrain):
            res_mem = cortex.think(
                emb, active_nids=zh_domain_all,
                collab_mode="continuous", fusion_mode="soft",
                memory_vectors=[(mem_vec, 0.5, 0.7)],
            )
        check("C1. 记忆带相位 → theta 按记忆相位对齐（entrain 0.7）",
              0.7 in seen_targets, f"seen={seen_targets}")
        tf = cortex.ensemble._get_task_field()
        check("C2. 记忆向量写入共振场（__memory_0__ 贡献存在）",
              "__memory_0__" in (tf._contributions or {}),
              f"keys={list((tf._contributions or {}).keys())[:4]}")
        fs = res_mem.get("field_state")
        fs_ok = (fs is not None and fs.numel() > 0
                 and bool(torch.isfinite(fs).all())
                 and float(fs.detach().norm()) > 1e-6)
        check("C3. 带记忆生成场状态非零有限", fs_ok,
              f"norm={None if fs is None else float(fs.detach().norm()):.3f}")
        mask_min = float(tf.inhibitory_mask.min())
        check("C4. GABA 门控温和（≥0.9），不衰减内容场写路径",
              mask_min >= 0.9 - 1e-6, f"min_mask={mask_min:.4f}")
    except Exception as e:
        check("C1. 记忆相位对齐", False, f"err={e}")
        check("C2. 记忆写入场", False, f"err={e}")
        check("C3. 场状态非零有限", False, f"err={e}")
        check("C4. GABA 门控温和", False, f"err={e}")

    # ── D. 实例路由 × 振荡协同（真实长生成）──
    print("\n[D] 实例路由 × 振荡协同（真实长生成）...", flush=True)
    orig_evolve = cortex._instance_route_evolve
    calls_d = {"n": 0}
    osc_ph_before = [float(o.phase) for o in cortex.ensemble.oscillators]

    def spy_d(*a, **k):
        calls_d["n"] += 1
        return orig_evolve(*a, **k)

    try:
        with mock.patch.object(cortex, "_instance_route_evolve", spy_d):
            out_d = cortex.generate(
                build_dialogue_prompt("写一篇关于夏天的小短文，描述天气和人们的活动。"),
                max_tokens=48, domain="zh", temperature=0.55,
            )
        osc_ph_after = [float(o.phase) for o in cortex.ensemble.oscillators]
        check("D1. 长生成输出非空不退化", isinstance(out_d, str)
              and len(out_d.strip()) > 0 and not cortex._is_degenerate_text(out_d),
              f"out={out_d[:36]!r}")
        check("D2. 实例级路由在 chunk 边界触发（SMCS × BioOSS 共存）",
              calls_d["n"] >= 1, f"evolve_calls={calls_d['n']}")
        advanced = [abs(a - b) > 1e-6 for a, b in zip(osc_ph_after, osc_ph_before)]
        check("D3. 振荡节点相位在生成期间实际推进",
              len(advanced) == len(osc_ph_before) and all(advanced),
              f"before={[round(x, 3) for x in osc_ph_before]} "
              f"after={[round(x, 3) for x in osc_ph_after]}")
        check("D4. 生成后 phase_mean 正常（节奏中心进生成状态）",
              isinstance(cortex.get_last_phase(), float),
              f"last_phase={cortex.get_last_phase()}")
    except Exception as e:
        check("D1. 长生成输出非空不退化", False, f"err={e}")
        check("D2. 实例级路由触发", False, f"err={e}")
        check("D3. 振荡推进", False, f"err={e}")
        check("D4. phase_mean 正常", False, f"err={e}")

    # ── E. 三机制同时开启的多轮生成稳定性 ──
    print("\n[E] 三机制同时开启的多轮生成稳定性 ...", flush=True)
    prompts = [
        "介绍一下什么是神经网络。",
        "怎么学好一门编程语言？",
        "写一段关于秋天的文字。",
    ]
    e_ok = True
    for i, p in enumerate(prompts):
        try:
            out = cortex.generate(
                build_dialogue_prompt(p), max_tokens=40,
                domain="zh", temperature=0.55,
            )
            lp = cortex.get_last_phase()
            ok = (isinstance(out, str) and len(out.strip()) > 0
                  and not cortex._is_degenerate_text(out)
                  and isinstance(lp, float) and lp == lp)
            if not ok:
                e_ok = False
                print(f"    [E{i+1}] FAIL out={out[:36]!r} phase={lp}", flush=True)
            else:
                print(f"    [E{i+1}] ok out={out[:24]!r}", flush=True)
        except Exception as e:
            e_ok = False
            print(f"    [E{i+1}] FAIL err={e}", flush=True)
    check("E1-E3. 三轮长生成全部非空不退化且相位编码有限", e_ok)

    print(f"\n  总耗时: {time.time() - t0:.1f}s", flush=True)
    print("=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
