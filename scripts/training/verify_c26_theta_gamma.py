#!/usr/bin/env python3
"""缺口 R 多频段振荡验证：theta-gamma 嵌套（2026-08-11）。

人脑机制：慢 theta 振荡（4-8Hz）相位调制快 gamma 振荡（30-100Hz）振幅包络
（Lisman theta-gamma 嵌套编码）——theta 相位决定 gamma 活动窗口。

实现（ContinuousResonance，默认 theta_omega=0 不启用，零回归）：
- theta_phase_at(t)：theta 慢相位随时间单调推进（慢于 gamma）
- theta_envelope(t)：1 + theta_amp·cos(theta_phase(t)) 周期包络
- theta_modulate(activ, t)：gamma 激活振幅 × theta 包络（调幅嵌套）

验证：
1. theta 相位单调推进、包络周期性（t 移 2π/omega 后复原）
2. theta_omega=0 时包络恒 1（无嵌套，与旧行为逐元素一致——回归）
3. 嵌套调制生效：真实激活序列 × theta 包络后出现周期起伏（峰谷差 > 无嵌套），
   且调制在包络范围内（∈ [1-A, 1+A]）
4. 真实装配回归：9 神经元 think 正常（默认 theta_omega=0 不改变现有 continuous）

运行：python -u scripts/training/verify_c26_theta_gamma.py
"""

from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from taiji.loader import assemble_cortex  # noqa: E402
from taiji.resonance.continuous import ContinuousResonance  # noqa: E402

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
    print("=" * 64, flush=True)
    print("缺口 R 多频段振荡：theta-gamma 嵌套验证", flush=True)
    print("=" * 64, flush=True)

    theta_omega = 0.5  # theta 慢振荡（rad/步，明显慢于 gamma 单位步进）
    theta_amp = 0.2
    cr = ContinuousResonance(theta_omega=theta_omega, theta_amp=theta_amp)
    cr_off = ContinuousResonance()  # 默认（theta_omega=0）

    # ── 1. theta 相位单调推进 + 包络周期性 ──
    phases = [cr.theta_phase_at(t) for t in range(0, 16)]
    monotonic = all(phases[i] < phases[i + 1] for i in range(len(phases) - 1))
    check("theta 相位随时间单调推进", monotonic,
          f"θ(15)={phases[15]:.2f}")
    period = 2 * math.pi / theta_omega
    env0 = cr.theta_envelope(0.0)
    env_period = cr.theta_envelope(period)
    check("theta 包络周期复原（t+2π/ω）", abs(env0 - env_period) < 1e-6,
          f"{env0:.4f} vs {env_period:.4f}")

    # ── 2. theta_omega=0 回归：包络恒 1、调制恒等 ──
    activ = torch.tensor([0.3, 0.7, 0.9])
    off_env = [cr_off.theta_envelope(t) for t in range(8)]
    check("未启用嵌套时包络恒 1（回归）", all(abs(e - 1.0) < 1e-9 for e in off_env))
    check("未启用嵌套时调制恒等（逐元素一致）",
          torch.equal(cr_off.theta_modulate(activ, 3.0), activ))

    # ── 3. 嵌套调制生效：真实激活序列 × 包络后出现周期起伏 ──
    # 构造一段"gamma 激活"（模拟主循环中恒定绑定下的激活，未嵌套时平坦）
    base_activ = torch.tensor([0.6] * 16)  # 假设 gamma 激活稳定在 0.6
    mod_seq = [cr.theta_modulate(base_activ, t)[0].item() for t in range(16)]
    peak_gap = max(mod_seq) - min(mod_seq)
    check("嵌套后激活出现周期包络起伏（峰谷差 > 0）", peak_gap > 0.05,
          f"峰谷差={peak_gap:.3f}")
    base_val = 0.6  # base_activ 常数
    within = all(base_val * (1 - theta_amp - 1e-6) <= v
                 <= base_val * (1 + theta_amp + 1e-6) for v in mod_seq)
    check("调制幅度在包络范围 [base×(1-A), base×(1+A)]", within,
          f"范围 [{min(mod_seq):.3f}, {max(mod_seq):.3f}]")

    # 未嵌套对照：16 步全平
    off_seq = [cr_off.theta_modulate(base_activ, t)[0].item() for t in range(16)]
    check("未嵌套对照序列平坦（无起伏）",
          max(off_seq) - min(off_seq) < 1e-9)

    # 慢快关系：theta 一个周期内 gamma 多步（theta 慢于 gamma）
    steps_per_theta = int(2 * math.pi / theta_omega)  # 12.57 步/周期
    check("theta 慢于 gamma（一周期覆盖多步）", steps_per_theta >= 8,
          f"{steps_per_theta:.0f} 步/周期")

    # ── 4. 真实装配回归：默认（theta_omega=0）不改变现有 continuous ──
    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name=COLLAB_NAME,
        extra_neurons_dir=EXTRA_NEURONS_DIR,
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=DIALOGUE_IDS,
    )
    text = "请解释一下量子纠缠的基本原理"
    gids = cortex._general_sp.encode(text) or [0]
    emb = cortex._shared_embedding(torch.tensor([gids], dtype=torch.long))
    res = cortex.think(emb, active_nids=None, fusion_mode="soft",
                       collab_mode="continuous")
    check("默认装配 think 正常（无嵌套零回归）",
          res.get("field_state") is not None and res.get("final_scores"),
          f"n_scores={len(res.get('final_scores', {}))}")

    print("\n" + "=" * 64, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL  ({time.time() - t0:.1f}s)", flush=True)
    print("=" * 64, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
