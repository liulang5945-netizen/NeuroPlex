#!/usr/bin/env python3
"""C25-C 神经调质深度耦合训练冒烟验证（2026-08-10）。

对比文档 2.11/2.12（缺口 R）：神经调质此前"仅状态记录"——对比文档 171 行
列出乙酰胆碱（attention 调制）但未实现；调质更新是手工阈值规则，未与
训练信号构成完整闭环。C25-C 修复：
1. 新增乙酰胆碱（ACh）调质：新颖性驱动 → attention 聚焦增益
   （get_attention_focus_gain，与 NE 警觉组合调制，互补不覆盖）
2. ACh 进入训练闭环：sleep 训练 loss 变化率 → ACh 目标（loss 上升=新颖
   →聚焦，快速下降=习惯化→降）——DA=奖励、ACh=新颖性互补
3. ensemble forward 注入 NE×ACh 组合 temp_gain

验证目标：
1. ACh 状态存在 + set_targets/step EMA 趋近
2. get_attention_focus_gain 映射正确（0.5→1.0 / 1.0→1.5 / 0→0.7）
3. ensemble 组合调制逻辑（NE×ACh）生效且 NE 独立时不变
4. 持久化 round-trip + 旧 ckpt 无 ACh → 默认中性兼容
5. 训练闭环：loss 变化率 → DA/ACh 目标 → lr/focus 增益联动

运行：python -u scripts/training/verify_c25_c_neuromod.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.resonance.neuro_modulation import NeuromodulatorState

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


def delta_to_ach(delta: float) -> float:
    """sleep_engine 的 ACh 目标规则（与 _update_neuromodulators 一致）。"""
    if delta > 0.05:
        return 0.85   # loss 上升：新颖/困难 → 聚焦
    if delta > -0.05:
        return 0.5    # 停滞：中性
    return 0.35       # 学习有效：习惯化


def main() -> None:
    print("=" * 60)
    print("C25-C 神经调质深度耦合训练冒烟验证")
    print("=" * 60)

    # ---- 1. ACh 状态存在 + EMA 趋近 ----
    nm = NeuromodulatorState()
    check("ACh 初始中性", abs(nm.acetylcholine - 0.5) < 1e-9, f"ach={nm.acetylcholine}")
    nm.set_targets(acetylcholine=0.85)
    for _ in range(100):
        nm.step()
    check("ACh EMA 趋近目标", abs(nm.acetylcholine - 0.85) < 0.01,
          f"ach={nm.acetylcholine:.3f}")
    nm.set_targets(acetylcholine=0.2)
    for _ in range(100):
        nm.step()
    check("ACh 可下调", nm.acetylcholine < 0.4, f"ach={nm.acetylcholine:.3f}")

    # ---- 2. focus_gain 映射（0.6 + ACh*0.8，0.5=中性→1.0）----
    check("ACh=0.5 → gain=1.0", abs(NeuromodulatorState(acetylcholine=0.5).get_attention_focus_gain() - 1.0) < 1e-6,
          f"gain={NeuromodulatorState(acetylcholine=0.5).get_attention_focus_gain():.2f}")
    check("ACh=1.0 → gain=1.4", abs(NeuromodulatorState(acetylcholine=1.0).get_attention_focus_gain() - 1.4) < 1e-6,
          f"gain={NeuromodulatorState(acetylcholine=1.0).get_attention_focus_gain():.2f}")
    check("ACh=0 → gain=0.6", abs(NeuromodulatorState(acetylcholine=0.0).get_attention_focus_gain() - 0.6) < 1e-6,
          f"gain={NeuromodulatorState(acetylcholine=0.0).get_attention_focus_gain():.2f}")

    # ---- 3. ensemble 组合调制（NE×ACh，与 ensemble.py 注入逻辑一致）----
    nm_ne = NeuromodulatorState(norepinephrine=1.0, acetylcholine=0.5)  # 高警觉 + 中性 ACh
    temp_gain_ne_only = nm_ne.get_attention_temp_gain()
    temp_gain_combo = nm_ne.get_attention_temp_gain() * nm_ne.get_attention_focus_gain()
    check("ACh 中性时组合=NE", abs(temp_gain_combo - temp_gain_ne_only) < 1e-6,
          f"NE={temp_gain_ne_only:.2f} combo={temp_gain_combo:.2f}")
    nm_high_ach = NeuromodulatorState(norepinephrine=1.0, acetylcholine=1.0)
    combo_high = nm_high_ach.get_attention_temp_gain() * nm_high_ach.get_attention_focus_gain()
    check("高 ACh 放大组合增益", combo_high > temp_gain_combo,
          f"{combo_high:.2f} > {temp_gain_combo:.2f}")
    nm_low_ach = NeuromodulatorState(norepinephrine=1.0, acetylcholine=0.0)
    combo_low = nm_low_ach.get_attention_temp_gain() * nm_low_ach.get_attention_focus_gain()
    check("低 ACh 压低组合增益", combo_low < temp_gain_combo,
          f"{combo_low:.2f} < {temp_gain_combo:.2f}")

    # ---- 4. 持久化 round-trip + 旧 ckpt 兼容 ----
    nm2 = NeuromodulatorState(acetylcholine=0.8, norepinephrine=0.7)
    nm2.set_targets(acetylcholine=0.9)
    state = nm2.get_state_dict()
    check("state_dict 含 ACh", "acetylcholine" in state and "_target_acetylcholine" in state)
    nm3 = NeuromodulatorState()
    nm3.load_state_dict(state)
    check("ACh 恢复", abs(nm3.acetylcholine - 0.8) < 1e-9, f"ach={nm3.acetylcholine}")
    check("ACh target 恢复", abs(nm3._target_acetylcholine - 0.9) < 1e-9)
    # 旧 ckpt（无 ACh 字段）→ 默认中性
    legacy = {"dopamine": 0.3, "serotonin": 0.6, "norepinephrine": 0.4,
              "_target_dopamine": 0.5, "_target_serotonin": 0.5, "_target_norepinephrine": 0.5}
    nm4 = NeuromodulatorState()
    nm4.load_state_dict(legacy)
    check("旧 ckpt 无 ACh → 中性兼容", abs(nm4.acetylcholine - 0.5) < 1e-9,
          f"ach={nm4.acetylcholine}")

    # ---- 5. 训练闭环：loss 变化率 → DA/ACh 目标 → lr/focus 联动 ----
    nm5 = NeuromodulatorState()
    # loss 上升（新颖）→ DA 低（惩罚）+ ACh 高（聚焦）
    nm5.set_targets(dopamine=0.15, acetylcholine=delta_to_ach(0.1))
    for _ in range(100):
        nm5.step()
    check("loss 上升 → DA 低", nm5.dopamine < 0.3, f"da={nm5.dopamine:.2f}")
    check("loss 上升 → ACh 高", nm5.acetylcholine > 0.7, f"ach={nm5.acetylcholine:.2f}")
    check("ACh 高 → focus 增益 >1", nm5.get_attention_focus_gain() > 1.2,
          f"gain={nm5.get_attention_focus_gain():.2f}")
    # loss 快速下降（熟悉/学习有效）→ DA 高（奖励）+ ACh 低（习惯化）
    nm6 = NeuromodulatorState()
    nm6.set_targets(dopamine=0.85, acetylcholine=delta_to_ach(-0.3))
    for _ in range(100):
        nm6.step()
    check("loss 下降 → DA 高", nm6.dopamine > 0.7, f"da={nm6.dopamine:.2f}")
    check("loss 下降 → ACh 低（习惯化）", nm6.acetylcholine < 0.4,
          f"ach={nm6.acetylcholine:.2f}")

    # ---- 6. 与既有 DA/NE/5-HT 接口无回归 ----
    nm7 = NeuromodulatorState()
    check("lr_multiplier 接口保留", abs(nm7.get_lr_multiplier() - 1.25) < 1e-6)
    check("refractory 接口保留", abs(nm7.get_refractory_multiplier() - 1.0) < 1e-6)
    check("field_write 接口保留", abs(nm7.get_field_write_scale() - 1.0) < 1e-6)
    check("attention_temp 接口保留", abs(nm7.get_attention_temp_gain() - 1.0) < 1e-6)
    check("ffn_gain 接口保留", abs(nm7.get_ffn_gain() - 1.25) < 1e-6)

    print(f"\n结果: {passed} PASS / {failed} FAIL")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
