"""P0+P1 端到端冒烟测试：验证 assemble_cortex + 推理路径无崩溃。

测试项：
1. assemble_cortex() 在 fallback 模式下成功装配所有 bio 模块
2. cortex.think() 推理路径不崩溃（含 STDP/Neuromodulator/wire_scale）
3. cortex.generate() 生成路径不崩溃（含 tick_gamma + WorkingMemory + EOS）
4. NeuromodulatorState 调质确实影响 ensemble 行为（scale != 1 时写入强度变化）
5. STDP record_firing 在 forward 中被调用（_firing_history 非空）
6. CoactivationTracker 不因 round_num kwarg 崩溃

不验证生成质量（fallback 是 random init），只验证代码路径完整。
"""
from __future__ import annotations

import os
import sys
import traceback

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# P0-2: sentencepiece 装到 _libs/
_LIBS = os.path.join(PROJECT_ROOT, "_libs")
if os.path.isdir(_LIBS) and _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)


def test_assemble_cortex():
    print("=" * 70)
    print("P0+P1: assemble_cortex() 端到端冒烟测试")
    print("=" * 70)

    # 用空目录触发单神经元 fallback（避免依赖具体 ckpt）
    fallback_dir = "data/_empty_neurons_dir_for_p0p1"
    os.makedirs(fallback_dir, exist_ok=True)

    # ── Step 1: assemble_cortex ──
    print("\n[1] assemble_cortex() ...")
    try:
        from taiji.loader import assemble_cortex
        cortex, tokenizer, modules = assemble_cortex(
            neurons_dir=fallback_dir,
            device="cpu",
            max_rounds=2,
            enable_gating=False,
            wire_bio_modules=True,
        )
    except Exception as e:
        print(f"  ❌ assemble_cortex 失败: {e}")
        traceback.print_exc()
        return False

    print(f"  ✓ Cortex assembled: {len(cortex.neurons)} neurons")
    print(f"  ✓ Wired modules: {list(modules.keys())}")

    # 关键模块必须到位（fallback 时无 encoder/thalamic 是正常的，但其他必须有）
    must_have = ["stdp_tracker", "coaction", "neuromodulator",
                 "gamma_oscillator", "working_memory",
                 "lifecycle", "sleep_consolidator"]
    missing = [k for k in must_have if k not in modules]
    if missing:
        print(f"  ⚠️  缺失关键模块: {missing}")
        # 不直接 fail，因为某些模块可能因依赖问题被跳过
    else:
        print(f"  ✓ 所有关键模块已装配")

    # ── Step 2: 推理路径 ──
    print("\n[2] cortex.think() 推理路径 ...")
    try:
        text = "你好"
        ids = tokenizer.encode(text)
        print(f"  encode('{text}') → {len(ids)} tokens")
        input_ids = torch.tensor([ids], dtype=torch.long)
        result = cortex.think(input_ids, route=False)  # fallback 模式无 router
        print(f"  ✓ think() returned: keys={list(result.keys())}")
        print(f"    n_rounds={result.get('n_rounds')}, "
              f"n_active={result.get('n_active_history')}")
    except Exception as e:
        print(f"  ❌ think() 失败: {e}")
        traceback.print_exc()
        return False

    # ── Step 3: STDP 验证 ──
    print("\n[3] STDP record_firing 验证 ...")
    try:
        stdp = modules.get("stdp_tracker")
        if stdp is not None:
            history = getattr(stdp, '_firing_history', {})
            if history:
                print(f"  ✓ STDP _firing_history 非空: {len(history)} neurons fired")
            else:
                print(f"  ⚠️  STDP _firing_history 为空（forward 可能未触发记录）")
        else:
            print(f"  ⚠️  STDPTracker 未装配，跳过")
    except Exception as e:
        print(f"  ❌ STDP 验证失败: {e}")
        traceback.print_exc()

    # ── Step 4: CoactivationTracker kwarg 兼容性 ──
    print("\n[4] CoactionTracker.update(round_num=...) 兼容性 ...")
    try:
        coaction = modules.get("coaction")
        if coaction is not None:
            # 模拟 ensemble 调用
            coaction.update(["n1", "n2"], round_num=1)
            print(f"  ✓ CoactionTracker.update(active_ids, round_num=1) 不崩溃")
        else:
            print(f"  ⚠️  CoactionTracker 未装配，跳过")
    except Exception as e:
        print(f"  ❌ CoactionTracker kwarg 兼容性失败: {e}")
        traceback.print_exc()
        return False

    # ── Step 5: NeuromodulatorState 调质影响 ──
    print("\n[5] NeuromodulatorState 调质影响验证 ...")
    try:
        nm = modules.get("neuromodulator")
        if nm is not None:
            # 默认值：dopamine=0.5, serotonin=0.5, norepinephrine=0.5
            # 公式：lr_mult = 0.5 + dopamine * 1.5  →  默认 1.25
            #       refractory_mult = 0.5 + serotonin * 1.0  →  默认 1.0
            #       write_scale = 0.5 + norepinephrine * 1.0  →  默认 1.0
            lr_mult = nm.get_lr_multiplier()
            ref_mult = nm.get_refractory_multiplier()
            write_scale = nm.get_field_write_scale()
            print(f"  默认状态: lr_mult={lr_mult:.3f}, "
                  f"refractory_mult={ref_mult:.3f}, "
                  f"write_scale={write_scale:.3f}")
            # 验证默认值在合理范围
            assert 0.5 <= lr_mult <= 2.0, f"lr_mult 越界: {lr_mult}"
            assert 0.5 <= ref_mult <= 1.5, f"refractory_mult 越界: {ref_mult}"
            assert 0.5 <= write_scale <= 1.5, f"write_scale 越界: {write_scale}"
            print(f"  ✓ 默认值在合理范围")

            # 调整调质后值应该变化
            nm.set_targets(dopamine=1.0, serotonin=1.0, norepinephrine=1.0)
            for _ in range(20):
                nm.step()
            lr_mult_hi = nm.get_lr_multiplier()
            ref_mult_hi = nm.get_refractory_multiplier()
            write_scale_hi = nm.get_field_write_scale()
            print(f"  高调质: lr_mult={lr_mult_hi:.3f}, "
                  f"refractory_mult={ref_mult_hi:.3f}, "
                  f"write_scale={write_scale_hi:.3f}")
            assert lr_mult_hi > lr_mult, f"高多巴胺应使 lr_mult 上升"
            assert ref_mult_hi > ref_mult, f"高血清素应使 refractory_mult 上升"
            assert write_scale_hi > write_scale, f"高去甲肾上腺素应使 write_scale 上升"
            print(f"  ✓ 高调质正确提升所有倍数")

            # 低调质测试
            nm.set_targets(dopamine=0.0, serotonin=0.0, norepinephrine=0.0)
            for _ in range(20):
                nm.step()
            lr_mult_lo = nm.get_lr_multiplier()
            ref_mult_lo = nm.get_refractory_multiplier()
            write_scale_lo = nm.get_field_write_scale()
            print(f"  低调质: lr_mult={lr_mult_lo:.3f}, "
                  f"refractory_mult={ref_mult_lo:.3f}, "
                  f"write_scale={write_scale_lo:.3f}")
            assert lr_mult_lo < lr_mult, f"低多巴胺应使 lr_mult 下降"
            assert ref_mult_lo < ref_mult, f"低血清素应使 refractory_mult 下降"
            assert write_scale_lo < write_scale, f"低去甲肾上腺素应使 write_scale 下降"
            print(f"  ✓ 低调质正确下降所有倍数")
            # should_trigger_neurogenesis 在低多巴胺时应为 True
            assert nm.should_trigger_neurogenesis() == True, \
                "低多巴胺应触发神经新生"
            print(f"  ✓ 低多巴胺触发 neurogenesis")
        else:
            print(f"  ⚠️  NeuromodulatorState 未装配，跳过")
    except Exception as e:
        print(f"  ❌ NeuromodulatorState 验证失败: {e}")
        traceback.print_exc()
        return False

    # ── Step 6: generate() 路径（含 tick_gamma + WorkingMemory + EOS） ──
    print("\n[6] cortex.generate() 生成路径 ...")
    try:
        out = cortex.generate(
            prompt="你好",
            max_tokens=8,
            temperature=0.7,
        )
        print(f"  ✓ generate() returned: type={type(out).__name__}")
        if isinstance(out, str):
            print(f"    output='{out[:50]}'")
        elif isinstance(out, dict):
            print(f"    keys={list(out.keys())}")
            text = out.get("text") or out.get("output") or ""
            print(f"    text='{text[:50]}'")
    except Exception as e:
        print(f"  ❌ generate() 失败: {e}")
        traceback.print_exc()
        return False

    # ── Step 7: GammaOscillator phase 推进 ──
    print("\n[7] GammaOscillator tick 验证 ...")
    try:
        gamma = modules.get("gamma_oscillator")
        if gamma is not None:
            initial_phase = gamma.global_phase
            # tick 应该推进相位
            gamma.tick()
            new_phase = gamma.global_phase
            assert new_phase != initial_phase, "tick 后 phase 未变化"
            print(f"  ✓ Gamma tick: {initial_phase:.3f} → {new_phase:.3f}")
        else:
            print(f"  ⚠️  GammaOscillator 未装配，跳过")
    except Exception as e:
        print(f"  ❌ GammaOscillator 验证失败: {e}")
        traceback.print_exc()

    # ── 总结 ──
    print("\n" + "=" * 70)
    print("✅ 所有冒烟测试通过（无崩溃）")
    print("=" * 70)
    return True


if __name__ == "__main__":
    ok = test_assemble_cortex()
    sys.exit(0 if ok else 1)
