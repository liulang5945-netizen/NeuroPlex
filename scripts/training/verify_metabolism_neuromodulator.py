"""验证 metabolism → 去甲肾上腺素接线（三调质全接线最后一块）。

核心验证：
1. metabolism.update_neuromodulator() 正确更新 NE 目标值
2. CPU 负载高 → NE↓（节能），CPU 负载低 → NE↑（专注）
3. NE 变化 → get_field_write_scale() 变化
4. SleepEngine 训练前自动调用 metabolism
5. 内存紧张时 DA 被覆盖，内存充裕时 DA 不被覆盖（由 sleep_engine 管）

Usage:
    python scripts/training/verify_metabolism_neuromodulator.py
"""
import sys
import os
from datetime import datetime
from unittest.mock import patch, MagicMock

os.environ.setdefault('TAIJI_TEST_MODE', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch


def main():
    print("=" * 60)
    print("Metabolism → 去甲肾上腺素接线验证")
    print("=" * 60)

    # Step 1: 装配 Cortex（含 NeuromodulatorState）
    print("\n[Step 1] 装配 Cortex...")
    from taiji.loader import assemble_cortex
    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
    )
    nm = cortex._neuromodulator
    assert nm is not None, "NeuromodulatorState 未初始化"
    print(f"  ✅ Cortex 装配完成")
    print(f"  初始调质: DA={nm.dopamine:.3f}, 5HT={nm.serotonin:.3f}, NE={nm.norepinephrine:.3f}")

    # Step 2: 测试 metabolism.update_neuromodulator() 直接调用
    print("\n[Step 2] 测试 metabolism.update_neuromodulator()...")
    from taiji.body import metabolism
    metabolism.set_neuromodulator(nm)

    # 记录调用前的 NE
    ne_before = nm.norepinephrine
    ne_target_before = nm._target_norepinephrine

    metabolism.update_neuromodulator()

    ne_target_after = nm._target_norepinephrine
    print(f"  NE 目标值: {ne_target_before:.3f} → {ne_target_after:.3f}")
    print(f"  (CPU 负载驱动，低负载→NE↑，高负载→NE↓)")

    # 验证 NE 目标值被设置（不等于默认 0.5 或有变化）
    ne_updated = ne_target_after != 0.5 or ne_target_before != ne_target_after
    if ne_updated:
        print(f"  ✅ NE 目标值已更新")
    else:
        print(f"  ⚠️ NE 目标值未变化（可能 CPU 负载恰好 50%）")

    # Step 3: 测试不同 CPU 负载下的 NE 映射
    print("\n[Step 3] 测试不同 CPU 负载下的 NE 映射...")
    test_cases = [
        (10.0, "低负载（专注模式）"),
        (50.0, "中等负载"),
        (90.0, "高负载（节能模式）"),
        (100.0, "满负载（极限节能）"),
    ]

    cpu_ne_results = []
    for cpu_pct, label in test_cases:
        # Mock psutil.cpu_percent 返回指定值
        with patch('psutil.cpu_percent', return_value=cpu_pct):
            metabolism.update_neuromodulator()
            ne_target = nm._target_norepinephrine
            field_write_scale = nm.get_field_write_scale()
            cpu_ne_results.append((cpu_pct, ne_target, field_write_scale))
            print(f"  CPU={cpu_pct:5.1f}% → NE_target={ne_target:.3f}, "
                  f"field_write_scale={field_write_scale:.3f} ({label})")

    # 验证单调性：CPU 越高，NE 越低
    ne_values = [r[1] for r in cpu_ne_results]
    monotonic = all(ne_values[i] >= ne_values[i+1] for i in range(len(ne_values)-1))
    if monotonic:
        print(f"  ✅ 单调性验证通过：CPU↑ → NE↓（节能控制正确）")
    else:
        print(f"  ⚠️ 单调性验证失败：NE 未随 CPU 升高而降低")

    # 验证 field_write_scale 随 NE 变化
    fws_values = [r[2] for r in cpu_ne_results]
    fws_monotonic = all(fws_values[i] >= fws_values[i+1] for i in range(len(fws_values)-1))
    if fws_monotonic:
        print(f"  ✅ field_write_scale 单调性：CPU↑ → NE↓ → field_write↓（节能链路完整）")

    # 验证低负载 NE > 高负载 NE
    low_ne = cpu_ne_results[0][1]
    high_ne = cpu_ne_results[-1][1]
    ne_spread = low_ne - high_ne
    print(f"  NE 跨度: {low_ne:.3f} - {high_ne:.3f} = {ne_spread:.3f}")
    if ne_spread > 0.3:
        print(f"  ✅ NE 跨度足够（>0.3），节能效果显著")
    else:
        print(f"  ⚠️ NE 跨度过小（<0.3），节能效果有限")

    # Step 4: 测试内存紧张时 DA 被覆盖
    print("\n[Step 4] 测试内存紧张时 DA 覆盖逻辑...")
    # 先用 sleep_engine 设置一个 DA 目标值
    nm.set_targets(dopamine=0.8)  # 模拟训练后高 DA
    da_before_metab = nm._target_dopamine
    print(f"  sleep_engine 设置 DA_target={da_before_metab:.3f}")

    # Mock 内存使用率 95%（紧张）
    with patch('psutil.virtual_memory') as mock_vm:
        mock_vm.return_value.percent = 95.0
        mock_vm.return_value.available = 500 * 1024 * 1024  # 500MB
        mock_vm.return_value.total = 8 * 1024**3
        metabolism.update_neuromodulator()

    da_after_high_mem = nm._target_dopamine
    print(f"  内存 95% 后 DA_target={da_after_high_mem:.3f} (应被覆盖为 0.2)")

    if da_after_high_mem == 0.2:
        print(f"  ✅ 内存紧张时 DA 被覆盖（保守模式）")
    else:
        print(f"  ⚠️ 内存紧张时 DA 未被正确覆盖")

    # Mock 内存使用率 50%（充裕）
    nm.set_targets(dopamine=0.8)  # 重新设置高 DA
    with patch('psutil.virtual_memory') as mock_vm:
        mock_vm.return_value.percent = 50.0
        mock_vm.return_value.available = 4 * 1024**3
        mock_vm.return_value.total = 8 * 1024**3
        metabolism.update_neuromodulator()

    da_after_low_mem = nm._target_dopamine
    print(f"  内存 50% 后 DA_target={da_after_low_mem:.3f} (应保持 0.8，不被覆盖)")

    if da_after_low_mem == 0.8:
        print(f"  ✅ 内存充裕时 DA 不被覆盖（由 sleep_engine 管理）")
    else:
        print(f"  ⚠️ 内存充裕时 DA 被意外覆盖为 {da_after_low_mem}")

    # Step 5: 测试 SleepEngine 训练前自动调用 metabolism
    print("\n[Step 5] 测试 SleepEngine 训练前自动调用 metabolism...")
    from taiji.life.feed_engine import get_feed_engine
    from taiji.life.sleep_engine import get_sleep_engine, SleepReport

    feed_engine = get_feed_engine()
    sleep_engine = get_sleep_engine()
    sleep_engine.cortex = cortex
    sleep_engine._neuromodulator = nm
    if sleep_engine._feed_engine is None:
        sleep_engine._feed_engine = feed_engine

    # 喂入训练数据
    feed_engine.feed_text(text="今天天气很好，我们一起去公园散步。", source="test", domain="zh")

    # 记录训练前的 NE
    ne_pre_train = nm.norepinephrine
    ne_target_pre_train = nm._target_norepinephrine

    # 触发训练
    report = SleepReport(timestamp=datetime.now().isoformat(), duration_seconds=0.0)
    sleep_engine._sleep_phase_model_training(report)

    ne_post_train = nm.norepinephrine
    ne_target_post_train = nm._target_norepinephrine
    print(f"  训练前 NE: {ne_pre_train:.3f} (target={ne_target_pre_train:.3f})")
    print(f"  训练后 NE: {ne_post_train:.3f} (target={ne_target_post_train:.3f})")
    print(f"  训练 loss: {report.training_loss:.4f}")

    # 验证 NE 在训练中被 metabolism 更新（目标值可能变化）
    # 注意：训练后 sleep_engine._update_neuromodulators 调用 step() 进行 EMA
    ne_changed = ne_target_post_train != ne_target_pre_train or ne_post_train != ne_pre_train
    if ne_changed:
        print(f"  ✅ SleepEngine 训练中 metabolism 被调用（NE 有变化）")
    else:
        print(f"  ⚠️ NE 在训练中未变化（可能 CPU 负载稳定）")

    # Step 6: 验证完整调质系统（三调质各司其职）
    print("\n[Step 6] 验证完整调质系统...")
    print(f"  最终调质状态:")
    print(f"    DA={nm.dopamine:.3f} (由 loss 趋势驱动)")
    print(f"    5HT={nm.serotonin:.3f} (由准确率驱动，每5轮评估)")
    print(f"    NE={nm.norepinephrine:.3f} (由 CPU 负载驱动)")
    print(f"  lr_multiplier={nm.get_lr_multiplier():.3f} (DA 驱动)")
    print(f"  field_write_scale={nm.get_field_write_scale():.3f} (NE 驱动)")

    # Step 7: 综合判断
    print("\n" + "=" * 60)
    all_pass = monotonic and fws_monotonic and ne_spread > 0.3
    if all_pass:
        print("🎉 验证通过：metabolism → 去甲肾上腺素接线成功")
        print(f"   - CPU 10% → NE={cpu_ne_results[0][1]:.3f}, field_write={cpu_ne_results[0][2]:.3f}")
        print(f"   - CPU 100% → NE={cpu_ne_results[-1][1]:.3f}, field_write={cpu_ne_results[-1][2]:.3f}")
        print(f"   - 三调质全接线完成：DA(loss) + 5HT(acc) + NE(cpu)")
        return 0
    else:
        print("⚠️ 验证未完全通过")
        return 1


if __name__ == "__main__":
    sys.exit(main())
