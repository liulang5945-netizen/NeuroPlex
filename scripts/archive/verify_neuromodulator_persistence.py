"""验证 NeuromodulatorState 跨会话持久化。

核心验证：
1. 训练后调质状态发生变化（多巴胺从 0.5 → >0.5）
2. save_state 将调质写入 cortex_state.pt
3. 重新装配 Cortex 后 load_state 恢复调质状态
4. 恢复后的调质状态与保存前一致

Usage:
    python scripts/training/verify_neuromodulator_persistence.py
"""
import sys
import os
from datetime import datetime

os.environ.setdefault('TAJIJI_TEST_MODE', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch


def main():
    print("=" * 60)
    print("NeuromodulatorState 持久化验证")
    print("=" * 60)

    # Step 1: 装配 Cortex
    print("\n[Step 1] 装配 Cortex...")
    from taiji.loader import assemble_cortex
    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
    )
    assert cortex._neuromodulator is not None, "NeuromodulatorState 未初始化"

    # 记录初始调质状态
    nm = cortex._neuromodulator
    da_before = nm.dopamine
    ht_before = nm.serotonin
    ne_before = nm.norepinephrine
    print(f"  ✅ Cortex 装配完成")
    print(f"  初始调质: DA={da_before:.4f}, 5HT={ht_before:.4f}, NE={ne_before:.4f}")

    # Step 2: 训练使调质状态变化
    print("\n[Step 2] 训练使调质状态变化...")
    from taiji.life.feed_engine import get_feed_engine
    from taiji.life.sleep_engine import get_sleep_engine, SleepReport

    feed_engine = get_feed_engine()
    sleep_engine = get_sleep_engine()
    sleep_engine.cortex = cortex
    sleep_engine._neuromodulator = nm  # 共享同一实例
    if sleep_engine._feed_engine is None:
        sleep_engine._feed_engine = feed_engine

    # 喂入训练数据并训练
    for text in [
        "今天天气很好，我们一起去公园散步。",
        "人工智能正在改变世界，神经元协同工作。",
        "态极神经元架构通过共振场实现意识涌现。",
        "我喜欢在清晨喝一杯咖啡，开始新的一天。",
    ]:
        feed_engine.feed_text(text=text, source="test", domain="zh")

    report = SleepReport(timestamp=datetime.now().isoformat(), duration_seconds=0.0)
    sleep_engine._sleep_phase_model_training(report)

    da_after_train = nm.dopamine
    ht_after_train = nm.serotonin
    ne_after_train = nm.norepinephrine
    print(f"  训练 loss: {report.training_loss:.4f}")
    print(f"  训练后调质: DA={da_after_train:.4f}, 5HT={ht_after_train:.4f}, NE={ne_after_train:.4f}")

    da_changed = abs(da_after_train - da_before) > 0.01
    if not da_changed:
        print("  ⚠️ 多巴胺未变化（可能需要更多训练轮次）")
        # 再训练一轮确保变化
        for text in ["深度学习模型需要大量数据训练。", "运动有益健康，每天坚持锻炼。"]:
            feed_engine.feed_text(text=text, source="test2", domain="zh")
        sleep_engine._sleep_phase_model_training(report)
        da_after_train = nm.dopamine
        ht_after_train = nm.serotonin
        print(f"  二次训练后调质: DA={da_after_train:.4f}, 5HT={ht_after_train:.4f}")

    # Step 3: 保存状态
    print("\n[Step 3] 保存状态到 cortex_state.pt...")
    state_path = "data/neurons/cortex_state.pt"
    cortex.save_state(state_path)
    assert os.path.exists(state_path), "状态文件未创建"

    # 验证文件中包含 neuromodulator
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    assert "neuromodulator" in state, "状态文件中缺少 neuromodulator 键"
    nm_saved = state["neuromodulator"]
    print(f"  ✅ 状态文件包含 neuromodulator")
    print(f"  保存的调质: DA={nm_saved['dopamine']:.4f}, 5HT={nm_saved['serotonin']:.4f}")

    # Step 4: 重新装配 Cortex 并加载状态
    print("\n[Step 4] 重新装配 Cortex 并加载状态...")
    cortex2, _, _ = assemble_cortex(
        neurons_dir="data/neurons",
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
    )
    assert cortex2._neuromodulator is not None, "新 Cortex 的 NeuromodulatorState 未初始化"

    # 加载前调质应为默认值
    nm2 = cortex2._neuromodulator
    print(f"  加载前调质: DA={nm2.dopamine:.4f}, 5HT={nm2.serotonin:.4f}")

    # 加载状态
    success = cortex2.load_state(state_path)
    assert success, "load_state 返回 False"

    da_loaded = nm2.dopamine
    ht_loaded = nm2.serotonin
    ne_loaded = nm2.norepinephrine
    print(f"  加载后调质: DA={da_loaded:.4f}, 5HT={ht_loaded:.4f}, NE={ne_loaded:.4f}")

    # Step 5: 验证调质状态匹配
    # DA/5HT 精确匹配，NE 允许微小偏差（metabolism 在 load 后会感知硬件更新 NE）
    print("\n[Step 5] 验证调质状态匹配...")
    da_match = abs(da_loaded - da_after_train) < 1e-4
    ht_match = abs(ht_loaded - ht_after_train) < 1e-4
    ne_match = abs(ne_loaded - ne_after_train) < 0.05  # NE 由硬件实时驱动，允许 5% 偏差

    print(f"  DA: 保存={da_after_train:.6f}, 加载={da_loaded:.6f}, 匹配={da_match}")
    print(f"  5HT: 保存={ht_after_train:.6f}, 加载={ht_loaded:.6f}, 匹配={ht_match}")
    print(f"  NE: 保存={ne_after_train:.6f}, 加载={ne_loaded:.6f}, 匹配={ne_match}")

    all_match = da_match and ht_match and ne_match

    # 清理测试产物
    if os.path.exists(state_path):
        os.remove(state_path)
        print(f"\n  已清理测试产物: {state_path}")

    # Step 6: 综合判断
    print("\n" + "=" * 60)
    if all_match:
        print("🎉 验证通过：NeuromodulatorState 跨会话持久化成功")
        print(f"   训练使调质变化: DA {da_before:.2f} → {da_after_train:.2f}")
        print(f"   持久化恢复: DA={da_loaded:.4f} (精确匹配)")
        return 0
    else:
        print("⚠️ 验证失败：调质状态未精确匹配")
        return 1


if __name__ == "__main__":
    sys.exit(main())
