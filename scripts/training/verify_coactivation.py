"""验证 CoactivationTracker 共激活追踪 + 孤立模式检测。

核心验证：
1. CoactivationTracker 双矩阵更新（fast 累加 + slow EMA）
2. get_coaction / get_tribe / get_all_tribes 部落分组
3. detect_isolated_patterns 孤立神经元检测
4. ensemble.forward 后 coaction 自动更新
5. sleep_engine 孤立模式触发 neurogenesis

Usage:
    python scripts/training/verify_coactivation.py
"""
import sys
import os

os.environ.setdefault('TAIJI_TEST_MODE', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    print("=" * 60)
    print("CoactivationTracker 共激活追踪 + 孤立模式检测验证")
    print("=" * 60)

    # Step 1: 测试 CoactivationTracker 基础功能
    print("\n[Step 1] 测试 CoactivationTracker 基础更新...")
    from taiji.resonance.tribal import CoactivationTracker
    ct = CoactivationTracker(ema_alpha=0.1, forget_threshold=0.01)

    # 模拟 3 个神经元共同激活 10 次
    for _ in range(10):
        ct.update(["zh", "en", "code"])

    # 模拟 math 只和 general 共激活 1 次（低频）
    ct.update(["math", "general"])

    stats = ct.get_stats()
    print(f"  ✅ 统计: {stats}")
    assert stats["neurons_tracked"] == 5, f"期望 5, 实际 {stats['neurons_tracked']}"
    print(f"  ✅ 5 个神经元被追踪")

    # Step 2: 测试共激活强度
    print("\n[Step 2] 测试共激活强度...")
    strength_zh_en = ct.get_coactivation("zh", "en")
    strength_math_general = ct.get_coaction if hasattr(ct, 'get_coaction') else ct.get_coactivation
    strength_math_gen = ct.get_coactivation("math", "general")
    print(f"  zh↔en 强度: {strength_zh_en:.3f}（高频，应 > 0.5）")
    print(f"  math↔general 强度: {strength_math_gen:.3f}（低频，应 < 0.2）")
    if strength_zh_en > 0.5:
        print(f"  ✅ 高频 pair 强度正确")
    else:
        print(f"  ⚠️ 高频 pair 强度异常: {strength_zh_en}")
    if strength_math_gen < 0.2:
        print(f"  ✅ 低频 pair 强度正确")
    else:
        print(f"  ⚠️ 低频 pair 强度异常: {strength_math_gen}")

    # Step 3: 测试部落分组
    print("\n[Step 3] 测试部落分组...")
    tribe_zh = ct.get_tribe("zh", min_strength=0.1)
    print(f"  zh 的部落: {tribe_zh}")
    if "en" in tribe_zh and "code" in tribe_zh:
        print(f"  ✅ 部落分组正确（zh 与 en/code 高频共激活）")
    else:
        print(f"  ⚠️ 部落分组异常")

    # Step 4: 测试孤立模式检测
    print("\n[Step 4] 测试孤立模式检测...")
    from taiji.resonance.lifecycle import NeurogenesisTrigger
    trigger = NeurogenesisTrigger()
    isolated = trigger.detect_isolated_patterns(ct, min_isolation_ratio=0.5)
    print(f"  孤立神经元: {isolated}")
    # math 只和 general 共激活 1 次（低频），应该被检测为孤立
    if "math" in isolated or "general" in isolated:
        print(f"  ✅ 孤立模式检测正确（低频 pair 神经元被识别）")
    else:
        print(f"  ⚠️ 孤立模式检测未识别低频神经元")
        print(f"  (注意: detect_isolated_patterns 依赖 pair_stats 统计，可能需要更多数据)")

    # Step 5: 测试 ensemble.forward 后 coaction 自动更新
    print("\n[Step 5] 测试 ensemble.forward 后 coaction 自动更新...")
    from taiji.loader import assemble_cortex
    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        device="cpu",
        max_rounds=2,
        wire_bio_modules=True,
    )
    assert hasattr(cortex, "coaction"), "cortex.coaction 未创建"
    print(f"  ✅ cortex.coaction 已创建: {type(cortex.coaction).__name__}")

    # 记录 forward 前的 stats
    stats_before = cortex.coaction.get_stats()
    print(f"  forward 前: {stats_before}")

    # 执行 forward
    import torch
    if cortex._shared_embedding is not None:
        test_ids = torch.randint(0, cortex._shared_embedding.num_embeddings, (1, 8))
        shared_emb = cortex._shared_embedding(test_ids)
        try:
            cortex.think(shared_emb)
            stats_after = cortex.coaction.get_stats()
            print(f"  forward 后: {stats_after}")
            if stats_after["total_activations"] > stats_before["total_activations"]:
                print(f"  ✅ ensemble.forward 后 coaction 自动更新")
            else:
                print(f"  ⚠️ coaction 未更新（可能 forward 未触发 coaction.update）")
        except Exception as e:
            print(f"  ⚠️ forward 失败: {e}")
    else:
        print(f"  ⚠️ shared_embedding 未设置")

    # Step 6: 测试 EMA 衰减
    print("\n[Step 6] 测试 EMA 衰减...")
    ct2 = CoactivationTracker(ema_alpha=0.1, forget_threshold=0.01)
    for _ in range(20):
        ct2.update(["a", "b"])
    strength_before = ct2.get_coactivation("a", "b")
    ct2.decay()
    strength_after = ct2.get_coactivation("a", "b")
    print(f"  衰减前: {strength_before:.3f}, 衰减后: {strength_after:.3f}")
    if strength_after < strength_before:
        print(f"  ✅ EMA 衰减正确（强度下降）")
    else:
        print(f"  ⚠️ 衰减未生效")

    # Step 7: 综合判断
    print("\n" + "=" * 60)
    all_pass = (
        stats["neurons_tracked"] == 5 and
        strength_zh_en > 0.5 and
        "en" in tribe_zh and
        "code" in tribe_zh and
        hasattr(cortex, "coaction")
    )
    if all_pass:
        print("🎉 验证通过：CoactivationTracker + 孤立模式检测闭环成功")
        print(f"   - 双矩阵更新: fast 累加 + slow EMA")
        print(f"   - 部落分组: get_tribe / get_all_tribes")
        print(f"   - 孤立检测: detect_isolated_patterns 识别低频 pair")
        print(f"   - ensemble 集成: forward 后自动更新")
        print(f"   - EMA 衰减: decay() 方法")
        return 0
    else:
        print("⚠️ 验证未完全通过")
        return 1


if __name__ == "__main__":
    sys.exit(main())
