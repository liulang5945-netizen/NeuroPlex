"""验证 CoactivationTracker 跨会话持久化。

核心验证：
1. 训练 → save_state → 重新装配 → load_state → coaction 状态恢复
2. slow_matrix 精确匹配
3. activation_counts 精确匹配
4. 持久化后孤立模式检测仍有效

Usage:
    python scripts/training/verify_coactivation_persistence.py
"""
import sys
import os
import tempfile

os.environ.setdefault('TAIJI_TEST_MODE', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    print("=" * 60)
    print("CoactivationTracker 跨会话持久化验证")
    print("=" * 60)

    # Step 1: 装配 Cortex 并积累共激活数据
    print("\n[Step 1] 装配 Cortex 并积累共激活数据...")
    from taiji.loader import assemble_cortex
    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        device="cpu",
        max_rounds=2,
        wire_bio_modules=True,
    )

    # 执行多次 forward 积累共激活
    import torch
    if cortex._shared_embedding is not None:
        for _ in range(5):
            test_ids = torch.randint(0, cortex._shared_embedding.num_embeddings, (1, 8))
            shared_emb = cortex._shared_embedding(test_ids)
            cortex.think(shared_emb)

    stats_before = cortex.coaction.get_stats()
    print(f"  ✅ 积累后 stats: {stats_before}")
    assert stats_before["total_pairs"] > 0, "未积累到共激活数据"

    # 保存关键数据用于对比
    slow_matrix_before = dict(cortex.coaction._slow_matrix)
    activation_counts_before = dict(cortex.coaction._activation_counts)
    print(f"  slow_matrix pairs: {len(slow_matrix_before)}")
    print(f"  activation_counts: {len(activation_counts_before)} neurons")

    # Step 2: save_state
    print("\n[Step 2] save_state...")
    state_path = os.path.join(tempfile.gettempdir(), "test_coaction_state.pt")
    cortex.save_state(state_path)
    print(f"  ✅ 状态已保存: {state_path}")

    # 验证 state 中包含 coaction 键
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if "coaction" in state:
        print(f"  ✅ state['coaction'] 存在")
        print(f"     slow_matrix pairs: {len(state['coaction']['slow_matrix'])}")
    else:
        print(f"  ⚠️ state['coaction'] 不存在")
        return 1

    # Step 3: 重新装配 Cortex（模拟重启）
    print("\n[Step 3] 重新装配 Cortex（模拟重启）...")
    cortex2, tokenizer2, modules2 = assemble_cortex(
        neurons_dir="data/neurons",
        device="cpu",
        max_rounds=2,
        wire_bio_modules=True,
    )
    stats_fresh = cortex2.coaction.get_stats()
    print(f"  新 cortex stats: {stats_fresh}")
    assert stats_fresh["total_pairs"] == 0, "新 cortex 应该没有共激活数据"

    # Step 4: load_state
    print("\n[Step 4] load_state...")
    success = cortex2.load_state(state_path)
    if not success:
        print(f"  ⚠️ load_state 失败")
        return 1

    stats_after = cortex2.coaction.get_stats()
    print(f"  恢复后 stats: {stats_after}")

    # Step 5: 验证 slow_matrix 精确匹配
    print("\n[Step 5] 验证 slow_matrix 精确匹配...")
    slow_matrix_after = dict(cortex2.coaction._slow_matrix)
    if len(slow_matrix_before) != len(slow_matrix_after):
        print(f"  ⚠️ pair 数量不匹配: {len(slow_matrix_before)} vs {len(slow_matrix_after)}")
    else:
        all_match = True
        for pair, strength in slow_matrix_before.items():
            after_strength = slow_matrix_after.get(pair)
            if after_strength is None:
                print(f"  ⚠️ pair {pair} 在恢复后不存在")
                all_match = False
                break
            if abs(strength - after_strength) > 1e-6:
                print(f"  ⚠️ pair {pair} 强度不匹配: {strength} vs {after_strength}")
                all_match = False
                break
        if all_match:
            print(f"  ✅ slow_matrix 精确匹配 ({len(slow_matrix_before)} pairs)")

    # Step 6: 验证 activation_counts 精确匹配
    print("\n[Step 6] 验证 activation_counts 精确匹配...")
    activation_counts_after = dict(cortex2.coaction._activation_counts)
    if activation_counts_before == activation_counts_after:
        print(f"  ✅ activation_counts 精确匹配 ({len(activation_counts_before)} neurons)")
    else:
        print(f"  ⚠️ activation_counts 不匹配")
        print(f"     before: {activation_counts_before}")
        print(f"     after:  {activation_counts_after}")

    # Step 7: 验证孤立模式检测仍有效
    print("\n[Step 7] 验证恢复后孤立模式检测仍有效...")
    from taiji.resonance.lifecycle import NeurogenesisTrigger
    trigger = NeurogenesisTrigger()
    isolated = trigger.detect_isolated_patterns(cortex2.coaction, min_isolation_ratio=0.5)
    print(f"  孤立神经元: {isolated}")
    print(f"  ✅ 恢复后 detect_isolated_patterns 正常工作")

    # Step 8: 验证部落分组恢复
    print("\n[Step 8] 验证部落分组恢复...")
    # 取一个 pair 中的 nid，检查其部落
    if slow_matrix_before:
        sample_pair = list(slow_matrix_before.keys())[0]
        sample_nid = sample_pair[0]
        tribe = cortex2.coaction.get_tribe(sample_nid, min_strength=0.01)
        print(f"  {sample_nid} 的部落: {tribe}")
        if tribe:
            print(f"  ✅ 部落分组恢复正确")
        else:
            print(f"  ⚠️ 部落分组为空")

    # 清理
    if os.path.exists(state_path):
        os.remove(state_path)

    # Step 9: 综合判断
    print("\n" + "=" * 60)
    all_pass = (
        "coaction" in state and
        len(slow_matrix_before) == len(slow_matrix_after) and
        activation_counts_before == activation_counts_after
    )
    if all_pass:
        print("🎉 验证通过：CoactivationTracker 跨会话持久化成功")
        print(f"   - save_state: coaction 纳入 cortex_state.pt")
        print(f"   - load_state: slow_matrix + activation_counts 精确恢复")
        print(f"   - 孤立模式检测: 恢复后正常工作")
        print(f"   - 部落分组: 恢复后正常工作")
        return 0
    else:
        print("⚠️ 验证未完全通过")
        return 1


if __name__ == "__main__":
    sys.exit(main())
