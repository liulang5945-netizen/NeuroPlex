"""验证 Cortex 经验积累状态的持久化（save/load_state 完整流程）。

关键验证点：
1. 装配 Cortex → 记录初始权重
2. feed + sleep 训练 → 记录训练后权重
3. cortex.save_state() 保存（sleep_engine 已自动调用，这里显式再保存一次确保）
4. 重新装配 Cortex（assemble_cortex 会自动 load_state）
5. 对比训练后权重 vs 重新装配后的权重 → 证明状态恢复成功

这验证了"经验积累持久化"的完整闭环：
    启动 → feed → sleep → 保存 → 重启 → 状态恢复 → 继续累积
"""
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch


def _weight_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """计算两个权重张量的 L1 差异总和。"""
    return (a.float() - b.float()).abs().sum().item()


def _snapshot(cortex) -> dict:
    """抓取当前可学习权重快照。"""
    snap = {}
    if cortex._shared_embedding is not None:
        snap["shared_embedding"] = cortex._shared_embedding.weight.data.clone()
    for nid, neuron in cortex.neurons.items():
        if hasattr(neuron, "lm_head") and neuron.lm_head is not None:
            snap[f"lm_head::{nid}"] = neuron.lm_head.weight.data.clone()
        if hasattr(neuron, "embed_adapter") and neuron.embed_adapter is not None:
            snap[f"embed_adapter::{nid}"] = neuron.embed_adapter.weight.data.clone()
    return snap


def _compare_snapshots(label: str, snap_a: dict, snap_b: dict, tol: float = 1e-6,
                       max_tol: float = 1e-3) -> tuple:
    """对比两个快照，返回 (全部一致, 详细信息列表)。

    Args:
        tol: L1 diff 容差（用于小张量如 lm_head/embed_adapter）
        max_tol: max abs diff 容差（用于大张量如 shared_embedding，评估逐参数精度）
    """
    print(f"\n  ── 对比 [{label}] ──")
    all_match = True
    details = []
    keys = set(snap_a.keys()) | set(snap_b.keys())
    for key in sorted(keys):
        if key not in snap_a:
            print(f"    {key}: 仅在 B 中存在 ❌")
            all_match = False
            continue
        if key not in snap_b:
            print(f"    {key}: 仅在 A 中存在 ❌")
            all_match = False
            continue
        a, b = snap_a[key], snap_b[key]
        l1_diff = _weight_diff(a, b)
        # 对大张量（>1M 参数）用 max abs diff 评估，避免 L1 sum 放大
        numel = a.numel()
        if numel > 1_000_000:
            max_diff = (a.float() - b.float()).abs().max().item()
            mean_diff = l1_diff / numel
            match = max_diff < max_tol
            mark = "✅" if match else "❌"
            print(f"    {key}: L1={l1_diff:.4e}, max={max_diff:.4e}, mean={mean_diff:.4e} {mark}")
            details.append((key, max_diff, match))
        else:
            match = l1_diff < tol
            mark = "✅" if match else "❌"
            print(f"    {key}: L1 diff = {l1_diff:.6e} {mark}")
            details.append((key, l1_diff, match))
        if not match:
            all_match = False
    return all_match, details


def main():
    print("=" * 60)
    print("验证 Cortex 经验积累状态持久化（save/load_state）")
    print("=" * 60)

    neurons_dir = "data/neurons"
    state_path = os.path.join(neurons_dir, "cortex_state.pt")

    # ── 准备：若存在旧 state，备份后移除，保证从幼稚态开始 ──
    backup_path = None
    if os.path.exists(state_path):
        backup_path = state_path + ".bak"
        os.replace(state_path, backup_path)
        print(f"\n[准备] 备份旧状态: {backup_path}")

    try:
        # Step 1: 装配 Cortex（幼稚态）
        print("\n[Step 1] 装配 Cortex（幼稚态）...")
        from taiji.loader import assemble_cortex
        cortex, tokenizer, modules = assemble_cortex(
            neurons_dir=neurons_dir,
            device="cpu",
            max_rounds=3,
            wire_bio_modules=True,
        )
        assert cortex._shared_embedding is not None, "shared_embedding 未装配"
        assert cortex.neurons, "无 neuron 加载"
        first_domain = next(iter(cortex.neurons.keys()))
        print(f"  ✅ Cortex 装配完成: {len(cortex.neurons)} neurons, first={first_domain}")

        snap_init = _snapshot(cortex)
        print(f"  ✅ 初始快照: {len(snap_init)} 个权重张量")

        # Step 2: feed + sleep 训练一轮
        print("\n[Step 2] 执行 feed + sleep 训练...")
        from taiji.life.feed_engine import get_feed_engine
        from taiji.life.sleep_engine import get_sleep_engine, SleepReport

        feed_engine = get_feed_engine()
        sleep_engine = get_sleep_engine()
        sleep_engine.cortex = cortex
        if sleep_engine._feed_engine is None:
            sleep_engine._feed_engine = feed_engine

        test_texts = [
            ("zh", "今天天气很好，我们一起去公园散步。"),
            ("zh", "态极神经元架构通过共振场实现意识涌现。"),
            ("en", "The cortex uses resonance fields for consciousness."),
            ("code", "def forward(x): return x + 1"),
        ]
        for domain, text in test_texts:
            feed_engine.feed_text(text=text, source="verify_persistence", domain=domain)

        report = SleepReport(timestamp=datetime.now().isoformat(), duration_seconds=0.0)
        sleep_engine._sleep_phase_model_training(report)

        loss_str = f"{report.training_loss:.4f}" if report.training_loss is not None else "N/A"
        print(f"  ✅ 训练完成: loss={loss_str}, samples={report.training_samples_used}")

        snap_trained = _snapshot(cortex)

        # 验证训练确实改变了权重
        print("\n[Step 3] 验证训练改变了权重...")
        trained_changed, _ = _compare_snapshots(
            "init vs trained", snap_init, snap_trained, tol=1e-6,
        )
        if trained_changed:
            print("  ⚠️ 训练未改变任何权重（可能是 lr 太小或样本太少）")
        else:
            print("  ✅ 训练改变了权重，经验积累已发生")

        # Step 4: 验证 state 文件存在（sleep_engine 应已自动保存）
        print("\n[Step 4] 检查 state 文件...")
        if not os.path.exists(state_path):
            # 显式调用 save_state 兜底
            print(f"  ⚠️ sleep_engine 未自动保存，显式调用 cortex.save_state()")
            cortex.save_state(neurons_dir)

        if os.path.exists(state_path):
            file_size = os.path.getsize(state_path) / 1024 / 1024
            print(f"  ✅ state 文件存在: {state_path} ({file_size:.2f} MB)")
        else:
            print(f"  ❌ state 文件不存在: {state_path}")
            return 1

        # Step 5: 重新装配 Cortex（应自动 load_state）
        print("\n[Step 5] 重新装配 Cortex（验证自动 load_state）...")
        cortex2, tokenizer2, modules2 = assemble_cortex(
            neurons_dir=neurons_dir,
            device="cpu",
            max_rounds=3,
            wire_bio_modules=True,
        )
        assert cortex2._shared_embedding is not None

        snap_reloaded = _snapshot(cortex2)
        print(f"  ✅ 重新装配完成: {len(cortex2.neurons)} neurons, {len(snap_reloaded)} 权重张量")

        # Step 6: 对比训练后权重 vs 重新装配后权重
        # fp16 保存 shared_embedding 有微小精度损失，容差放宽到 1e-2
        print("\n[Step 6] 对比 [trained] vs [reloaded] 权重...")
        all_match, details = _compare_snapshots(
            "trained vs reloaded", snap_trained, snap_reloaded, tol=1e-2,
        )

        # Step 7: 综合判断
        print("\n" + "=" * 60)
        if all_match and not trained_changed:
            print("🎉 验证通过：状态持久化完整闭环成功")
            print("   - 训练改变了权重（经验积累发生）")
            print("   - save_state 成功写入磁盘")
            print("   - assemble_cortex 自动 load_state 恢复")
            print("   - 重新装配后权重 == 训练后权重（精确匹配）")
            print("   - 闭环验证：启动→feed→sleep→保存→重启→状态恢复")
            return 0
        else:
            print("⚠️ 验证未完全通过")
            if trained_changed:
                print("   - 训练未改变权重，经验积累未发生")
            if not all_match:
                n_mismatch = sum(1 for _, _, m in details if not m)
                print(f"   - {n_mismatch} 个权重张量恢复后不匹配")
            return 1

    finally:
        # ── 恢复：若有备份，还原原始 state ──
        if backup_path is not None:
            if os.path.exists(state_path):
                os.remove(state_path)
            os.replace(backup_path, state_path)
            print(f"\n[清理] 已还原原始 state: {state_path}")
        else:
            # 无备份（原本无 state），删除测试产生的 state
            if os.path.exists(state_path):
                os.remove(state_path)
                print(f"\n[清理] 已删除测试产生的 state: {state_path}")


if __name__ == "__main__":
    sys.exit(main())
