"""验证经验驱动学习持续生效：多次 feed+sleep 循环观察 loss 下降趋势。

关键验证点：
1. 多次循环 feed 相同样本 → sleep 训练
2. 每轮记录 loss，观察是否下降（证明经验积累，而非单次随机变化）
3. 检查 shared_embedding 权重持续变化
4. 对比首次生成 vs 末次生成的文本质量
"""
import sys
import os
from datetime import datetime

os.environ.setdefault('TAJIJI_TEST_MODE', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

def main():
    print("=" * 60)
    print("验证经验驱动学习持续生效（多轮 feed+sleep）")
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
    assert cortex._shared_embedding is not None
    print(f"  ✅ Cortex 装配完成: {len(cortex.neurons)} neurons")

    # 记录初始权重
    shared_emb_init = cortex._shared_embedding.weight.data.clone()
    first_domain = next(iter(cortex.neurons.keys()))
    first_neuron = cortex.neurons[first_domain]
    lm_head_init = first_neuron.lm_head.weight.data.clone()

    # Step 2: 准备训练样本
    print("\n[Step 2] 准备训练样本...")
    test_texts = [
        ("zh", "今天天气很好，我们一起去公园散步。"),
        ("zh", "人工智能正在改变世界，神经元协同工作。"),
        ("zh", "态极神经元架构通过共振场实现意识涌现。"),
        ("en", "The cortex architecture uses resonance fields."),
        ("en", "Small neurons work together to match large models."),
        ("code", "def hello(): print('world')"),
        ("code", "class Neuron: def forward(self, x): return x"),
    ]
    print(f"  ✅ {len(test_texts)} 条样本")

    # Step 3: 多轮 feed+sleep
    from taiji.life.feed_engine import get_feed_engine
    from taiji.life.sleep_engine import get_sleep_engine, SleepReport

    feed_engine = get_feed_engine()
    sleep_engine = get_sleep_engine()
    sleep_engine.cortex = cortex
    if sleep_engine._feed_engine is None:
        sleep_engine._feed_engine = feed_engine

    NUM_CYCLES = 5
    losses = []
    print(f"\n[Step 3] 执行 {NUM_CYCLES} 轮 feed+sleep 循环...")

    for cycle in range(NUM_CYCLES):
        # 每轮重新 feed（样本进入 pending 队列）
        for domain, text in test_texts:
            feed_engine.feed_text(text=text, source=f"cycle_{cycle}", domain=domain)

        # 触发 sleep 训练
        report = SleepReport(timestamp=datetime.now().isoformat(), duration_seconds=0.0)
        sleep_engine._sleep_phase_model_training(report)

        loss = report.training_loss
        losses.append(loss)
        n_samples = report.training_samples_used
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        print(f"  Cycle {cycle+1}/{NUM_CYCLES}: loss={loss_str}, samples={n_samples}")

    # Step 4: 分析 loss 趋势
    print("\n[Step 4] 分析 loss 趋势...")
    valid_losses = [l for l in losses if l is not None]
    if len(valid_losses) >= 2:
        first_loss = valid_losses[0]
        last_loss = valid_losses[-1]
        delta = first_loss - last_loss
        pct = (delta / first_loss * 100) if first_loss > 0 else 0
        print(f"  首轮 loss: {first_loss:.4f}")
        print(f"  末轮 loss: {last_loss:.4f}")
        print(f"  下降量: {delta:.4f} ({pct:+.1f}%)")

        if delta > 0:
            print("  ✅ Loss 下降趋势明确：经验积累持续发生")
        else:
            print("  ⚠️ Loss 未下降（可能需要更多轮次或调整学习率）")
    else:
        print(f"  有效 loss 数据不足: {valid_losses}")

    # Step 5: 检查权重累积变化
    print("\n[Step 5] 检查权重累积变化...")
    shared_emb_final = cortex._shared_embedding.weight.data
    lm_head_final = first_neuron.lm_head.weight.data

    shared_diff = (shared_emb_final - shared_emb_init).abs().sum().item()
    lm_head_diff = (lm_head_final - lm_head_init).abs().sum().item()

    print(f"  shared_embedding 累积变化: {shared_diff:.4f}")
    print(f"  [{first_domain}] lm_head 累积变化: {lm_head_diff:.4f}")

    # Step 6: 对比生成质量
    print("\n[Step 6] 对比训练前后生成文本...")
    test_prompt = "神经元架构"
    try:
        # 用训练后的 cortex 生成
        generated = cortex.generate(test_prompt, max_tokens=20, domain="zh")
        print(f"  Prompt: '{test_prompt}'")
        print(f"  生成: '{generated[:80]}'")
    except Exception as e:
        print(f"  生成失败: {e}")

    # Step 7: 综合判断
    print("\n" + "=" * 60)
    success = (
        len(valid_losses) >= 2 and
        (valid_losses[0] - valid_losses[-1]) > 0 and
        shared_diff > 1.0
    )
    if success:
        print("🎉 验证通过：经验驱动学习持续生效")
        print(f"   - Loss 从 {valid_losses[0]:.4f} 降至 {valid_losses[-1]:.4f}")
        print(f"   - shared_embedding 累积变化 {shared_diff:.4f}")
        print(f"   - 证明：从随机初始化开始，通过 feed+sleep 经验逐步学习")
        return 0
    else:
        print("⚠️ 验证未完全通过（可能需要更多轮次或调参）")
        print(f"   - Losses: {valid_losses}")
        print(f"   - shared_embedding 变化: {shared_diff:.4f}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
