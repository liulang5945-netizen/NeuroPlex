"""验证规模化训练数据下的生成质量改善。

关键验证点：
1. 喂入多样化训练数据（每域 20+ 条，覆盖多主题）
2. 多轮 feed+sleep 训练
3. 每轮评估生成质量：loss、token 多样性、训练数据覆盖率
4. 对比训练前后的生成文本

Usage:
    python scripts/training/verify_scaled_training.py
"""
import sys
import os
from datetime import datetime
from collections import Counter

os.environ.setdefault('TAJIJI_TEST_MODE', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch


# ── 多样化训练数据 ──
TRAINING_DATA = {
    "zh": [
        # 天气主题
        "今天天气很好，我们一起去公园散步。",
        "明天天气怎么样？会下雨吗？",
        "天气预报说周末是晴天，适合出游。",
        # 科技主题
        "人工智能正在改变世界，神经元协同工作。",
        "态极神经元架构通过共振场实现意识涌现。",
        "深度学习模型需要大量数据训练。",
        "大语言模型展现了强大的理解能力。",
        # 生活主题
        "我喜欢在清晨喝一杯咖啡，开始新的一天。",
        "读书是获取知识的重要途径。",
        "运动有益健康，每天坚持锻炼。",
        "音乐能陶冶情操，让人心情愉悦。",
        # 学习主题
        "学习新语言需要耐心和持续练习。",
        "数学是科学的基础，逻辑思维很重要。",
        "历史告诉我们很多道理。",
        # 自然主题
        "春天来了，万物复苏，花开满园。",
        "秋天是收获的季节，树叶变黄飘落。",
        "海洋是地球上最大的生态系统。",
        "森林是地球的肺，提供氧气。",
        # 社会主题
        "城市生活节奏快，交通便利。",
        "乡村生活宁静祥和，空气清新。",
        "教育是改变命运的重要途径。",
        "团队合作能完成更复杂的任务。",
    ],
    "en": [
        "The cortex architecture uses resonance fields.",
        "Small neurons work together to match large models.",
        "Machine learning models learn from data.",
        "Neural networks are inspired by the brain.",
        "Language models process text efficiently.",
        "The weather is nice today, let's go for a walk.",
        "Reading books is important for learning.",
        "Exercise is good for your health.",
        "Music can affect our emotions deeply.",
        "Science advances through careful experimentation.",
        "Technology changes the way we live.",
        "Education opens doors to new opportunities.",
        "Teamwork makes difficult tasks achievable.",
        "Nature provides beauty and resources.",
        "Cities offer convenience and diversity.",
    ],
    "code": [
        "def hello(): print('world')",
        "class Neuron: def forward(self, x): return x",
        "def add(a, b): return a + b",
        "class ResonanceField: def reset(self): pass",
        "def train(model, data): model.fit(data)",
        "import torch; print(torch.__version__)",
        "def generate(prompt): return prompt + ' generated'",
        "class Cortex: def think(self, x): return x",
        "def validate_input(x): assert x is not None",
        "class FeedEngine: def feed(self, data): pass",
        "def save_state(model, path): torch.save(model, path)",
        "class SleepEngine: def sleep(self): pass",
        "def encode(text, tokenizer): return tokenizer.encode(text)",
        "class Ensemble: def forward(self, x): return x",
        "def loss_fn(pred, target): return (pred - target).mean()",
    ],
    "math": [
        "1 + 1 = 2",
        "2 * 3 = 6",
        "x^2 + y^2 = r^2",
        "f(x) = ax + b",
        "integral of x dx = x^2/2 + C",
        "derivative of x^2 is 2x",
        "sin(0) = 0, cos(0) = 1",
        "e^(i*pi) + 1 = 0",
        "limit of 1/x as x -> inf is 0",
        "sum of 1 to n is n*(n+1)/2",
        "a^2 + b^2 = c^2 for right triangles",
        "log(a*b) = log(a) + log(b)",
        "The area of circle is pi * r^2",
        "The volume of sphere is 4/3 * pi * r^3",
        "Matrix multiplication is associative",
    ],
}


def compute_diversity(text: str) -> float:
    """计算文本 token 多样性（unique chars / total chars）"""
    if not text:
        return 0.0
    chars = list(text.replace(" ", "").replace("▁", ""))
    if not chars:
        return 0.0
    return len(set(chars)) / len(chars)


def compute_repetition_ratio(text: str) -> float:
    """计算重复率（最高频 char 出现次数 / 总长度）"""
    if not text:
        return 0.0
    chars = list(text.replace(" ", "").replace("▁", ""))
    if not chars:
        return 0.0
    counter = Counter(chars)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count / len(chars)


def main():
    print("=" * 60)
    print("规模化训练数据生成质量验证")
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
    total_samples = sum(len(v) for v in TRAINING_DATA.values())
    print(f"  ✅ Cortex 装配完成: {len(cortex.neurons)} neurons")
    print(f"  训练数据: {total_samples} 条, 域分布: {', '.join(f'{d}={len(v)}' for d, v in TRAINING_DATA.items())}")

    # Step 2: 训练前基线生成
    print("\n[Step 2] 训练前基线生成...")
    test_prompts = [
        ("今天天气", "zh"),
        ("人工智能", "zh"),
        ("def hello", "code"),
        ("1+1=", "math"),
    ]
    baselines = {}
    for prompt, domain in test_prompts:
        try:
            gen = cortex.generate(prompt, max_tokens=30, domain=domain, temperature=0.8)
            diversity = compute_diversity(gen)
            rep_ratio = compute_repetition_ratio(gen)
            baselines[prompt] = (gen, diversity, rep_ratio)
            print(f"  [{domain}] '{prompt}' → '{gen[:50]}' (diversity={diversity:.2f}, rep={rep_ratio:.2f})")
        except Exception as e:
            baselines[prompt] = ("", 0.0, 1.0)
            print(f"  [{domain}] '{prompt}' → 生成失败: {e}")

    # Step 3: 多轮 feed+sleep 训练
    print("\n[Step 3] 多轮 feed+sleep 训练...")
    from taiji.life.feed_engine import get_feed_engine
    from taiji.life.sleep_engine import get_sleep_engine, SleepReport

    feed_engine = get_feed_engine()
    sleep_engine = get_sleep_engine()
    sleep_engine.cortex = cortex
    if sleep_engine._feed_engine is None:
        sleep_engine._feed_engine = feed_engine

    NUM_CYCLES = 5
    losses_by_cycle = []
    domain_losses_history = {d: [] for d in TRAINING_DATA.keys()}

    for cycle in range(NUM_CYCLES):
        # 每轮喂入全部数据
        for domain, texts in TRAINING_DATA.items():
            for text in texts:
                feed_engine.feed_text(text=text, source=f"cycle_{cycle}", domain=domain)

        # 触发 sleep 训练
        report = SleepReport(timestamp=datetime.now().isoformat(), duration_seconds=0.0)
        sleep_engine._sleep_phase_model_training(report)

        loss = report.training_loss
        losses_by_cycle.append(loss)
        n_samples = report.training_samples_used
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"

        # 打印调质状态（自主进化监控）
        nm = sleep_engine._neuromodulator
        if nm is not None:
            lr_mult = nm.get_lr_multiplier()
            print(f"  Cycle {cycle+1}/{NUM_CYCLES}: loss={loss_str}, samples={n_samples} | "
                  f"DA={nm.dopamine:.2f} 5HT={nm.serotonin:.2f} lr_mult={lr_mult:.2f}")
        else:
            print(f"  Cycle {cycle+1}/{NUM_CYCLES}: loss={loss_str}, samples={n_samples}")

    # Step 4: Loss 趋势分析
    print("\n[Step 4] Loss 趋势分析...")
    valid_losses = [l for l in losses_by_cycle if l is not None]
    if len(valid_losses) >= 2:
        first_loss = valid_losses[0]
        last_loss = valid_losses[-1]
        delta = first_loss - last_loss
        pct = (delta / first_loss * 100) if first_loss > 0 else 0
        print(f"  首轮 loss: {first_loss:.4f}")
        print(f"  末轮 loss: {last_loss:.4f}")
        print(f"  下降量: {delta:.4f} ({pct:+.1f}%)")
        if delta > 0:
            print("  ✅ Loss 持续下降")
        else:
            print("  ⚠️ Loss 未下降")

    # Step 5: 训练后生成对比
    print("\n[Step 5] 训练后生成对比...")
    improvements = 0
    total_compared = 0
    for prompt, domain in test_prompts:
        try:
            gen = cortex.generate(prompt, max_tokens=30, domain=domain, temperature=0.8)
            diversity = compute_diversity(gen)
            rep_ratio = compute_repetition_ratio(gen)
            base_gen, base_div, base_rep = baselines[prompt]

            div_improved = diversity > base_div
            rep_improved = rep_ratio < base_rep
            if div_improved or rep_improved:
                improvements += 1
            total_compared += 1

            div_arrow = "↑" if div_improved else "↓" if diversity < base_div else "="
            rep_arrow = "↓" if rep_improved else "↑" if rep_ratio > base_rep else "="
            print(f"  [{domain}] '{prompt}'")
            print(f"    前: '{base_gen[:40]}' (div={base_div:.2f}, rep={base_rep:.2f})")
            print(f"    后: '{gen[:40]}' (div={diversity:.2f}{div_arrow}, rep={rep_ratio:.2f}{rep_arrow})")
        except Exception as e:
            print(f"  [{domain}] '{prompt}' → 生成失败: {e}")

    # Step 6: Next-token 预测准确率
    print("\n[Step 6] Next-token 预测准确率...")
    coverage_results = {}
    for domain in ["zh"]:
        correct = 0
        total = 0
        top5_hits = 0
        domain_sp = cortex._tokenizer_hub.get_tokenizer(domain)

        for text in TRAINING_DATA[domain][:10]:  # 检查前 10 条
            domain_ids = cortex._tokenizer_hub.encode(text, domain=domain)
            if len(domain_ids) < 4:
                continue

            # 用逐 token 映射构造输入（与训练路径一致）
            general_ids = []
            for did in domain_ids:
                piece = domain_sp.id_to_piece(did)
                gen_ids = cortex._general_sp.EncodeAsIds(piece)
                if gen_ids:
                    general_ids.append(gen_ids[0])

            # 对每个位置，用前缀预测下一个 token
            for i in range(1, min(len(general_ids) - 1, 10)):
                prefix = general_ids[:i+1]
                if len(prefix) < 2:
                    continue
                ids_tensor = torch.tensor([prefix], dtype=torch.long)
                shared_emb = cortex._shared_embedding(ids_tensor)

                with torch.no_grad():
                    result = cortex.think(shared_emb)
                logits = result.get("neuron_logits", {}).get(domain)
                if logits is None:
                    continue

                last_logits = logits[0, -1, :]
                pred_token = torch.argmax(last_logits).item()
                true_token = domain_ids[i+1] if i+1 < len(domain_ids) else domain_ids[-1]

                total += 1
                if pred_token == true_token:
                    correct += 1
                top5 = torch.topk(last_logits, 5).indices.tolist()
                if true_token in top5:
                    top5_hits += 1

        accuracy = correct / total if total > 0 else 0
        top5_acc = top5_hits / total if total > 0 else 0
        coverage_results[domain] = accuracy
        print(f"  [{domain}] next-token 准确率: {accuracy:.1%} ({correct}/{total})")
        print(f"  [{domain}] top-5 准确率: {top5_acc:.1%} ({top5_hits}/{total})")

    # Step 7: 综合判断
    print("\n" + "=" * 60)
    loss_success = len(valid_losses) >= 2 and (valid_losses[0] - valid_losses[-1]) > 0
    quality_success = improvements >= total_compared * 0.5 if total_compared > 0 else False
    coverage_success = any(c > 0.05 for c in coverage_results.values()) if coverage_results else False

    if loss_success and (quality_success or coverage_success):
        print("🎉 验证通过：规模化训练数据改善生成质量")
        print(f"   - Loss: {valid_losses[0]:.4f} → {valid_losses[-1]:.4f}")
        print(f"   - 生成质量改善: {improvements}/{total_compared}")
        if coverage_results:
            print(f"   - Token 覆盖率: {', '.join(f'{d}={c:.1%}' for d, c in coverage_results.items())}")
        return 0
    else:
        print("⚠️ 验证未完全通过")
        print(f"   - Loss 下降: {loss_success}")
        print(f"   - 质量改善: {improvements}/{total_compared}")
        print(f"   - Token 覆盖: {coverage_success}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
