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


# ── 多样化训练数据（200+ 条，每域 50+） ──
TRAINING_DATA = {
    "zh": [
        # 天气主题 (8)
        "今天天气很好，我们一起去公园散步。",
        "明天天气怎么样？会下雨吗？",
        "天气预报说周末是晴天，适合出游。",
        "夏天的雷阵雨总是来得很突然。",
        "冬天的雪景真美，孩子们在堆雪人。",
        "春天暖风吹过，百花竞相开放。",
        "秋高气爽，蓝天白云令人心旷神怡。",
        "台风来了，记得关好窗户待在家里。",
        # 科技主题 (8)
        "人工智能正在改变世界，神经元协同工作。",
        "态极神经元架构通过共振场实现意识涌现。",
        "深度学习模型需要大量数据训练。",
        "大语言模型展现了强大的理解能力。",
        "量子计算机将在未来解决复杂问题。",
        "区块链技术让数据更加安全可靠。",
        "云计算让企业无需自建数据中心。",
        "物联网连接了万物，让生活更智能。",
        # 生活主题 (8)
        "我喜欢在清晨喝一杯咖啡，开始新的一天。",
        "读书是获取知识的重要途径。",
        "运动有益健康，每天坚持锻炼。",
        "音乐能陶冶情操，让人心情愉悦。",
        "做饭是一门艺术，需要不断练习。",
        "旅行能开阔眼界，增长见识。",
        "和朋友聚会是最快乐的时光。",
        "养花种草让生活充满绿意。",
        # 学习主题 (8)
        "学习新语言需要耐心和持续练习。",
        "数学是科学的基础，逻辑思维很重要。",
        "历史告诉我们很多道理。",
        "物理研究自然界的基本规律。",
        "化学实验需要注意安全操作。",
        "写作能力需要通过大量阅读来提升。",
        "编程是一项实用的技能。",
        "哲学思考帮助人们认识自我。",
        # 自然主题 (8)
        "春天来了，万物复苏，花开满园。",
        "秋天是收获的季节，树叶变黄飘落。",
        "海洋是地球上最大的生态系统。",
        "森林是地球的肺，提供氧气。",
        "河流奔腾不息，汇入大海。",
        "高山巍峨耸立，终年积雪。",
        "沙漠虽然干旱，也有独特生态。",
        "极地的极光绚丽多彩，美不胜收。",
        # 社会主题 (8)
        "城市生活节奏快，交通便利。",
        "乡村生活宁静祥和，空气清新。",
        "教育是改变命运的重要途径。",
        "团队合作能完成更复杂的任务。",
        "诚信是为人之本，不可丢弃。",
        "创新推动社会不断向前发展。",
        "文化交流增进不同民族的理解。",
        "保护环境是每个人的责任。",
        # 情感主题 (6)
        "快乐不在于拥有多少，而在于知足。",
        "勇气不是不害怕，而是害怕了仍然前行。",
        "友谊需要用心经营，才能长久。",
        "家是温暖的港湾，永远等着你回来。",
        "梦想是前进的动力，不要轻易放弃。",
        "感恩生活中的每一个美好瞬间。",
    ],
    "en": [
        # Technology (10)
        "The cortex architecture uses resonance fields.",
        "Small neurons work together to match large models.",
        "Machine learning models learn from data.",
        "Neural networks are inspired by the brain.",
        "Language models process text efficiently.",
        "Quantum computers will solve complex problems.",
        "Blockchain makes data secure and reliable.",
        "Cloud computing eliminates data centers.",
        "The internet connects people worldwide.",
        "Artificial intelligence transforms industries.",
        # Daily life (10)
        "The weather is nice today, let's go for a walk.",
        "Reading books is important for learning.",
        "Exercise is good for your health.",
        "Music can affect our emotions deeply.",
        "Cooking is an art that requires practice.",
        "Travel broadens the mind and perspective.",
        "Friends make life more enjoyable.",
        "Gardening brings joy and beauty.",
        "A good cup of coffee starts the day right.",
        "Sleep is essential for good health.",
        # Science (8)
        "Science advances through careful experimentation.",
        "Technology changes the way we live.",
        "Physics studies the laws of nature.",
        "Chemistry explores matter and its properties.",
        "Biology investigates living organisms.",
        "Astronomy reveals the mysteries of universe.",
        "Mathematics is the language of science.",
        "Evolution explains the diversity of life.",
        # Society (8)
        "Education opens doors to new opportunities.",
        "Teamwork makes difficult tasks achievable.",
        "Nature provides beauty and resources.",
        "Cities offer convenience and diversity.",
        "Honesty is the foundation of trust.",
        "Innovation drives society forward.",
        "Cultural exchange promotes understanding.",
        "Protecting the environment is our duty.",
        # Emotions (8)
        "Happiness comes from within, not from things.",
        "Courage is acting despite fear.",
        "Friendship requires effort to maintain.",
        "Home is where the heart belongs.",
        "Dreams give us motivation to continue.",
        "Gratitude makes life more meaningful.",
        "Love is the most powerful force.",
        "Patience brings rewards in time.",
        # Nature (6)
        "The ocean is the largest ecosystem on Earth.",
        "Mountains rise majestically into the sky.",
        "Forests provide oxygen for the planet.",
        "Rivers flow endlessly to the sea.",
        "Seasons change in a beautiful cycle.",
        "Stars light up the night sky.",
    ],
    "code": [
        # Functions (12)
        "def hello(): print('world')",
        "def add(a, b): return a + b",
        "def multiply(x, y): return x * y",
        "def is_even(n): return n % 2 == 0",
        "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
        "def reverse(s): return s[::-1]",
        "def max_of_list(lst): return max(lst)",
        "def count_words(text): return len(text.split())",
        "def fib(n): a, b = 0, 1\n    for _ in range(n): a, b = b, a+b\n    return a",
        "def sort_list(items): return sorted(items)",
        "def check_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
        "def gcd(a, b): return a if b == 0 else gcd(b, a % b)",
        # Classes (12)
        "class Neuron: def forward(self, x): return x",
        "class ResonanceField: def reset(self): pass",
        "class Cortex: def think(self, x): return x",
        "class FeedEngine: def feed(self, data): pass",
        "class SleepEngine: def sleep(self): pass",
        "class Ensemble: def forward(self, x): return x",
        "class PlayEngine: def play(self): pass",
        "class Stack:\n    def __init__(self): self.items = []\n    def push(self, x): self.items.append(x)",
        "class Queue:\n    def __init__(self): self.items = []\n    def enqueue(self, x): self.items.append(x)",
        "class Point:\n    def __init__(self, x, y): self.x = x; self.y = y",
        "class Vector:\n    def __init__(self, data): self.data = data\n    def sum(self): return sum(self.data)",
        "class Logger:\n    def info(self, msg): print(f'[INFO] {msg}')",
        # Utility (10)
        "import torch; print(torch.__version__)",
        "import json; data = json.loads('{\"key\": \"value\"}')",
        "import os; files = os.listdir('.')",
        "import sys; print(sys.path)",
        "import math; result = math.sqrt(16)",
        "import random; choice = random.choice([1,2,3])",
        "import datetime; now = datetime.datetime.now()",
        "import collections; counter = collections.Counter('abracadabra')",
        "import itertools; pairs = list(itertools.combinations([1,2,3], 2))",
        "import functools; result = functools.reduce(lambda a,b: a+b, [1,2,3,4])",
        # Patterns (10)
        "def generate(prompt): return prompt + ' generated'",
        "def validate_input(x): assert x is not None",
        "def save_state(model, path): torch.save(model, path)",
        "def load_state(path): return torch.load(path)",
        "def encode(text, tokenizer): return tokenizer.encode(text)",
        "def decode(ids, tokenizer): return tokenizer.decode(ids)",
        "def train(model, data): model.fit(data)",
        "def predict(model, x): return model(x)",
        "def loss_fn(pred, target): return (pred - target).mean()",
        "def accuracy(preds, labels): return (preds == labels).float().mean()",
        # Algorithms (8)
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid+1\n        else: hi = mid-1\n    return -1",
        "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]: arr[j], arr[j+1] = arr[j+1], arr[j]",
        "def merge_sort(arr):\n    if len(arr) <= 1: return arr\n    mid = len(arr)//2\n    return merge(merge_sort(arr[:mid]), merge_sort(arr[mid:]))",
        "def quick_sort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[0]\n    return quick_sort([x for x in arr[1:] if x < pivot]) + [pivot] + quick_sort([x for x in arr[1:] if x >= pivot])",
        "def linear_search(arr, target):\n    for i, x in enumerate(arr):\n        if x == target: return i\n    return -1",
        "def insertion_sort(arr):\n    for i in range(1, len(arr)):\n        key = arr[i]\n        j = i-1\n        while j >= 0 and arr[j] > key:\n            arr[j+1] = arr[j]\n            j -= 1\n        arr[j+1] = key",
        "def depth_first_search(graph, start, visited=None):\n    if visited is None: visited = set()\n    visited.add(start)\n    for next in graph[start] - visited:\n        depth_first_search(graph, next, visited)",
        "def breadth_first_search(graph, start):\n    visited = set([start])\n    queue = [start]\n    while queue:\n        vertex = queue.pop(0)\n        for next in graph[vertex] - visited:\n            visited.add(next)\n            queue.append(next)",
    ],
    "math": [
        # Arithmetic (10)
        "1 + 1 = 2",
        "2 * 3 = 6",
        "10 - 4 = 6",
        "15 / 3 = 5",
        "7 + 8 = 15",
        "9 * 9 = 81",
        "100 / 4 = 25",
        "3 + 5 = 8",
        "6 * 7 = 42",
        "20 - 12 = 8",
        # Algebra (10)
        "x^2 + y^2 = r^2",
        "f(x) = ax + b",
        "a^2 + b^2 = c^2 for right triangles",
        "log(a*b) = log(a) + log(b)",
        "(a + b)^2 = a^2 + 2ab + b^2",
        "a^2 - b^2 = (a+b)(a-b)",
        "quadratic formula: x = (-b +- sqrt(b^2-4ac)) / 2a",
        "slope of line: m = (y2-y1)/(x2-x1)",
        "distance formula: d = sqrt((x2-x1)^2 + (y2-y1)^2)",
        "midpoint: M = ((x1+x2)/2, (y1+y2)/2)",
        # Calculus (10)
        "integral of x dx = x^2/2 + C",
        "derivative of x^2 is 2x",
        "derivative of sin(x) is cos(x)",
        "derivative of e^x is e^x",
        "derivative of ln(x) is 1/x",
        "integral of 1/x dx = ln|x| + C",
        "integral of cos(x) dx = sin(x) + C",
        "integral of e^x dx = e^x + C",
        "limit of 1/x as x -> inf is 0",
        "Taylor series: f(x) = sum of f^(n)(a)/n! * (x-a)^n",
        # Trigonometry (8)
        "sin(0) = 0, cos(0) = 1",
        "sin(30) = 0.5, cos(60) = 0.5",
        "sin(90) = 1, cos(90) = 0",
        "tan(45) = 1",
        "sin^2(x) + cos^2(x) = 1",
        "sin(a+b) = sin(a)cos(b) + cos(a)sin(b)",
        "cos(a+b) = cos(a)cos(b) - sin(a)sin(b)",
        "law of cosines: c^2 = a^2 + b^2 - 2ab*cos(C)",
        # Famous formulas (8)
        "e^(i*pi) + 1 = 0",
        "sum of 1 to n is n*(n+1)/2",
        "The area of circle is pi * r^2",
        "The volume of sphere is 4/3 * pi * r^3",
        "The circumference of circle is 2 * pi * r",
        "F = ma (Newton's second law)",
        "E = mc^2 (Einstein's mass-energy equivalence)",
        "Pythagorean theorem: a^2 + b^2 = c^2",
        # Linear Algebra (6)
        "Matrix multiplication is associative",
        "The determinant of 2x2 matrix [[a,b],[c,d]] is ad - bc",
        "Identity matrix has 1s on diagonal and 0s elsewhere",
        "Transpose of matrix A is A^T where (A^T)_ij = A_ji",
        "The inverse of A satisfies A * A^(-1) = I",
        "Eigenvalue equation: Av = lambda*v",
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

    NUM_CYCLES = 8
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
