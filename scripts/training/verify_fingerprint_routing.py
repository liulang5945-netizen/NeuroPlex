"""训练前后 fingerprint 区分度 + Level 1 vs Level 2 路由对比验证。

流程：
1. 装配 Cortex
2. 记录训练前 fingerprint 分数 + L1/L2 生成
3. 8 轮 feed+sleep 训练
4. 记录训练后 fingerprint 分数 + L1/L2 生成
5. 对比分析
"""
import sys
sys.path.insert(0, "e:/taiji-neuron")
import torch
from datetime import datetime

from taiji.loader import assemble_cortex
from taiji.life.feed_engine import get_feed_engine
from taiji.life.sleep_engine import get_sleep_engine, SleepReport

# ── 训练数据（精简，每域 40 条，共 160 条，8 轮 × 32 = 256 samples）──
TRAINING_DATA = {
    "zh": [
        "今天天气真好，适合出去散步。",
        "人工智能正在改变我们的生活方式。",
        "中国的历史文化源远流长。",
        "学习编程需要耐心和练习。",
        "春天来了，花儿都开了。",
        "我们应该保护环境，减少污染。",
        "这本书写得很好，值得一看。",
        "健康饮食对身体很重要。",
        "科技发展日新月异。",
        "教育是国家发展的根本。",
        "音乐能陶冶人的情操。",
        "旅行可以开阔眼界。",
        "网络安全越来越重要。",
        "量子计算是未来的方向。",
        "太极拳是一种很好的运动。",
        "茶文化在中国有悠久历史。",
        "城市化进程带来了许多挑战。",
        "可再生能源是未来的趋势。",
        "社交媒体改变了人们的交流方式。",
        "人工智能伦理问题需要重视。",
        "气候变化是全人类面临的挑战。",
        "传统文化的传承很重要。",
        "科技创新推动社会进步。",
        "阅读是获取知识的重要途径。",
        "运动有益于身心健康。",
        "互联网让世界变得更小。",
        "大数据时代隐私保护至关重要。",
        "远程办公成为新常态。",
        "人工智能在医疗领域应用广泛。",
        "生态文明建设功在当代利在千秋。",
        "科学精神需要从小培养。",
        "艺术创作需要灵感和技术。",
        "语言学习需要长期坚持。",
        "历史告诉我们和平来之不易。",
        "哲学思考帮助理解人生意义。",
        "经济学研究资源配置问题。",
        "心理学帮助人们认识自我。",
        "法律维护社会公平正义。",
        "医学进步延长了人类寿命。",
        "天文学探索宇宙奥秘。",
    ],
    "en": [
        "The weather is nice today, perfect for a walk.",
        "Artificial intelligence is changing our lifestyle.",
        "Technology develops rapidly in the modern era.",
        "Education is the foundation of national development.",
        "Music can cultivate one's mind and soul.",
        "Travel broadens one's horizons.",
        "Cybersecurity is becoming increasingly important.",
        "Quantum computing is the direction of the future.",
        "We should protect the environment and reduce pollution.",
        "Healthy diet is important for the body.",
        "This book is well written and worth reading.",
        "Climate change is a challenge for all humanity.",
        "Social media has changed how people communicate.",
        "Reading is an important way to acquire knowledge.",
        "Exercise is beneficial for physical and mental health.",
        "The internet has made the world smaller.",
        "Privacy protection is crucial in the big data era.",
        "Remote work has become the new normal.",
        "AI ethics issues need attention.",
        "Innovation drives social progress.",
        "Sustainable development is a global goal.",
        "Collaboration is key to success in science.",
        "Open source software empowers developers worldwide.",
        "Machine learning models require large datasets.",
        "The scientific method relies on hypothesis testing.",
        "Renewable energy is the trend of the future.",
        "Critical thinking helps evaluate information.",
        "Cultural exchange promotes mutual understanding.",
        "Biodiversity is essential for ecosystem stability.",
        "Space exploration pushes the boundaries of knowledge.",
        "Distributed systems require careful consistency design.",
        "Neural networks learn hierarchical representations.",
        "Statistics provides tools for understanding uncertainty.",
        "Algorithms are fundamental to computer science.",
        "Good documentation improves code maintainability.",
        "Version control is essential for team collaboration.",
        "Testing ensures software reliability and quality.",
        "Optimization algorithms find minima of functions.",
        "Graph theory studies networks and connections.",
        "Probability theory underpins statistical inference.",
    ],
    "code": [
        "def hello_world(): print('Hello, World!')",
        "import numpy as np; arr = np.array([1, 2, 3])",
        "class Dog: def bark(self): return 'Woof!'",
        "for i in range(10): print(i)",
        "if x > 0: return x else: return -x",
        "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
        "from typing import List; def foo(items: List[int]) -> int:",
        "async def fetch_data(url): response = await client.get(url)",
        "result = [x**2 for x in range(10)]",
        "with open('file.txt', 'r') as f: data = f.read()",
        "try: result = 10 / 0 except ZeroDivisionError: pass",
        "import torch; model = torch.nn.Linear(512, 10)",
        "def binary_search(arr, target): lo, hi = 0, len(arr) - 1",
        "class NeuralNetwork(nn.Module): def forward(self, x):",
        "const add = (a, b) => a + b;",
        "let nums = [1, 2, 3].map(x => x * 2);",
        "SELECT * FROM users WHERE age > 18;",
        "def quicksort(arr): if len(arr) <= 1: return arr",
        "import pandas as pd; df = pd.read_csv('data.csv')",
        "git commit -m 'fix: resolve memory leak issue'",
        "docker build -t myapp . && docker run -p 8080:8080 myapp",
        "def merge_sort(arr): mid = len(arr) // 2",
        "class Stack: def push(self, item): self.items.append(item)",
        "import json; data = json.loads('{\"key\": \"value\"}')",
        "def bfs(graph, start): queue = [start]; visited = set()",
        "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100));",
        "def gradient_descent(f, x0, lr=0.01, n_iter=100):",
        "import matplotlib.pyplot as plt; plt.plot(x, y)",
        "class Queue: def enqueue(self, item): self.items.append(item)",
        "def dfs(graph, node, visited=None): if visited is None: visited = set()",
        "npm install express && node server.js",
        "def fibonacci(n): a, b = 0, 1; for _ in range(n): a, b = b, a+b",
        "import requests; response = requests.get('https://api.example.com')",
        "class LinkedList: def __init__(self): self.head = None",
        "def dijkstra(graph, start): distances = {node: float('inf') for node in graph}",
        "kubectl apply -f deployment.yaml",
        "def train_epoch(model, dataloader, optimizer):",
        "import sqlite3; conn = sqlite3.connect('test.db')",
        "class TreeNode: def __init__(self, val): self.val = val; self.left = None",
        "def reverse_string(s): return s[::-1]",
    ],
    "math": [
        "The derivative of x^2 is 2x.",
        "The integral of 1/x is ln|x| + C.",
        "Pythagorean theorem: a^2 + b^2 = c^2.",
        "Euler's identity: e^(iπ) + 1 = 0.",
        "The sum of angles in a triangle is 180 degrees.",
        "Prime numbers are divisible only by 1 and themselves.",
        "The Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34.",
        "The quadratic formula: x = (-b ± √(b²-4ac)) / 2a.",
        "Matrix multiplication: (AB)_ij = Σ A_ik * B_kj.",
        "The fundamental theorem of calculus links differentiation and integration.",
        "sin²(θ) + cos²(θ) = 1.",
        "The natural logarithm ln(x) is the inverse of e^x.",
        "A vector space is a set closed under addition and scalar multiplication.",
        "The mean of a dataset is the sum divided by the count.",
        "Standard deviation measures the spread of data.",
        "The chain rule: d/dx f(g(x)) = f'(g(x)) * g'(x).",
        "Eigenvalues satisfy Av = λv for eigenvector v.",
        "The determinant of a matrix determines invertibility.",
        "Probability of A given B: P(A|B) = P(A∩B) / P(B).",
        "Expected value E[X] = Σ x * P(X=x).",
        "The binomial theorem: (a+b)^n = Σ C(n,k) a^(n-k) b^k.",
        "Taylor series: f(x) = Σ f^(n)(a)/n! * (x-a)^n.",
        "The Gaussian distribution: f(x) = (1/σ√2π) e^(-(x-μ)²/2σ²).",
        "Linear regression minimizes the sum of squared residuals.",
        "Gradient of f: ∇f = (∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ).",
        "The Riemann integral is the limit of Riemann sums.",
        "Fourier transform decomposes a function into frequencies.",
        "The p-value determines statistical significance.",
        "Bayes' theorem: P(A|B) = P(B|A)P(A) / P(B).",
        "The law of large numbers: sample mean converges to expected value.",
        "Markov chains have memoryless transitions: P(X_{n+1}|X_n) only.",
        "The central limit theorem: sample means approach normal distribution.",
        "L'Hôpital's rule for indeterminate forms 0/0 or ∞/∞.",
        "Dot product: a·b = |a||b|cos(θ).",
        "Cross product: a × b is perpendicular to both a and b.",
        "The Laplacian: ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z².",
        "Modular arithmetic: a ≡ b (mod n) means n divides (a-b).",
        "The pigeonhole principle: n items in m<n boxes implies collision.",
        "Proof by induction: base case + inductive step.",
    ],
}

# ── 测试 prompt ──
TEST_PROMPTS = [
    ("你好", "zh"),
    ("Hello", "en"),
    ("def foo", "code"),
    ("1+1=", "math"),
]


def compute_fingerprint_scores(cortex, prompt):
    """计算 prompt 对所有 neuron 的 fingerprint cosine 分数。"""
    general_ids = cortex._general_sp.encode(prompt)
    if not general_ids:
        general_ids = [0]
    ids_tensor = torch.tensor([general_ids], dtype=torch.long, device=cortex.device)
    prompt_emb = cortex._shared_embedding(ids_tensor)
    prompt_vec = prompt_emb.mean(dim=1).squeeze(0)
    prompt_norm = prompt_vec / (prompt_vec.norm() + 1e-8)

    scores = {}
    for nid, neuron in cortex.neurons.items():
        fp = getattr(neuron, "fingerprint", None)
        if fp is not None and fp.norm() > 1e-8 and fp.shape[0] == prompt_norm.shape[0]:
            sim = float(torch.dot(prompt_norm, fp / (fp.norm() + 1e-8)))
            scores[nid] = sim
    return scores


def fingerprint_spread(scores):
    """计算 fingerprint 分数的区分度（max - min）。"""
    if not scores:
        return 0.0
    return max(scores.values()) - min(scores.values())


def main():
    print("=" * 60)
    print("Fingerprint 路由对比验证")
    print("=" * 60)

    # Step 1: 装配 Cortex
    print("\n[Step 1] 装配 Cortex...")
    cortex, _, _ = assemble_cortex()
    total_samples = sum(len(v) for v in TRAINING_DATA.values())
    print(f"  Neurons: {list(cortex.neurons.keys())}")
    print(f"  训练数据: {total_samples} 条")

    # Step 2: 训练前 fingerprint 分数
    print("\n[Step 2] 训练前 fingerprint 分数...")
    pre_scores = {}
    pre_spreads = {}
    for prompt, domain in TEST_PROMPTS:
        scores = compute_fingerprint_scores(cortex, prompt)
        pre_scores[prompt] = scores
        spread = fingerprint_spread(scores)
        pre_spreads[prompt] = spread
        score_str = ", ".join(f"{n}={s:.4f}" for n, s in
                              sorted(scores.items(), key=lambda x: x[1], reverse=True))
        print(f"  [{domain}] '{prompt}': spread={spread:.4f} | {score_str}")

    # Step 3: 训练前 L1 vs L2 生成
    print("\n[Step 3] 训练前 L1 vs L2 生成...")
    pre_l1_gens = {}
    pre_l2_gens = {}
    for prompt, domain in TEST_PROMPTS:
        try:
            l1 = cortex.generate(prompt, max_tokens=20, domain=domain, routing_level=1)
            pre_l1_gens[prompt] = l1
            print(f"  [{domain}] '{prompt}' L1: '{l1[:50]}'")
        except Exception as e:
            pre_l1_gens[prompt] = ""
            print(f"  [{domain}] '{prompt}' L1 错误: {e}")
        try:
            l2 = cortex.generate(prompt, max_tokens=20, domain=domain, routing_level=2)
            pre_l2_gens[prompt] = l2
            print(f"  [{domain}] '{prompt}' L2: '{l2[:50]}'")
        except Exception as e:
            pre_l2_gens[prompt] = ""
            print(f"  [{domain}] '{prompt}' L2 错误: {e}")

    # Step 4: 8 轮训练
    print("\n[Step 4] 8 轮 feed+sleep 训练...")
    feed_engine = get_feed_engine()
    sleep_engine = get_sleep_engine()
    sleep_engine.cortex = cortex
    if sleep_engine._feed_engine is None:
        sleep_engine._feed_engine = feed_engine

    NUM_CYCLES = 8
    losses = []
    for cycle in range(NUM_CYCLES):
        for domain, texts in TRAINING_DATA.items():
            for text in texts:
                feed_engine.feed_text(text=text, source=f"cycle_{cycle}", domain=domain)

        report = SleepReport(timestamp=datetime.now().isoformat(), duration_seconds=0.0)
        sleep_engine._sleep_phase_model_training(report)

        loss = report.training_loss
        losses.append(loss)
        n = report.training_samples_used
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        print(f"  Cycle {cycle+1}/{NUM_CYCLES}: loss={loss_str}, samples={n}")

    # Step 5: 训练后 fingerprint 分数
    print("\n[Step 5] 训练后 fingerprint 分数...")
    post_scores = {}
    post_spreads = {}
    for prompt, domain in TEST_PROMPTS:
        scores = compute_fingerprint_scores(cortex, prompt)
        post_scores[prompt] = scores
        spread = fingerprint_spread(scores)
        post_spreads[prompt] = spread
        pre_spread = pre_spreads.get(prompt, 0.0)
        score_str = ", ".join(f"{n}={s:.4f}" for n, s in
                              sorted(scores.items(), key=lambda x: x[1], reverse=True))
        arrow = "↑" if spread > pre_spread else "↓" if spread < pre_spread else "="
        print(f"  [{domain}] '{prompt}': spread={spread:.4f} ({pre_spread:.4f}{arrow}) | {score_str}")

    # Step 6: 训练后 L1 vs L2 生成
    print("\n[Step 6] 训练后 L1 vs L2 生成...")
    post_l1_gens = {}
    post_l2_gens = {}
    for prompt, domain in TEST_PROMPTS:
        try:
            l1 = cortex.generate(prompt, max_tokens=20, domain=domain, routing_level=1)
            post_l1_gens[prompt] = l1
            print(f"  [{domain}] '{prompt}' L1: '{l1[:50]}'")
        except Exception as e:
            post_l1_gens[prompt] = ""
            print(f"  [{domain}] '{prompt}' L1 错误: {e}")
        try:
            l2 = cortex.generate(prompt, max_tokens=20, domain=domain, routing_level=2)
            post_l2_gens[prompt] = l2
            print(f"  [{domain}] '{prompt}' L2: '{l2[:50]}'")
        except Exception as e:
            post_l2_gens[prompt] = ""
            print(f"  [{domain}] '{prompt}' L2 错误: {e}")

    # Step 7: 综合分析
    print("\n" + "=" * 60)
    print("综合分析")
    print("=" * 60)
    valid_losses = [l for l in losses if l is not None]
    if len(valid_losses) >= 2:
        delta = valid_losses[0] - valid_losses[-1]
        pct = delta / valid_losses[0] * 100 if valid_losses[0] > 0 else 0
        print(f"  Loss: {valid_losses[0]:.4f} → {valid_losses[-1]:.4f} ({pct:+.1f}%)")

    print("\n  Fingerprint 区分度变化 (spread = max_score - min_score):")
    for prompt, domain in TEST_PROMPTS:
        pre = pre_spreads.get(prompt, 0.0)
        post = post_spreads.get(prompt, 0.0)
        arrow = "↑" if post > pre else "↓" if post < pre else "="
        print(f"    [{domain}] '{prompt}': {pre:.4f} → {post:.4f} {arrow}")

    print("\n  结论:")
    avg_pre = sum(pre_spreads.values()) / len(pre_spreads) if pre_spreads else 0
    avg_post = sum(post_spreads.values()) / len(post_spreads) if post_spreads else 0
    if avg_post > avg_pre:
        print(f"    ✅ Fingerprint 区分度提升 ({avg_pre:.4f} → {avg_post:.4f})")
        print("    训练使 field_write 权重分化，fingerprint 获得更好的域区分能力")
    else:
        print(f"    ⚠️ Fingerprint 区分度未提升 ({avg_pre:.4f} → {avg_post:.4f})")
        print("    需要更多训练轮次或更大学习率")


if __name__ == "__main__":
    main()
