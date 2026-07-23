"""路由准确性验证：训练前后 L1/L2/L3 路由正确率对比"""
import sys
sys.path.insert(0, "e:/taiji-neuron")
import torch
from datetime import datetime

from taiji.loader import assemble_cortex
from taiji.life.feed_engine import get_feed_engine
from taiji.life.sleep_engine import get_sleep_engine, SleepReport

# ── 精简训练数据（每域 50 条，共 200 条）──
TRAINING_DATA = {
    "zh": [
        "今天天气真好，适合出去散步。", "人工智能正在改变我们的生活方式。",
        "中国的历史文化源远流长。", "学习编程需要耐心和练习。",
        "我们应该保护环境，减少污染。", "健康饮食对身体很重要。",
        "科技发展日新月异。", "教育是国家发展的根本。",
        "音乐能陶冶人的情操。", "旅行可以开阔眼界。",
        "网络安全越来越重要。", "量子计算是未来的方向。",
        "茶文化在中国有悠久历史。", "城市化进程带来了许多挑战。",
        "可再生能源是未来的趋势。", "社交媒体改变了人们的交流方式。",
        "人工智能伦理问题需要重视。", "气候变化是全人类面临的挑战。",
        "传统文化的传承很重要。", "科技创新推动社会进步。",
        "阅读是获取知识的重要途径。", "运动有益于身心健康。",
        "互联网让世界变得更小。", "大数据时代隐私保护至关重要。",
        "远程办公成为新常态。", "人工智能在医疗领域应用广泛。",
        "生态文明建设功在当代。", "科学精神需要从小培养。",
        "艺术创作需要灵感和技术。", "语言学习需要长期坚持。",
        "历史告诉我们和平来之不易。", "哲学思考帮助理解人生意义。",
        "经济学研究资源配置问题。", "心理学帮助人们认识自我。",
        "法律维护社会公平正义。", "医学进步延长了人类寿命。",
        "天文学探索宇宙奥秘。", "可持续发展是长期目标。",
        "文化多样性应该被保护。", "团队合作是成功的关键。",
        "创新思维需要跨界交流。", "批判性思维是重要能力。",
        "信息时代知识更新很快。", "终身学习是必然趋势。",
        "数字化正在重塑各行各业。", "生物多样性保护刻不容缓。",
        "食品安全事关民生。", "绿色能源是未来的方向。",
        "城市化带来机遇和挑战。", "科普教育需要加强。",
    ],
    "en": [
        "The weather is nice today for a walk.", "AI is changing our lifestyle.",
        "Technology develops rapidly.", "Education is the foundation.",
        "Music can cultivate the mind.", "Travel broadens horizons.",
        "Cybersecurity is important.", "Quantum computing is the future.",
        "Protect the environment.", "Healthy diet for the body.",
        "This book is worth reading.", "Climate change is a challenge.",
        "Social media changes communication.", "Reading acquires knowledge.",
        "Exercise is good for health.", "Internet made the world smaller.",
        "Privacy protection is crucial.", "Remote work is the new normal.",
        "AI ethics needs attention.", "Innovation drives progress.",
        "Sustainable development.", "Collaboration is key to success.",
        "Open source empowers developers.", "ML models need large datasets.",
        "The scientific method requires testing.", "Renewable energy is the trend.",
        "Critical thinking is essential.", "Cultural exchange promotes understanding.",
        "Biodiversity for ecosystem stability.", "Space exploration extends knowledge.",
        "Distributed systems need consistency.", "Neural networks learn hierarchies.",
        "Statistics for uncertainty.", "Algorithms are fundamental to CS.",
        "Good documentation improves maintainability.", "Version control for collaboration.",
        "Testing ensures software quality.", "Optimization finds function minima.",
        "Graph theory studies networks.", "Probability underpins inference.",
        "The sun rises in the east.", "Water is essential for life.",
        "Mathematics is the language of nature.", "Physics explains the universe.",
        "Chemistry studies matter.", "Biology examines living organisms.",
        "History teaches us lessons.", "Geography maps our world.",
        "Literature reflects society.", "Philosophy questions existence.",
        "Art expresses human emotion.", "Democracy values all voices.",
        "Freedom is a fundamental right.", "Knowledge is power.",
    ],
    "code": [
        "def hello(): print('hello')", "import numpy as np; x = np.array([1,2])",
        "class Dog: def bark(self): return 'woof'", "for i in range(10): print(i)",
        "if x > 0: return x else: return -x", "def fact(n): return 1 if n<=1 else n*fact(n-1)",
        "async def fetch(url): return await client.get(url)", "result = [x**2 for x in range(10)]",
        "with open('f.txt') as f: data = f.read()", "try: 1/0 except: pass",
        "import torch; m = torch.nn.Linear(512,10)", "def bs(arr,target): lo,hi=0,len(arr)-1",
        "const add = (a,b) => a+b;", "let nums = [1,2,3].map(x=>x*2);",
        "SELECT * FROM users WHERE age>18;", "def qsort(arr): return arr if len(arr)<=1 else qsort([x for x in arr[1:] if x<=arr[0]])+[arr[0]]+qsort([x for x in arr[1:] if x>arr[0]])",
        "import pandas as pd; df=pd.read_csv('d.csv')", "git commit -m 'fix: memory leak'",
        "docker build -t app . && docker run -p 8080:8080 app", "def ms(arr): m=len(arr)//2; return",
        "class Stack: def push(self,item): self.items.append(item)", "import json; d=json.loads('{}')",
        "def bfs(g,start): q=[start]; visited=set()", "CREATE TABLE users (id INT PRIMARY KEY);",
        "def gd(f,x0,lr=0.01,n=100):", "import matplotlib.pyplot as plt; plt.plot(x,y)",
        "class Queue: def enqueue(self,item): self.items.append(item)", "def dfs(g,node,visited=None):",
        "npm install express && node server.js", "def fib(n): a,b=0,1; exec('a,b=b,a+b;'*n); return a",
        "import requests; r=requests.get('https://api.example.com')", "class LinkedList: def __init__(self): self.head=None",
        "def dijkstra(g,start): dist={n:float('inf') for n in g}", "kubectl apply -f deploy.yaml",
        "def train_epoch(m,dl,opt):", "import sqlite3; conn=sqlite3.connect('t.db')",
        "class TreeNode: def __init__(self,v): self.val=v", "def rev(s): return s[::-1]",
        "const express = require('express'); const app = express();", "package main; import 'fmt'; func main() { fmt.Println('hi') }",
        "#include <stdio.h>; int main() { printf('hello'); return 0; }", "import React from 'react'; const App = () => <div>hello</div>;",
        "def decorator(func): def wrapper(*args): return func(*args); return wrapper;", "SELECT COUNT(*) FROM orders GROUP BY customer_id;",
        "def singleton(cls): instances={}; def get(*args): if cls not in instances: instances[cls]=cls(*args); return instances[cls]; return get;",
        "class BinaryTree: def __init__(self): self.root=None;", "import asyncio; async def main(): await asyncio.sleep(1)",
        "class LRUCache: def __init__(self,capacity): self.capacity=capacity; self.cache=OrderedDict();", "def topo_sort(graph): indegree={n:0 for n in graph};",
        "#include <vector>; int binary_search(vector<int> arr, int target) { int lo=0,hi=arr.size()-1; while(lo<=hi){int mid=lo+(hi-lo)/2; if(arr[mid]==target) return mid; else if(arr[mid]<target) lo=mid+1; else hi=mid-1;} return -1; }",
        "const mongo = require('mongodb'); const client = new mongo.MongoClient(url);", "pip install torch transformers datasets",
    ],
    "math": [
        "The derivative of x^2 is 2x.", "∫ 1/x dx = ln|x| + C.",
        "a^2 + b^2 = c^2.", "e^(iπ) + 1 = 0.",
        "Sum of angles = 180°.", "Primes: 2,3,5,7,11,13.",
        "Fibonacci: 0,1,1,2,3,5,8.", "x = (-b ± √(b²-4ac))/2a.",
        "(AB)_ij = Σ A_ik B_kj.", "Fundamental theorem of calculus.",
        "sin²θ + cos²θ = 1.", "ln(x) inverse of e^x.",
        "Vector space: closed under + and ·.", "Mean = sum/count.",
        "Standard deviation: spread measure.", "Chain rule: d/dx f(g(x)).",
        "Eigen: Av = λv.", "Determinant: invertibility.",
        "P(A|B) = P(A∩B)/P(B).", "E[X] = Σ x·P(X=x).",
        "(a+b)^n = Σ C(n,k)a^(n-k)b^k.", "Taylor: f(x) = Σ f^(n)(a)/n! · (x-a)^n.",
        "Gaussian: N(μ,σ²) f(x)=1/(σ√2π) e^(-(x-μ)²/2σ²).", "Linear regression: min Σ(y-ŷ)².",
        "∇f = (∂f/∂x₁, ∂f/∂x₂, ...).", "Riemann integral: limit of sums.",
        "Fourier: frequency decomposition.", "p-value: significance test.",
        "Bayes: P(A|B)=P(B|A)P(A)/P(B).", "Law of large numbers: x̄→μ.",
        "Markov: memoryless transition.", "CLT: sample mean ~ N(μ,σ²/n).",
        "L'Hôpital for 0/0 or ∞/∞.", "a·b = |a||b|cosθ.",
        "a×b ⊥ a and b.", "∇²f = Σ ∂²f/∂xᵢ².",
        "a ≡ b (mod n) iff n|(a-b).", "Pigeonhole: n items, m<n boxes.",
        "Proof by induction: base + step.", "Proof by contradiction: assume ¬P.",
        "Laplace: ∇²f = 0.", "Gradient descent: x_{t+1}=x_t-α∇f(x_t).",
        "Lagrange multiplier: ∇f=λ∇g.", "Convex: f''(x) ≥ 0.",
        "Matrix rank: dim of column space.", "Orthogonal: A^T = A^{-1}.",
        "Positive definite: x^T A x > 0.", "Singular value: A = UΣV^T.",
        "Central limit theorem explanation.", "Law of total probability.",
        "Maximum likelihood estimation.", "Hypothesis testing: H0 vs H1.",
    ],
}

# ── 路由准确性测试 ──
ROUTING_TESTS = [
    ("你好，今天天气如何", "zh"),
    ("Hello how are you today", "en"),
    ("def merge_sort(arr): return sorted(arr)", "code"),
    ("The derivative of sin(x) is cos(x)", "math"),
    ("import torch; model = torch.nn.Linear(512, 256)", "code"),
    ("人工智能技术的快速发展", "zh"),
    ("Calculate the integral of x^2 from 0 to 1", "math"),
    ("Create a REST API endpoint", "en"),
    ("量子计算的基本原理是什么", "zh"),
    ("Write a function to reverse a string", "code"),
    ("The chain rule: d/dx f(g(x)) = f'(g(x))g'(x)", "math"),
    ("The future of artificial intelligence", "en"),
]

# ── 生成质量测试 ──
GEN_TEST_PROMPTS = [
    ("今天天气", "zh"),
    ("人工智能", "zh"),
    ("def hello", "code"),
    ("1+1=", "math"),
    ("The weather", "en"),
]

from collections import Counter

def compute_diversity(text):
    """Unique chars ratio"""
    if not text:
        return 0.0
    return len(set(text)) / len(text)

def compute_repetition(text):
    """Most frequent char ratio"""
    if not text:
        return 0.0
    c = Counter(text)
    return c.most_common(1)[0][1] / len(text)

def measure_gen_quality(cortex):
    """返回 {prompt: (text, diversity, repetition)}"""
    results = {}
    for prompt, domain in GEN_TEST_PROMPTS:
        try:
            gen = cortex.generate(prompt, max_tokens=30, domain=domain, temperature=0.8, routing_level=1)
            div = compute_diversity(gen)
            rep = compute_repetition(gen)
            results[prompt] = (gen, div, rep)
        except Exception as e:
            results[prompt] = ("", 0.0, 1.0)
    return results


def measure_routing_accuracy(cortex):
    """返回 (L1正确数, L2正确数, 总数, L1详情, L2详情)"""
    correct_l1 = 0
    correct_l2 = 0
    total = len(ROUTING_TESTS)
    l1_details = []
    l2_details = []

    for prompt, expected in ROUTING_TESTS:
        general_ids = cortex._general_sp.encode(prompt)
        if not general_ids:
            general_ids = [0]

        l1_domain = cortex._infer_domain(prompt)
        l2_nids = cortex._fingerprint_route(general_ids, top_k=2)
        non_general = [n for n in l2_nids if n != "general"]
        l2_top = non_general[0] if non_general else "unknown"

        if l1_domain == expected:
            correct_l1 += 1
        if l2_top == expected:
            correct_l2 += 1

        l1_details.append((prompt, expected, l1_domain, l1_domain == expected))
        l2_details.append((prompt, expected, l2_top, l2_top == expected))

    return correct_l1, correct_l2, total, l1_details, l2_details


def main():
    print("=" * 60)
    print("路由准确性验证 — 训练前 L1/L2 对比")
    print("=" * 60)

    # Step 1: 装配
    print("\n[Step 1] 装配 Cortex...")
    cortex, _, _ = assemble_cortex()
    print(f"  Neurons: {list(cortex.neurons.keys())}")

    # Step 2: 训练前路由准确性
    print(f"\n[Step 2] 训练前路由准确性（{len(ROUTING_TESTS)} 条测试）...")
    c1, c2, total, l1d, l2d = measure_routing_accuracy(cortex)
    print(f"  L1 域路由: {c1}/{total} ({c1/total:.0%})")
    print(f"  L2 共振路由: {c2}/{total} ({c2/total:.0%})")
    for prompt, expected, pred, ok in l1d:
        s = "✓" if ok else "✗"
        if not ok:
            print(f"    [{expected}] '{prompt[:30]}' L1:{pred} {s}")

    # Step 2.5: 训练前生成质量
    print(f"\n[Step 2.5] 训练前生成质量...")
    pre_gen = measure_gen_quality(cortex)
    for prompt, domain in GEN_TEST_PROMPTS:
        text, div, rep = pre_gen[prompt]
        print(f"  [{domain}] '{prompt}' → '{text[:40]}' (div={div:.2f}, rep={rep:.2f})")

    # Step 3: 训练
    CYCLES = 12
    print(f"\n[Step 3] 训练 {CYCLES} 轮...")
    feed_engine = get_feed_engine()
    sleep_engine = get_sleep_engine()
    sleep_engine.cortex = cortex
    if sleep_engine._feed_engine is None:
        sleep_engine._feed_engine = feed_engine

    losses = []
    for cycle in range(CYCLES):
        for domain, texts in TRAINING_DATA.items():
            for text in texts:
                feed_engine.feed_text(text=text, source=f"c{cycle}", domain=domain)
        report = SleepReport(timestamp=datetime.now().isoformat(), duration_seconds=0.0)
        sleep_engine._sleep_phase_model_training(report)
        loss = report.training_loss
        losses.append(loss)
        if (cycle + 1) % 3 == 0 or cycle == 0:
            print(f"  Cycle {cycle+1}/{CYCLES}: loss={loss:.4f}, samples={report.training_samples_used}")

    # Step 4: 训练后路由准确性
    print(f"\n[Step 4] 训练后路由准确性...")
    c1_post, c2_post, _, _, _ = measure_routing_accuracy(cortex)
    delta_l1 = c1_post - c1
    delta_l2 = c2_post - c2
    print(f"  L1 域路由: {c1}→{c1_post}/{total} ({c1_post/total:.0%}, {delta_l1:+d})")
    print(f"  L2 共振路由: {c2}→{c2_post}/{total} ({c2_post/total:.0%}, {delta_l2:+d})")

    # Step 4.5: 训练后生成质量
    print(f"\n[Step 4.5] 训练后生成质量...")
    post_gen = measure_gen_quality(cortex)
    div_improved = 0
    rep_improved = 0
    for prompt, domain in GEN_TEST_PROMPTS:
        text, div, rep = post_gen[prompt]
        pre_text, pre_div, pre_rep = pre_gen[prompt]
        div_arrow = "↑" if div > pre_div else "↓"
        rep_arrow = "↓" if rep < pre_rep else "↑"
        if div > pre_div:
            div_improved += 1
        if rep < pre_rep:
            rep_improved += 1
        print(f"  [{domain}] '{prompt}': div {pre_div:.2f}→{div:.2f}{div_arrow} | rep {pre_rep:.2f}→{rep:.2f}{rep_arrow} | '{text[:30]}'")

    # Step 5: Loss 趋势
    print("\n[Step 5] Loss 趋势...")
    valid = [l for l in losses if l is not None]
    if len(valid) >= 2:
        d = valid[0] - valid[-1]
        pct = d/valid[0]*100 if valid[0] > 0 else 0
        print(f"  {valid[0]:.4f} → {valid[-1]:.4f} ({pct:+.1f}%)")
        for i in range(0, len(valid), 3):
            print(f"    C{i+1}: {valid[i]:.4f}")

    # Step 6: 结论
    n_gen = len(GEN_TEST_PROMPTS)
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    print(f"  训练 {CYCLES} 轮后：")
    print(f"  - Loss: {valid[0]:.4f} → {valid[-1]:.4f}")
    print(f"  - L1 路由: {c1}→{c1_post}/{total} ({c1_post/total:.0%})")
    print(f"  - L2 路由: {c2}→{c2_post}/{total} ({c2_post/total:.0%})")
    print(f"  - 多样性提升: {div_improved}/{n_gen}")
    print(f"  - 重复度降低: {rep_improved}/{n_gen}")
    if c1_post > c2_post:
        print(f"  - 建议：当前训练阶段使用 L1 域路由（{c1_post/total:.0%} vs L2 {c2_post/total:.0%}）")
    else:
        print(f"  - 建议：L2 共振路由已可用")


if __name__ == "__main__":
    main()
