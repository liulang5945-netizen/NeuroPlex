"""缩短版路由基准（8轮）——快速验证对比损失修复的端到端效果。

直接 print 输出（无管道缓冲），8轮训练 + L1/L2 路由对比。
"""
import sys
sys.path.insert(0, "e:/taiji-neuron")

import torch
from datetime import datetime
from taiji.loader import assemble_cortex
from taiji.life.feed_engine import get_feed_engine
from taiji.life.sleep_engine import get_sleep_engine, SleepReport

# 精简训练数据（每域 20 条，共 100 条，加速训练）
TRAINING_DATA = {
    "zh": ["今天天气真好适合散步", "人工智能改变生活方式", "中国历史文化源远流长",
           "学习编程需要耐心练习", "保护环境减少污染", "健康饮食对身体重要",
           "科技发展日新月异", "教育是国家发展根本", "音乐能陶冶情操", "旅行开阔眼界",
           "网络安全越来越重要", "量子计算是未来方向", "茶文化在中国悠久", "城市化带来挑战",
           "可再生能源是趋势", "社交媒体改变交流", "AI伦理需重视", "气候变化是挑战",
           "传统文化传承重要", "科技创新推动进步"],
    "en": ["The weather is nice today", "AI is changing lifestyle", "Technology develops rapidly",
           "Education is the foundation", "Music cultivates the mind", "Travel broadens horizons",
           "Cybersecurity is important", "Quantum computing future", "Protect the environment",
           "Healthy diet matters", "This book is worth reading", "Climate change challenge",
           "Social media changes communication", "Reading acquires knowledge", "Exercise good for health",
           "Internet made world smaller", "Privacy protection crucial", "Remote work new normal",
           "AI ethics needs attention", "Innovation drives progress"],
    "code": ["def hello(): print('hello')", "import numpy as np", "class Dog: def bark(self)",
             "for i in range(10): print(i)", "if x > 0: return x", "def fact(n): return n*fact(n-1)",
             "async def fetch(url): pass", "result = [x**2 for x in range(10)]",
             "with open('f.txt') as f: pass", "try: 1/0 except: pass",
             "import torch; m = torch.nn.Linear(512,10)", "def bs(arr,t): lo,hi=0,len(arr)-1",
             "const add = (a,b) => a+b", "let nums = [1,2,3].map(x=>x*2)",
             "SELECT * FROM users WHERE age>18", "def qsort(arr): return arr",
             "import pandas as pd", "git commit -m 'fix'", "docker build -t app .",
             "class Stack: def push(self,item): pass"],
    "math": ["integral of sin(x) dx", "derivative of x^2 is 2x", "sum of 1 to n is n(n+1)/2",
             "e^(i*pi) + 1 = 0", "limit of 1/x as x->inf is 0", "a^2 + b^2 = c^2",
             "f(x) = ax + b", "area of circle pi*r^2", "volume of sphere 4/3*pi*r^3",
             "log(xy) = log(x) + log(y)", "sin^2(x) + cos^2(x) = 1", "d/dx e^x = e^x",
             "integral of 1/x is ln|x|", "matrix multiplication AB", "probability P(A|B)",
             "mean = sum(x)/n", "variance = E[(X-mu)^2]", "normal distribution N(mu,sigma^2)",
             "gradient descent update", "Taylor series expansion"],
    "general": ["系统设计模式概述", "system design overview", "软件架构最佳实践",
                "best practices in engineering", "项目管理方法论", "project management methodology",
                "团队协作工具", "team collaboration tools", "产品需求文档", "product requirements doc",
                "敏捷开发流程", "agile development process", "代码审查标准", "code review standards",
                "持续集成部署", "CI/CD pipeline", "微服务架构设计", "microservices architecture",
                "数据驱动决策", "data driven decisions"],
}

ROUTING_TESTS = [
    ("你好世界", "zh"), ("今天天气不错", "zh"), ("机器学习很有趣", "zh"),
    ("hello world", "en"), ("the quick brown fox", "en"), ("neural network model", "en"),
    ("def foo(): return 42", "code"), ("import os; print(os.getcwd())", "code"),
    ("class Cat: def meow(self)", "code"),
    ("integral of x^2", "math"), ("a^2 + b^2 = c^2", "math"), ("sin^2 + cos^2 = 1", "math"),
    ("系统设计模式", "general"), ("best practices", "general"),
]


def measure_routing(cortex):
    c1 = c2 = 0
    l1_details = []
    for prompt, expected in ROUTING_TESTS:
        # L1 域路由
        l1_domain = cortex._infer_domain(prompt)
        l1_ok = (l1_domain == expected)
        c1 += l1_ok
        l1_details.append((prompt, expected, l1_domain, l1_ok))
        # L2 指纹路由
        general_ids = cortex._general_sp.encode(prompt) or [0]
        l2_active = cortex._fingerprint_route(general_ids, top_k=2)
        l2_domain = l2_active[0] if l2_active else "general"
        c2 += (l2_domain == expected)
    return c1, c2, len(ROUTING_TESTS), l1_details


def main():
    print("=" * 60, flush=True)
    print("缩短版路由基准（8轮）— 对比损失修复端到端验证", flush=True)
    print("=" * 60, flush=True)

    print("\n[1] 装配 Cortex...", flush=True)
    cortex, _, _ = assemble_cortex()
    print(f"  Neurons: {list(cortex.neurons.keys())}", flush=True)

    print(f"\n[2] 训练前路由准确性（{len(ROUTING_TESTS)} 条）...", flush=True)
    c1, c2, total, l1d = measure_routing(cortex)
    print(f"  L1 域路由: {c1}/{total} ({c1/total:.0%})", flush=True)
    print(f"  L2 指纹路由: {c2}/{total} ({c2/total:.0%})", flush=True)
    for prompt, expected, pred, ok in l1d:
        if not ok:
            print(f"    [L1✗] '{prompt[:25]}' expected={expected} got={pred}", flush=True)

    print(f"\n[3] 训练 8 轮（每轮 100 样本 + contrastive phase）...", flush=True)
    feed_engine = get_feed_engine()
    sleep_engine = get_sleep_engine()
    sleep_engine.cortex = cortex
    if sleep_engine._feed_engine is None:
        sleep_engine._feed_engine = feed_engine

    for cycle in range(8):
        for domain, texts in TRAINING_DATA.items():
            for text in texts:
                feed_engine.feed_text(text=text, source=f"c{cycle}", domain=domain)
        report = SleepReport(timestamp=datetime.now().isoformat(), duration_seconds=0.0)
        sleep_engine._sleep_phase_model_training(report)
        print(f"  Cycle {cycle+1}/8: loss={report.training_loss:.4f}", flush=True)

    print(f"\n[4] 训练后路由准确性...", flush=True)
    c1p, c2p, _, _ = measure_routing(cortex)
    print(f"  L1 域路由: {c1p}/{total} ({c1p/total:.0%})", flush=True)
    print(f"  L2 指纹路由: {c2p}/{total} ({c2p/total:.0%})", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"结果对比:", flush=True)
    print(f"  L1 域路由:  {c1/total:.0%} → {c1p/total:.0%}", flush=True)
    print(f"  L2 指纹路由: {c2/total:.0%} → {c2p/total:.0%}  (修复前24轮: 8%)", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
