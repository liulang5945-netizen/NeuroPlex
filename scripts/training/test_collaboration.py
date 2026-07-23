"""多同域神经元协作测试——验证"小神经元协作"核心假设。

实验对比：
  A. 单 neuron (zh, 3MB 训练): active_nids=["zh", "general"]
  B. 3 neuron 协作 (zh_1/zh_2/zh_3, 各 1MB): active_nids=["zh_1","zh_2","zh_3","general"]

如果 B 的生成质量优于 A（更连贯、更少重复），则证明：
  - 共振场 + MoCo logit 融合能涌现集体智能
  - 多个弱神经元协作 > 单个较强神经元
  - "小神经元协作匹配大模型" 理念成立
"""
import sys
sys.path.insert(0, "e:/taiji-neuron")

import torch
from taiji.loader import assemble_cortex

PROMPTS = [
    "你好，请介绍一下你自己",
    "什么是人工智能？",
    "深度学习在自然语言处理中的应用",
    "请解释神经网络的工作原理",
]

def main():
    print("=" * 70, flush=True)
    print("多同域神经元协作实验", flush=True)
    print("  A: 单 zh neuron (3MB) — baseline", flush=True)
    print("  B: 3 zh neurons (各1MB) 协作 — experimental", flush=True)
    print("=" * 70, flush=True)

    print("\n[1] 装配 Cortex...", flush=True)
    cortex, _, _ = assemble_cortex()
    all_nids = list(cortex.neurons.keys())
    print(f"  All neurons: {all_nids}", flush=True)

    # 检查 zh_1, zh_2, zh_3 是否存在
    zh_multi = [f"zh_{i}" for i in range(1, 4)]
    has_multi = all(nid in cortex.neurons for nid in zh_multi)
    has_single = "zh" in cortex.neurons
    print(f"  zh (single, 3MB): {'✓' if has_single else '✗'}", flush=True)
    print(f"  zh_1/2/3 (multi, 1MB each): {'✓' if has_multi else '✗'}", flush=True)

    if not has_multi:
        print("\n[!] zh_1/zh_2/zh_3 未找到。请先运行 train_multi_zh.py", flush=True)
        return

    gen_kwargs = dict(max_tokens=80, temperature=0.8, top_k=40,
                      domain="zh", repetition_penalty=1.2)

    # ── Test A: Single zh neuron ──
    print(f"\n{'='*70}", flush=True)
    print("[A] 单 zh neuron (3MB 训练)", flush=True)
    print(f"    active_nids = ['zh', 'general']", flush=True)
    print(f"{'='*70}", flush=True)
    for prompt in PROMPTS:
        try:
            out = cortex.generate(
                prompt=prompt, active_nids=["zh", "general"], **gen_kwargs,
            )
        except Exception as e:
            out = f"[ERROR] {e}"
        print(f"\n  prompt: {prompt}", flush=True)
        print(f"  output: {out[:200] if out else '(empty)'}", flush=True)

    # ── Test B: 3 zh neurons collaborating ──
    print(f"\n{'='*70}", flush=True)
    print("[B] 3 zh neurons 协作 (各 1MB 训练)", flush=True)
    print(f"    active_nids = ['zh_1', 'zh_2', 'zh_3', 'general']", flush=True)
    print(f"{'='*70}", flush=True)
    for prompt in PROMPTS:
        try:
            out = cortex.generate(
                prompt=prompt,
                active_nids=["zh_1", "zh_2", "zh_3", "general"],
                **gen_kwargs,
            )
        except Exception as e:
            out = f"[ERROR] {e}"
        print(f"\n  prompt: {prompt}", flush=True)
        print(f"  output: {out[:200] if out else '(empty)'}", flush=True)

    # ── Test C: 2 zh neurons (subset) ──
    if has_single and has_multi:
        print(f"\n{'='*70}", flush=True)
        print("[C] 单 zh + zh_1 混合 (3MB + 1MB)", flush=True)
        print(f"    active_nids = ['zh', 'zh_1', 'general']", flush=True)
        print(f"{'='*70}", flush=True)
        for prompt in PROMPTS[:2]:
            try:
                out = cortex.generate(
                    prompt=prompt,
                    active_nids=["zh", "zh_1", "general"],
                    **gen_kwargs,
                )
            except Exception as e:
                out = f"[ERROR] {e}"
            print(f"\n  prompt: {prompt}", flush=True)
            print(f"  output: {out[:200] if out else '(empty)'}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("判断标准：", flush=True)
    print("  - B 是否比 A 更连贯？（协作涌现）", flush=True)
    print("  - B 是否比 A 重复更少？（共振场消除噪声）", flush=True)
    print("  - C 是否介于 A 和 B 之间？（剂量效应）", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
