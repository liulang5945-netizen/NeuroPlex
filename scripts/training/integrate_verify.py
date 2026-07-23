"""整合验证——把质量好的 neuron 整合，族长主导验证稳定性。

摸底结论：质量好的只有 zh_1/2/3（best 模型）；zh(末步)过拟合；en/code/math/general 崩溃。
本测试：
  S:  zh_3 纯单干（对照）
  L3: zh_1/2/3 族长主导（3 best，已验证 L>S）
  L4: zh+zh_1/2/3 族长主导（含过拟合zh，看族长能否避免被差neuron主导）
  4 个 prompt 验证稳定性
"""
import sys
sys.path.insert(0, "e:/taiji-neuron")

from taiji.loader import assemble_cortex

PROMPTS = [
    "你好，请介绍一下自己",
    "什么是人工智能？",
    "深度学习在自然语言处理中的应用",
    "请解释神经网络的工作原理",
]

def run(cortex, label, active_nids, prompts, collab_mode="leader"):
    print(f"\n{'='*70}", flush=True)
    print(f"[{label}] active_nids={active_nids} mode={collab_mode}", flush=True)
    print(f"{'='*70}", flush=True)
    gk = dict(max_tokens=70, temperature=0.8, top_k=40,
              domain="zh", repetition_penalty=1.2)
    for p in prompts:
        try:
            out = cortex.generate(prompt=p, active_nids=active_nids,
                                  collab_mode=collab_mode, **gk)
        except Exception as e:
            out = f"[ERROR] {e}"
        print(f"\n  prompt: {p}", flush=True)
        print(f"  output: {out[:200] if out else '(empty)'}", flush=True)

def main():
    print("装配 Cortex...", flush=True)
    cortex, _, _ = assemble_cortex()
    print(f"  neurons: {list(cortex.neurons.keys())}", flush=True)

    # S: zh_3 纯单干（对照）
    run(cortex, "S: zh_3 纯单干 (对照)",
        ["zh_3"], PROMPTS)

    # L3: zh_1/2/3 族长主导（3 best）
    run(cortex, "L3: zh_1/2/3 族长主导 (3 best)",
        ["zh_1", "zh_2", "zh_3"], PROMPTS)

    # L4: zh+zh_1/2/3 族长主导（含过拟合zh）
    run(cortex, "L4: zh+zh_1/2/3 族长主导 (含过拟合zh)",
        ["zh", "zh_1", "zh_2", "zh_3"], PROMPTS)

    print(f"\n{'='*70}", flush=True)
    print("判读：", flush=True)
    print("  - L3 vs S：族长+场上下文是否稳定优于单干（4 prompt 一致性）", flush=True)
    print("  - L4 vs L3：加入过拟合zh是否拖累？族长能否避免被差neuron主导？", flush=True)
    print("  - 若 L4≈L3 → 族长主导鲁棒（共振分能避开过拟合neuron）", flush=True)
    print("  - 若 L4<L3 → 过拟合neuron共振分高被选为族长，需改进族长选择", flush=True)
    print(f"{'='*70}", flush=True)

if __name__ == "__main__":
    main()
