"""协作崩溃诊断——定位 B 输出乱码的根因。

假设：B 崩溃是因为混入 general 神经元（en vocab 空间），
     与 zh_1/2/3（zh vocab 空间）跨 vocab 错位融合。

验证三组：
  B1: zh_1/2/3 only（同 vocab，无 general 污染）→ 若输出变中文，假设成立
  B2: zh_3 single（最弱单干 baseline）
  B3: zh_1/2/3 + general（复现原 B 崩溃）
"""
import sys
sys.path.insert(0, "e:/taiji-neuron")

from taiji.loader import assemble_cortex

PROMPTS = [
    "你好，请介绍一下自己",
    "什么是人工智能？",
]

def run(cortex, label, active_nids, prompts):
    print(f"\n{'='*70}", flush=True)
    print(f"[{label}] active_nids = {active_nids}", flush=True)
    print(f"{'='*70}", flush=True)
    gk = dict(max_tokens=60, temperature=0.8, top_k=40,
              domain="zh", repetition_penalty=1.2)
    for p in prompts:
        try:
            out = cortex.generate(prompt=p, active_nids=active_nids, **gk)
        except Exception as e:
            out = f"[ERROR] {e}"
        print(f"\n  prompt: {p}", flush=True)
        print(f"  output: {out[:180] if out else '(empty)'}", flush=True)

def main():
    print("装配 Cortex...", flush=True)
    cortex, _, _ = assemble_cortex()
    print(f"  neurons: {list(cortex.neurons.keys())}", flush=True)

    # B1: 同 vocab 协作（去掉 general 污染源）
    run(cortex, "B1: zh_1/2/3 only (同vocab, 无general)",
        ["zh_1", "zh_2", "zh_3"], PROMPTS)

    # B2: 最弱单干 baseline
    run(cortex, "B2: zh_3 single (最弱单干)",
        ["zh_3", "general"], PROMPTS)

    # B3: 复现原 B 崩溃
    run(cortex, "B3: zh_1/2/3 + general (复现崩溃)",
        ["zh_1", "zh_2", "zh_3", "general"], PROMPTS)

    print(f"\n{'='*70}", flush=True)
    print("诊断判读：", flush=True)
    print("  - 若 B1 输出中文词组（非符号乱码）→ 确认 general 跨vocab污染是崩溃根因", flush=True)
    print("  - 若 B1 仍乱码 → 同vocab融合本身有问题（弱模型logits冲突）", flush=True)
    print("  - B2 vs B1：单干 vs 同域协作的对比", flush=True)
    print(f"{'='*70}", flush=True)

if __name__ == "__main__":
    main()
