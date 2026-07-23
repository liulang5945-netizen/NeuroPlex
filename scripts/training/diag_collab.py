"""协作机制对比——logit融合 vs 族长主导 vs 单干。

best模型 + weighted_logits修复后，logit融合协作(B1)失败（符号噪声）。
本测试对比：
  A:  zh单干(3MB) — 基准
  S:  zh_3纯单干(only, 无共振) — best弱模型独立能力
  L:  zh_1/2/3族长主导(场上下文, 不融合logits) — Plan C雏形
  B1: zh_1/2/3 logit融合(已知失败, 对照)
"""
import sys
sys.path.insert(0, "e:/taiji-neuron")

from taiji.loader import assemble_cortex

PROMPTS = [
    "你好，请介绍一下自己",
    "什么是人工智能？",
]

def run(cortex, label, active_nids, prompts, collab_mode="fusion"):
    print(f"\n{'='*70}", flush=True)
    print(f"[{label}] active_nids = {active_nids} collab_mode={collab_mode}", flush=True)
    print(f"{'='*70}", flush=True)
    gk = dict(max_tokens=60, temperature=0.8, top_k=40,
              domain="zh", repetition_penalty=1.2)
    for p in prompts:
        try:
            out = cortex.generate(prompt=p, active_nids=active_nids,
                                  collab_mode=collab_mode, **gk)
        except Exception as e:
            out = f"[ERROR] {e}"
        print(f"\n  prompt: {p}", flush=True)
        print(f"  output: {out[:180] if out else '(empty)'}", flush=True)

def main():
    print("装配 Cortex...", flush=True)
    cortex, _, _ = assemble_cortex()
    print(f"  neurons: {list(cortex.neurons.keys())}", flush=True)

    # A: zh 单干基准 (3MB, PPL=62.4)
    run(cortex, "A: zh 单干 (3MB基准)",
        ["zh", "general"], PROMPTS)

    # S: zh_3 纯单干 (best, 无共振无general) — 弱模型独立能力
    run(cortex, "S: zh_3 纯单干 (best, only)",
        ["zh_3"], PROMPTS)

    # L: 族长主导 (zh_1/2/3共振, 族长logits输出, 不融合) — Plan C雏形
    run(cortex, "L: zh_1/2/3 族长主导 (场上下文)",
        ["zh_1", "zh_2", "zh_3"], PROMPTS, collab_mode="leader")

    # B1: logit融合 (已知失败, 对照)
    run(cortex, "B1: zh_1/2/3 logit融合 (失败对照)",
        ["zh_1", "zh_2", "zh_3"], PROMPTS, collab_mode="fusion")

    print(f"\n{'='*70}", flush=True)
    print("判读（核心：L vs S）：", flush=True)
    print("  - L(族长+场上下文) vs S(纯单干)：场上下文协作是否提升？", flush=True)
    print("  - 若 L > S → 场上下文协作有效，族长模式成立 → 推进Plan C", flush=True)
    print("  - 若 L ≤ S → 场上下文也无帮助 → 协作范式需根本重设计", flush=True)
    print("  - B1 确认 logit融合失败（对照）", flush=True)
    print(f"{'='*70}", flush=True)

if __name__ == "__main__":
    main()
