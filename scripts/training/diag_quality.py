"""快速诊断：最佳神经元 zh_2 用不同解码策略生成，判断是模型质量还是解码参数问题。

如果贪婪解码(t=0)输出比采样(t=0.8)更连贯 → 解码参数问题，调参即可
如果贪婪解码仍乱码 → 模型训练不足，需加数据/步数
"""
import sys, os
sys.path.insert(0, "e:/taiji-neuron")

from taiji.loader import assemble_cortex

PROMPTS = [
    "你好，请介绍一下自己",
    "什么是人工智能？",
    "今天天气真好，我想",
]

def main():
    cortex, _, _ = assemble_cortex()
    print(f"neurons: {list(cortex.neurons.keys())}", flush=True)

    # 测试 zh_2（输出最像中文的那个）和 zh_3
    for nid in ["zh_2", "zh_3", "zh"]:
        print("\n" + "=" * 70, flush=True)
        print(f"[{nid}] 三种解码策略对比", flush=True)
        print("=" * 70, flush=True)
        for prompt in PROMPTS:
            print(f"\n  prompt: {prompt}", flush=True)
            # A: 贪婪解码（t=0, 看模型真实最佳猜测）
            out_greedy = cortex.generate(
                prompt=prompt, active_nids=[nid], collab_mode="leader",
                max_tokens=40, temperature=0.01, top_k=1,
                domain="zh", repetition_penalty=1.0,  # 关闭重复惩罚，看原始分布
            )
            print(f"  greedy(t=0.01): {out_greedy[:150] if out_greedy else '(empty)'}", flush=True)
            # B: 低温采样
            out_low = cortex.generate(
                prompt=prompt, active_nids=[nid], collab_mode="leader",
                max_tokens=40, temperature=0.3, top_k=20,
                domain="zh", repetition_penalty=1.1,
            )
            print(f"  low-t(t=0.3):   {out_low[:150] if out_low else '(empty)'}", flush=True)

    # 汇总判读
    print("\n" + "=" * 70, flush=True)
    print("判读：", flush=True)
    print("  - greedy 输出有完整中文句子 → 模型OK，解码参数需调", flush=True)
    print("  - greedy 输出仍乱码/词组堆砌 → 模型训练不足，需加数据+步数", flush=True)

if __name__ == "__main__":
    main()
