"""C21 词库多词表架构验证（临时脚本，验证后清理）。

验证目标：
1. 多词表 decode（_generate_p7 按 leader 词表空间）下 executive 生成
2. C16 LoRA 是否扭曲 dialogue neuron 的 zh 能力（已由 loader 按 lm_head
   空间过滤解决——≠256K 头的 lora_state 不注入；原 --no-dialogue-lora
   清零逻辑会误伤 v3 微调自带的 LoRA，已废弃移除）
"""
import os
import sys

os.environ.setdefault("TAIJI_TEST_MODE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from taiji.loader import assemble_cortex

PROMPTS = [
    ("code", "Write a Python function to compute the Fibonacci sequence"),
    ("math", "If a train travels at 60 mph for 3 hours, how many miles does it travel?"),
    ("zh", "写一个 Python 函数计算斐波那契数列"),
    ("dialogue", "你好，请介绍一下你自己"),
    ("en", "What is the capital of France?"),
]

DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt20", default="data/neurons/collab_v3_c20.ckpt.pt")
    args = ap.parse_args()

    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name="collab_v3_c16.ckpt.pt",
        extra_neurons_dir="data/foundation_v1_general",
        neuron_ids=DIALOGUE_IDS,
    )
    print(f"[assemble_cortex] neurons: {list(cortex.neurons.keys())}")

    # 注入 C20 head（回合级判定校准）
    ck20 = torch.load(args.ckpt20, map_location="cpu", weights_only=False)
    hs = ck20.get("head_state", {})
    for nid, neuron in cortex.neurons.items():
        if nid in hs and getattr(neuron, "quality_head", None) is not None:
            neuron.quality_head.load_state_dict(hs[nid])
    print(f"  C20 head 注入 {len(hs)}")

    # 预热 EMA（多样文本）
    for k in range(30):
        cortex._executive_route(PROMPTS[k % len(PROMPTS)][1])

    print("\n=== 回合级判定 ===")
    for tag, prompt in PROMPTS:
        dom, conf, _ = cortex._executive_route(prompt)
        print(f"[{tag:<8}] → {dom}")

    print("\n=== executive 生成（40 token）===")
    for tag, prompt in PROMPTS:
        try:
            out = cortex.generate(
                prompt, max_tokens=40, temperature=0.9, top_k=50,
                collab_mode="executive",
            )
            print(f"\n── [{tag}] {prompt}\n  → {out}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n── [{tag}] ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
