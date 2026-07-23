"""快速生成质量测试——直接看态极当前能生成什么。

用已有 checkpoint 装配 cortex，对几个典型 prompt 生成文本，
让用户直观判断"能否正常对话"。
"""
import sys
sys.path.insert(0, "e:/taiji-neuron")

import torch
from taiji.loader import assemble_cortex

PROMPTS = [
    ("你好，请介绍一下你自己", "zh"),
    ("什么是人工智能？", "zh"),
    ("hello, how are you", "en"),
    ("def fibonacci(n):", "code"),
]

def main():
    print("=" * 60, flush=True)
    print("态极当前生成质量实测", flush=True)
    print("=" * 60, flush=True)

    print("\n[1] 装配 Cortex（从已有 checkpoint）...", flush=True)
    cortex, _, _ = assemble_cortex()
    print(f"  Neurons: {list(cortex.neurons.keys())}", flush=True)

    print(f"\n[2] 生成测试（max_tokens=64, temperature=0.8）...", flush=True)
    for prompt, domain in PROMPTS:
        try:
            out = cortex.generate(
                prompt=prompt, max_tokens=64, temperature=0.8,
                top_k=40, domain=domain, repetition_penalty=1.2,
                routing_level=1,
            )
        except Exception as e:
            out = f"[ERROR] {e}"
        print(f"\n  [{domain}] prompt: {prompt}", flush=True)
        print(f"  -> output: {out[:200] if out else '(empty)'}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("判断标准：输出是否连贯、是否切题、是否可读", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
