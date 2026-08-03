"""API 等价日常对话实测（fastapi 不可用时的内部等价路径）。

模拟 api/routes_taiji.py 的 POST /api/taiji/cortex/chat 端点行为：
  assemble_cortex(neuron_ids=DEFAULT_NEURON_IDS) + cortex.generate(...)

用 API 默认采样参数（temperature=0.8, top_k=50），验证真实 API 体验。
"""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from taiji.core.model_loader import _resolve_neuron_ids  # noqa: E402
from taiji.loader import assemble_cortex  # noqa: E402

# 日常对话问题（贴近真实使用场景）
QUESTIONS = [
    "你好",
    "你是谁？",
    "今天天气怎么样？",
    "1+1等于几？",
    "帮我写一首关于春天的诗",
    "推荐一本好书",
    "什么是幸福？",
    "怎么学好英语？",
]

MAX_TOKENS = 60  # 保守（API 默认 256，长序列小模型易失控）
TEMPERATURE = 0.55  # 保守（API 默认 0.8）
TOP_K = 15  # 保守（API 默认 50）
REPETITION_PENALTY = 1.4  # 保守（API 默认 1.2）


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("态极综合体 API 等价日常对话实测", flush=True)
    print("参数: max_tokens=%d temperature=%s top_k=%s rep=%s"
          % (MAX_TOKENS, TEMPERATURE, TOP_K, REPETITION_PENALTY), flush=True)
    print("=" * 60, flush=True)

    print("[1] 装配 Cortex（对话综合体）...", flush=True)
    neuron_ids = _resolve_neuron_ids()
    print(f"  神经元: {neuron_ids}", flush=True)
    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=neuron_ids,
    )
    print(f"  装配完成: {list(cortex.neurons.keys())}", flush=True)
    print(f"  装配耗时: {time.time() - t0:.1f}s", flush=True)

    print("\n[2] 日常对话...", flush=True)
    for i, q in enumerate(QUESTIONS):
        t1 = time.time()
        try:
            # 训练格式 prompt（"问：xxx\n答："）与 finetune 数据一致
            prompt = f"问：{q}\n答："
            text = cortex.generate(
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                domain="zh",
                repetition_penalty=REPETITION_PENALTY,
                fusion_mode="soft",  # 与 eval 一致（per_position 异构 neuron 易出符号噪声）
            )
        except Exception as e:
            text = f"[生成失败: {e}]"
        dt = time.time() - t1
        print(f"\n  Q{i+1}: {q}", flush=True)
        print(f"  A: {text}", flush=True)
        print(f"  (耗时 {dt:.1f}s)", flush=True)

    print(f"\n  总计耗时: {time.time() - t0:.1f}s", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
