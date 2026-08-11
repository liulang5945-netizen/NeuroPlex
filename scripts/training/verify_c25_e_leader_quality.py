#!/usr/bin/env python3
"""C25-E 增量四修复验证：continuous leader 用 round1_scores 质量信号。

复现路径：9 neuron 装配 + 训练格式 prompt + domain="zh"（此前 continuous
空输出 5/8 的场景）。验证：continuous 空输出消除、leader 有区分度。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.loader import assemble_cortex  # noqa: E402

passed = 0
failed = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name} {extra}", flush=True)
    else:
        failed += 1
        print(f"  [FAIL] {name} {extra}", flush=True)


DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"

QUESTIONS = [
    "你好", "你是谁？", "今天天气怎么样？", "1+1等于几？",
    "帮我写一首关于春天的诗", "推荐一本好书", "什么是幸福？", "怎么学好英语？",
]


def main():
    print("=" * 60, flush=True)
    print("C25-E 增量四：continuous leader 质量信号修复验证", flush=True)
    print("=" * 60, flush=True)
    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name=COLLAB_NAME,
        extra_neurons_dir=EXTRA_NEURONS_DIR,
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=DIALOGUE_IDS,
    )
    warm = ["你好", "帮我写代码", "解一道数学题", "What is this?", "写一首诗"]
    for _ in range(30):
        for wp in warm:
            cortex._executive_route(wp)

    for mode in ("continuous", "executive"):
        non_empty = 0
        print(f"\n[1] {mode} 挂载实测（8 问）", flush=True)
        for q in QUESTIONS:
            prompt = f"问：{q}\n答："
            text = cortex.generate(
                prompt=prompt, max_tokens=60, temperature=0.55, top_k=15,
                domain="zh", repetition_penalty=1.4, fusion_mode="soft",
                collab_mode=mode,
            )
            if text.strip():
                non_empty += 1
            print(f"  [{q}] {text[:60]!r} len={len(text)}", flush=True)
        check(f"{mode} 非空率", non_empty >= 7, f"{non_empty}/8")

    # 修复验证：continuous 域内 leader 用质量信号（round1_scores 有区分度）
    print("\n[2] continuous leader 质量信号", flush=True)
    emb_prompt = "问：你好\n答："
    general_ids = cortex._general_sp.encode(emb_prompt)
    ids_t = __import__("torch").tensor([general_ids], dtype=__import__("torch").long)
    neuron_embeddings = {}
    for nid, emb in cortex._neuron_shared_embeddings.items():
        neuron_embeddings[nid] = emb(ids_t)
    r = cortex.think(
        active_nids=[nid for nid in cortex.neurons if nid.startswith("zh_")],
        neuron_embeddings=neuron_embeddings,
        collab_mode="continuous",
    )
    r1s = r.get("round1_scores", {})
    fs = r.get("final_scores", {})
    print(f"  round1_scores: {r1s}", flush=True)
    print(f"  final_scores (时间平均激活): {fs}", flush=True)
    check("round1_scores 有区分度",
          len(r1s) >= 2 and max(r1s.values()) - min(r1s.values()) > 1e-4,
          f"max-min={max(r1s.values()) - min(r1s.values()):.4f}")

    print("=" * 60, flush=True)
    print(f"结果: {passed}/{passed + failed} PASS", flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
