#!/usr/bin/env python3
"""zh 容量 vs leader 信号诊断（2026-08-11）。

diag_zh_leader 发现：leader（round1_scores 场共振分）偏好多为 zh_aug*（51M compact
dialogue），zh_std0（134M standard dialogue）0/5 当选且分数最低——场共振分衡量
"输入与场方向匹配"而非生成能力，134M 容量可能被浪费。

本脚本对比 3 个候选 neuron 的**单 neuron 生成质量**（绕过 leader 选择）：
- zh_std0_dialogue（134M standard，C24 dialogue 最大）
- zh_aug2_dialogue（51M compact，leader 常客）
- zh（51M general，C24 域目标 SFT）
判断容量提升是否真实带来生成质量改善 → 决定 leader 信号是否需改进/升级路径。

运行：python -u scripts/training/diag_zh_capacity.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.loader import assemble_cortex  # noqa: E402

DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"

CANDIDATES = [
    ("zh_std0_dialogue", "134M standard dialogue"),
    ("zh_aug2_dialogue", "51M compact dialogue"),
    ("zh", "51M general SFT"),
]

PROMPTS = [
    "请介绍什么是神经网络",
    "什么是注意力机制",
    "请解释梯度下降的原理",
    "你好，请介绍一下你自己",
]


def main():
    print("=" * 60, flush=True)
    print("zh 容量 vs leader 信号诊断（单 neuron 生成对比）", flush=True)
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
    print(f"  装配: {list(cortex.neurons.keys())}", flush=True)

    for nid, desc in CANDIDATES:
        print(f"\n{'=' * 50}\n[{nid}] {desc}", flush=True)
        lens = []
        for prompt in PROMPTS:
            try:
                out = cortex.generate(prompt, max_tokens=40, domain="zh",
                                      active_nids=[nid], collab_mode="continuous")
                lens.append(len(out))
                print(f"  {prompt[:12]}... → {out[:55]!r}", flush=True)
            except Exception as e:
                lens.append(0)
                print(f"  {prompt[:12]}... → ERR {e}", flush=True)
        non_empty = sum(1 for l in lens if l > 0)
        print(f"  非空 {non_empty}/{len(PROMPTS)}，平均长度 {sum(lens)/max(len(lens),1):.0f}", flush=True)


if __name__ == "__main__":
    main()
