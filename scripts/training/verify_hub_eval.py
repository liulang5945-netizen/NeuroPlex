#!/usr/bin/env python3
"""hub 阶段 4 跨域协作评估（缺口 L 成功标准，2026-08-14）。

在装配综合体（cortex 产品路径）上评估 hub 加入后的跨域协作效果：
- A1. 装配 10 neuron 综合体（含 hub），hub 规格正确（链路）
- A2. hub 锚点效应：hub 与各域 neuron field_vector cosine（统一空间，投影后）
      —— smoke 随机 hub 基线 ~0（未对齐）；训练后应提升（草案标准 3: >0.5）
- B1. 跨域能力涌现：中文指令 → 代码生成非空不退化（草案标准 2 链路）
- B2. 对照：无 hub 9 neuron 装配 + 生成不受影响（草案标准 1 不退化链路）
- C. 数值报告（供训练前后对比）

用法：
    python -u scripts/training/verify_hub_eval.py                      # smoke hub 基线
    python -u scripts/training/verify_hub_eval.py --hub-ckpt <路径>    # 指定 hub ckpt
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse  # noqa: E402
import random  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
from neuroplex.loader import assemble_cortex  # noqa: E402
from neuroplex.resonance.dialogue_format import build_dialogue_prompt  # noqa: E402

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
EXTRA_NEURONS_DIR = "data/foundation_v1_general"
HUB_CKPT = "data/hub_neuron/neuron_hub.pt"
# 锚点 cos 评估文本（zh + code 混合，覆盖 hub 跨域语义）
ANCHOR_TEXTS = [
    "写一个函数来计算给定数字的阶乘。",
    "def factorial(n):\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result",
    "编写一个将给定字符串转换为大写的函数。",
    "def convert_to_uppercase(s):\n    return s.upper()",
    "计算给定数字的平方根。",
    "import math\nnumber = 9\nsquare_root = math.sqrt(number)\nprint(square_root)",
]


def make_extra_with_hub(hub_ckpt: str) -> str:
    tmp = tempfile.mkdtemp(prefix="hub_eval_")
    for f in os.listdir(EXTRA_NEURONS_DIR):
        if f.endswith(".pt"):
            shutil.copy(os.path.join(EXTRA_NEURONS_DIR, f), os.path.join(tmp, f))
    shutil.copy(hub_ckpt, os.path.join(tmp, "neuron_hub.pt"))
    return tmp


def hub_anchor_cos(cortex) -> dict:
    """hub 与各域 neuron 的 field cosine（统一空间，投影后）。"""
    emb_table = cortex._shared_embedding
    ensemble = cortex.ensemble
    hub = cortex.neurons["hub"]
    general_sp = cortex._general_sp
    cos_by_nid: dict = {}
    with torch.no_grad():
        for text in ANCHOR_TEXTS:
            g_ids = general_sp.encode(text)[:64] or [0]
            emb = emb_table(torch.tensor([g_ids], dtype=torch.long))
            v_hub = hub.forward(emb, round_num=1)["field_vector"]
            if "hub" in ensemble._cross_spec_projectors:
                v_hub = ensemble._cross_spec_projectors["hub"](v_hub)
            for nid, n in cortex.neurons.items():
                if nid == "hub":
                    continue
                v = n.forward(emb, round_num=1)["field_vector"]
                if nid in ensemble._cross_spec_projectors:
                    v = ensemble._cross_spec_projectors[nid](v)
                c = float(F.cosine_similarity(v_hub, v).mean().item())
                cos_by_nid.setdefault(nid, []).append(c)
    return {nid: float(sum(v) / len(v)) for nid, v in cos_by_nid.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub-ckpt", default="data/hub_neuron/neuron_hub.pt",
                        help="hub ckpt 路径")
    parser.add_argument("--collab-name", default=COLLAB_NAME,
                        help="协作层 ckpt 文件名（含 hub 通道的训练产物）")
    args = parser.parse_args()
    hub_ckpt = args.hub_ckpt
    collab_name = args.collab_name

    t0 = time.time()
    print("=" * 60, flush=True)
    print("hub 阶段 4 跨域协作评估", flush=True)
    print(f"hub ckpt: {hub_ckpt}", flush=True)
    print("=" * 60, flush=True)

    # ── A. 含 hub 综合体 ──
    print("\n[A] 含 hub 综合体装配 + 锚点效应...", flush=True)
    extra = make_extra_with_hub(hub_ckpt)
    try:
        cortex, tokenizer, _ = assemble_cortex(
            neurons_dir="data/neurons",
            collab_name=collab_name,
            extra_neurons_dir=extra,
            device="cpu",
            max_rounds=3,
            wire_bio_modules=True,
            neuron_ids=DIALOGUE_IDS,
        )
        hn = cortex.neurons["hub"]
        check("A1. 10 neuron 综合体装配（hub 规格正确）",
              len(cortex.neurons) == 10 and hn.config.spec == "expert"
              and hn.lm_head.out_features == 256000,
              f"n={len(cortex.neurons)} hidden={hn.config.hidden_size}")
        cos_map = hub_anchor_cos(cortex)
        hub_cos = {nid: c for nid, c in cos_map.items()}
        valid = all(-1.0 <= c <= 1.0 for c in hub_cos.values())
        check("A2. hub 锚点 cosine 可计算（统一空间，有限）", valid,
              f"cos={ {k: '%.3f' % v for k, v in hub_cos.items()} }")
        print(f"    hub 锚点 cos 均值: "
              f"{sum(hub_cos.values()) / max(len(hub_cos), 1):.3f} "
              f"(smoke 基线 ~0；训练后应显著提升)", flush=True)

        # ── B1. 跨域能力涌现（中文指令 → 代码）──
        print("\n[B1] 跨域生成（中文指令 → 代码）...", flush=True)
        out = cortex.generate(
            build_dialogue_prompt("用 Python 写一个计算阶乘的函数。"),
            max_tokens=48, domain="zh", temperature=0.55,
        )
        check("B1. 跨域生成非空不退化",
              isinstance(out, str) and len(out.strip()) > 0
              and not cortex._is_degenerate_text(out),
              f"out={out[:48]!r}")
    except Exception as e:
        check("A1. 装配", False, f"err={e}")
        check("A2. 锚点 cos", False, f"err={e}")
        check("B1. 跨域生成", False, f"err={e}")

    # ── B2. 对照：无 hub 综合体 ──
    print("\n[B2] 对照（无 hub 9 neuron）...", flush=True)
    try:
        cortex_base, _, _ = assemble_cortex(
            neurons_dir="data/neurons",
            collab_name=COLLAB_NAME,
            extra_neurons_dir=EXTRA_NEURONS_DIR,
            device="cpu",
            max_rounds=3,
            wire_bio_modules=True,
            neuron_ids=DIALOGUE_IDS,
        )
        check("B2. 无 hub 装配不受影响（9 neuron）",
              len(cortex_base.neurons) == 9)
        out_base = cortex_base.generate(
            build_dialogue_prompt("介绍一下什么是机器学习。"),
            max_tokens=32, domain="zh", temperature=0.55,
        )
        check("B2b. 无 hub 基线生成非空（对照）",
              isinstance(out_base, str) and len(out_base.strip()) > 0,
              f"out={out_base[:36]!r}")
    except Exception as e:
        check("B2. 无 hub 装配", False, f"err={e}")
        check("B2b. 无 hub 基线生成", False, f"err={e}")

    shutil.rmtree(extra, ignore_errors=True)
    print(f"\n  总耗时: {time.time() - t0:.1f}s", flush=True)
    print("=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
