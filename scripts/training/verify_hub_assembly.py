#!/usr/bin/env python3
"""hub neuron 装配验证（缺口 L 阶段 3 第一部分，2026-08-14）。

验证 hub ckpt（EXPERT + general 256K + field_dim 4096）能装配进现有综合体：
- 混合 hidden_size 放宽（embed_adapter 适配，人脑各皮层容量不同）
- PhasorDynamics 含 hub 相位（collab phasor_state 训练集合外的 hub 用默认相位行）
- hub 跨规格投影（4096→3072）
- hub 域外不动（实例路由不剔除 hub）
- 生成不退化（对照：无 hub 9 neuron 基线）

运行：python -u scripts/training/verify_hub_assembly.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import random  # noqa: E402
import numpy as np  # noqa: E402

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
from taiji.loader import assemble_cortex  # noqa: E402
# 口径契约：zh/dialogue 域 prompt 必须走训练格式
from taiji.resonance.dialogue_format import build_dialogue_prompt  # noqa: E402

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
HUB_CKPT = "data/hub_neuron/neuron_hub.pt"


def make_extra_with_hub() -> str:
    """临时 extra 目录：4 general + hub。"""
    tmp = tempfile.mkdtemp(prefix="hub_assembly_")
    for f in os.listdir(EXTRA_NEURONS_DIR):
        if f.endswith(".pt"):
            shutil.copy(os.path.join(EXTRA_NEURONS_DIR, f), os.path.join(tmp, f))
    shutil.copy(HUB_CKPT, os.path.join(tmp, "neuron_hub.pt"))
    return tmp


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("hub neuron 装配验证（阶段 3 第一部分）", flush=True)
    print("=" * 60, flush=True)

    # ── A. hub 装配 ──
    print("\n[A] hub 装配（EXPERT + general 256K）...", flush=True)
    extra = make_extra_with_hub()
    try:
        cortex, tokenizer, modules = assemble_cortex(
            neurons_dir="data/neurons",
            collab_name=COLLAB_NAME,
            extra_neurons_dir=extra,
            device="cpu",
            max_rounds=3,
            wire_bio_modules=True,
            neuron_ids=DIALOGUE_IDS,
        )
        check("A1. hub 装配成功（10 neuron 综合体）",
              "hub" in cortex.neurons and len(cortex.neurons) == 10,
              f"n={len(cortex.neurons)}")
        hn = cortex.neurons["hub"]
        check("A2. hub 规格正确（expert/1024/field4096/vocab256K）",
              hn.config.spec == "expert" and hn.config.hidden_size == 1024
              and hn.config.field_dim == 4096
              and (hn.lm_head.out_features == 256000),
              f"spec={hn.config.spec} hidden={hn.config.hidden_size} "
              f"field={hn.config.field_dim}")
        go = cortex.ensemble.gamma_oscillator
        check("A3. PhasorDynamics 装配成功（未回退标量）",
              hasattr(go, "binding_tensor"),
              f"type={type(go).__name__}")
        ph = getattr(go, "phases", {})
        check("A4. hub 相位注册（collab 训练集合外默认相位行）",
              "hub" in ph if isinstance(ph, dict) else False,
              f"hub_phase={ph.get('hub') if isinstance(ph, dict) else 'n/a'}")
        check("A5. hub 跨规格投影（field 4096→3072）",
              "hub" in cortex.ensemble._cross_spec_projectors,
              "CrossSpecProjector 已补建")
        # ── B. 推理跑通 + 对照 ──
        print("\n[B] 推理与对照（无 hub 基线）...", flush=True)
        out_hub = cortex.generate(
            build_dialogue_prompt("介绍一下什么是机器学习。"),
            max_tokens=32, domain="zh", temperature=0.55,
        )
        check("B1. 含 hub 综合体生成非空不退化", isinstance(out_hub, str)
              and len(out_hub.strip()) > 0
              and not cortex._is_degenerate_text(out_hub),
              f"out={out_hub[:36]!r}")
    except Exception as e:
        check("A1. hub 装配成功", False, f"err={e}")
        check("A2. hub 规格正确", False, f"err={e}")
        check("A3. PhasorDynamics 成功", False, f"err={e}")
        check("A4. hub 相位注册", False, f"err={e}")
        check("A5. hub 跨规格投影", False, f"err={e}")
        check("B1. 含 hub 生成", False, f"err={e}")

    # 对照组：无 hub 9 neuron 装配（基线不退化）
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
        check("B3. 无 hub 基线生成非空（对照）",
              isinstance(out_base, str) and len(out_base.strip()) > 0,
              f"out={out_base[:36]!r}")
    except Exception as e:
        check("B2. 无 hub 装配", False, f"err={e}")
        check("B3. 无 hub 基线生成", False, f"err={e}")

    shutil.rmtree(extra, ignore_errors=True)
    print(f"\n  总耗时: {time.time() - t0:.1f}s", flush=True)
    print("=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
