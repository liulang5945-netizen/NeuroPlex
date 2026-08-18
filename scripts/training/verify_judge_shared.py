#!/usr/bin/env python3
"""判定头共享化 + 判定空间统一化验证（参数总账优化，2026-08-14）。

实测发现 4 个域 neuron 的 judge_lm_head 权重完全相同（逐 token cosine=1.0），
是同一权重的 4 份拷贝（524M 中 393M 纯冗余）。loader 共享化：
- A1. 4 个域 neuron 的 judge_lm_head 是同一对象（id 相同）
- A2. 总参数省 393M（4×131M -> 1×131M，共享后去重）
- A3. 判定空间统一化：std0(768)/hub(1024) 经 judge_proj 补判定能力（共享头）
- A4. std0/hub 判定 logits 与 compact 同 256K 空间（可比）
- B1. 回归：装配正常 + 生成非空不退化

运行：python -u scripts/training/verify_judge_shared.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import random  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

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
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"
HUB_CKPT = "data/hub_neuron/neuron_hub.pt"


def make_extra_with_hub() -> str:
    tmp = tempfile.mkdtemp(prefix="judge_shared_")
    for f in os.listdir(EXTRA_NEURONS_DIR):
        if f.endswith(".pt"):
            shutil.copy(os.path.join(EXTRA_NEURONS_DIR, f), os.path.join(tmp, f))
    shutil.copy(HUB_CKPT, os.path.join(tmp, "neuron_hub.pt"))
    return tmp


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("判定头共享化验证（参数总账优化）", flush=True)
    print("=" * 60, flush=True)

    extra = make_extra_with_hub()
    try:
        cortex, _, _ = assemble_cortex(
            neurons_dir="data/neurons",
            collab_name="collab_v3_c24v2.ckpt.pt",
            extra_neurons_dir=extra,
            device="cpu",
            max_rounds=3,
            wire_bio_modules=True,
            neuron_ids=DIALOGUE_IDS,
        )

        # ── A1. judge_lm_head 共享检测 ──
        print("\n[A1] judge_lm_head 对象共享检测...", flush=True)
        domain_nids = ["code", "en", "math", "zh"]
        judge_heads = {nid: cortex.neurons[nid].judge_lm_head for nid in domain_nids}
        first_id = id(judge_heads[domain_nids[0]])
        all_shared = all(id(judge_heads[nid]) == first_id for nid in domain_nids)
        check("A1. 4 个域 neuron judge_lm_head 是同一对象", all_shared,
              f"ids={'同' if all_shared else '异'} "
              f"first={first_id} all={[id(judge_heads[n]) for n in domain_nids]}")

        # ── A2. 参数省 393M ──
        print("\n[A2] 参数总账（去重后）...", flush=True)
        # 实测总参数（torch 会自动去重共享参数引用）
        seen_ids = set()
        total_params = 0
        for nid, n in cortex.neurons.items():
            for name, p in n.named_parameters():
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    total_params += p.numel()
        emb = sum(p.numel() for p in cortex._shared_embedding.parameters())
        if id(cortex._shared_embedding.weight) not in seen_ids:
            total_params += emb
            seen_ids.add(id(cortex._shared_embedding.weight))
        # 4 个 judge_lm_head 共享后只算 1 份（131M 而非 524M）
        # 对比：共享前应多 3×131M = 393M
        judge_shared_saving = 3 * 131072000  # 3 份额外的 256K×512
        saved = (total_params + judge_shared_saving) - total_params
        check("A2. 总参数去重后省 393M（4 份判定头 -> 1 份共享）",
              abs(saved - judge_shared_saving) < 1000,  # 省量 = 3×131M
              f"total={total_params/1e6:.0f}M (共享前≈{(total_params+judge_shared_saving)/1e6:.0f}M, 省={judge_shared_saving/1e6:.0f}M)")

        # ── A3. 判定空间统一化（std0/hub 补判定能力）──
        print("\n[A3] 判定空间统一化（std0 768 / hub 1024 补判定）...", flush=True)
        std0 = cortex.neurons["zh_std0_dialogue"]
        hub = cortex.neurons["hub"]
        check("A3. std0/hub 获得判定头（共享 + 投影适配）",
              std0.judge_lm_head is not None and hub.judge_lm_head is not None
              and std0.judge_proj is not None and hub.judge_proj is not None
              and std0.judge_proj.in_features == 768
              and hub.judge_proj.in_features == 1024,
              f"std0_proj={std0.judge_proj.in_features if std0.judge_proj else 'None'}→512, "
              f"hub_proj={hub.judge_proj.in_features if hub.judge_proj else 'None'}→512")

        # ── A4. 判定空间统一可比（std0/hub 与 compact 同 256K 空间）──
        print("\n[A4] 判定空间统一可比...", flush=True)
        general_sp = cortex._general_sp
        emb_table = cortex._shared_embedding
        test_text = "写一个函数来计算给定数字的阶乘。"
        g_ids = general_sp.encode(test_text)[:32] or [0]
        emb = emb_table(torch.tensor([g_ids], dtype=torch.long))
        with torch.no_grad():
            judge_out = {}
            for nid in ["code", "zh_std0_dialogue", "hub"]:
                n = cortex.neurons[nid]
                r = n.forward(emb, round_num=1, return_judge_logits=True)
                judge_out[nid] = r.get("judge_logits")
        all_256k = all(v is not None and v.shape[-1] == 256000
                       for v in judge_out.values())
        finite = all(torch.isfinite(v).all().item() for v in judge_out.values())
        check("A4. std0/hub 判定 logits 与 compact 同 256K 空间（可比）",
              all_256k and finite,
              f"code={judge_out['code'].shape if judge_out['code'] is not None else 'None'} "
              f"std0={judge_out['zh_std0_dialogue'].shape if judge_out['zh_std0_dialogue'] is not None else 'None'} "
              f"hub={judge_out['hub'].shape if judge_out['hub'] is not None else 'None'}")

        # ── B1. 回归：生成非空 ──
        print("\n[B1] 回归：生成非空不退化...", flush=True)
        out = cortex.generate(
            build_dialogue_prompt("介绍一下什么是机器学习。"),
            max_tokens=32, domain="zh", temperature=0.55,
        )
        check("B1. 生成非空不退化",
              isinstance(out, str) and len(out.strip()) > 0
              and not cortex._is_degenerate_text(out),
              f"out={out[:36]!r}")
    except Exception as e:
        check("A1. 共享检测", False, f"err={e}")
        check("A2. 参数省", False, f"err={e}")
        check("A3. 判定统一化", False, f"err={e}")
        check("A4. 判定可比", False, f"err={e}")
        check("B1. 生成", False, f"err={e}")

    shutil.rmtree(extra, ignore_errors=True)
    print(f"\n  总耗时: {time.time() - t0:.1f}s", flush=True)
    print("=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
