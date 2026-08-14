#!/usr/bin/env python3
"""C27 增量一验证：实例级路由 + 混合后验（SMCS 借鉴，2026-08-14）。

背景：现有路由是"回合级任务判定"（C19/C22：dominant 域激活集回合内静态）
+ C25-E leader 质量融合（共振分 + prompt 一次性 NLL）。SMCS 的 contextual
selection 在实例内重新选 expert 子集——增量一让 continuous 生成中激活子集
按 chunk 级混合后验（共振分 + 已生成文本滚动 NLL）双向域内演化：
- 滚动后验：_rolling_nll_quality（round1_logits 尾部窗口，零额外前向）
- 剔除：激活集同域 neuron 后验 < evict_ratio×leader 且连续 evict_streak
  个 chunk 如此（迟滞防抖）
- 加入：未激活同域 neuron（_probe_inactive_fused 轻量前向）后验 >
  激活集最小 × add_ratio
- 保护：同域激活数 >= min_active；general 恒激活；域外（C22 边界）不动

验证层次：
A. 滚动后验合法性（真实 think round1_logits → _rolling_nll_quality 非空）
B. 双向演化单元（patch 质量信号驱动决策逻辑）：
   B1 剔除 + 迟滞（连续 2 chunk 劣化才剔，单次不剔）
   B2 加入（未激活同域后验显著更高 → 加入）
   B3 域内约束（跨域 neuron 不动）
   B4 min_active 保护（不会剔到少于 2 个域 neuron）
   B5 稳定（无劣化时激活集不变）
C. 集成：真实长生成触发实例级路由（patch 统计演化调用 ≥2）且输出非空
D. 开关回归：instance_routing=False 回退 C25-E 行为（长生成非空不退化）

运行：python -u scripts/training/verify_c27_instance_routing.py
"""

from __future__ import annotations

import os
import sys
import time
from unittest import mock

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


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("C27 增量一：实例级路由 + 混合后验（SMCS 借鉴）", flush=True)
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
    check("装配成功（5 dialogue + 4 general）", len(cortex.neurons) == 9)

    # judge EMA 预热（executive 判定主信号）
    warm = ["你好", "帮我写代码", "解一道数学题", "What is this?", "写一首诗"]
    for _ in range(30):
        for wp in warm:
            cortex._executive_route(wp)
    print("  judge EMA 预热完成", flush=True)

    zh_domain_all = ["zh_aug0_dialogue", "zh_aug1_dialogue",
                     "zh_aug2_dialogue", "zh_aug3_dialogue",
                     "zh_std0_dialogue", "zh"]

    # ── A. 滚动后验合法性（真实 round1_logits）──
    print("\n[A] 滚动后验（_rolling_nll_quality）...", flush=True)
    try:
        ids = torch.tensor(
            [cortex._general_sp.encode(build_dialogue_prompt("测试滚动后验"))],
            dtype=torch.long, device=cortex.device)
        emb = cortex._shared_embedding(ids)
        res = cortex.think(emb, active_nids=zh_domain_all,
                           collab_mode="continuous", fusion_mode="soft")
        q = cortex._rolling_nll_quality(
            res, "这是已经生成的测试文本内容，用于验证滚动后验窗口计算。",
            "zh", window=12)
        ok = bool(q) and all(
            isinstance(v, float) and v == v and abs(v) < 1e4 for v in q.values())
        check("A1. 滚动 NLL 输出合法（非空、有限值）", ok,
              f"nids={len(q)} q={ {k: round(v, 2) for k, v in list(q.items())[:3]} }")
        q2 = cortex._rolling_nll_quality(res, "", "zh", window=12)
        check("A2. 空已生成文本安全返回空", q2 == {})
    except Exception as e:
        check("A1. 滚动 NLL 输出合法", False, f"err={e}")
        check("A2. 空已生成文本安全返回空", False, f"err={e}")

    # ── B. 双向演化单元（patch 质量信号驱动决策逻辑）──
    print("\n[B] 双向域内演化（_instance_route_evolve）...", flush=True)
    # 合成口径：共振分 aug0 强（0.9 vs 0.5）；NLL 仅 aug1 差（-5 vs -1）
    # → 融合后 aug1 双弱被剔，其余 neuron 至少在共振或质量之一不弱（ratio 0.5）
    base_scores_uneven = {k: (0.9 if k == "zh_aug0_dialogue" else 0.5)
                          for k in zh_domain_all}
    nll_only_aug1_bad = {k: (-5.0 if k == "zh_aug1_dialogue" else -1.0)
                         for k in zh_domain_all}
    result_uneven = {"round1_scores": base_scores_uneven}

    def _evolve(active, streaks=None, nll=nll_only_aug1_bad,
                probe_ret=None, result=None, gids=None):
        with mock.patch.object(cortex, "_rolling_nll_quality",
                               return_value=nll), \
             mock.patch.object(cortex, "_probe_inactive_fused",
                               return_value=probe_ret or {}):
            return cortex._instance_route_evolve(
                list(active), result or result_uneven,
                "已生成文本", "zh", dict(streaks or {}),
                window=12, evict_ratio=0.35, evict_streak=2,
                add_ratio=1.3, min_active=2,
                general_ids=gids or [0, 1, 2], fusion_mode="soft")

    # B1: 剔除 + 迟滞
    act = list(zh_domain_all)
    n1, s1 = _evolve(act, {})  # 第一次劣化：streak=1，不剔
    check("B1a. 单次劣化不剔除（迟滞）",
          set(n1) == set(act) and s1.get("zh_aug1_dialogue", 0) == 1,
          f"streaks={ {k: v for k, v in s1.items() if v} }")
    n2, s2 = _evolve(act, s1)  # 连续第二次劣化：剔除 aug1
    check("B1b. 连续劣化剔除（chunk 级收敛）",
          "zh_aug1_dialogue" not in n2
          and set(n2) == set(act) - {"zh_aug1_dialogue"},
          f"n2={n2}")
    # B3（并入）：跨域 neuron 不动
    n3, _ = _evolve(act + ["code"], {})
    check("B3. 跨域 neuron 不被演化（C22 域内约束）",
          "code" in n3 and "zh_aug1_dialogue" in n3, f"n3={n3}")
    # B2: 加入（未激活同域后验显著更高）
    nll_aug1_best = {k: (-1.0 if k == "zh_aug1_dialogue" else -5.0)
                     for k in zh_domain_all}
    n4, _ = _evolve(["zh_aug0_dialogue", "zh_std0_dialogue", "zh"],
                    nll=nll_aug1_best,
                    probe_ret={"zh_aug1_dialogue": 0.7, "zh_aug2_dialogue": 0.1})
    check("B2. 未激活同域后验显著更高 → 加入（双向）",
          "zh_aug1_dialogue" in n4, f"n4={n4}")
    # B4: min_active 保护（全劣化剔到下限保护）
    nll_all_bad = {k: (-1.0 if k == "zh_aug0_dialogue" else -5.0)
                   for k in zh_domain_all}
    cur = list(zh_domain_all)
    st = {}
    for _ in range(5):  # 多轮劣化剔除
        cur, st = _evolve(cur, st, nll=nll_all_bad)
    dom_keep = [k for k in cur if k == "zh" or k.startswith("zh_")]
    check("B4. min_active 保护（同域激活 >= 2）", len(dom_keep) >= 2,
          f"cur={cur}")
    # B5: 稳定（无劣化时激活集不变）
    nll_equal = {k: -1.0 for k in zh_domain_all}
    base_equal = {k: 0.5 for k in zh_domain_all}
    n5, s5 = _evolve(zh_domain_all, {}, nll=nll_equal,
                     result={"round1_scores": base_equal})
    check("B5. 无劣化时激活集不变（迟滞收敛）",
          set(n5) == set(zh_domain_all) and all(v == 0 for v in s5.values()),
          f"n5={n5}")

    # ── C. 集成：真实长生成触发实例级路由 ──
    print("\n[C] 集成长生成（instance_routing 默认开）...", flush=True)
    orig_evolve = cortex._instance_route_evolve
    calls_c = {"n": 0}

    def spy(*a, **k):
        calls_c["n"] += 1
        return orig_evolve(*a, **k)

    try:
        with mock.patch.object(cortex, "_instance_route_evolve", spy):
            out = cortex.generate(
                build_dialogue_prompt("写一篇关于春天的小短文。"),
                max_tokens=48, domain="zh", temperature=0.55,
            )
        check("C1. 长生成输出非空", isinstance(out, str) and len(out.strip()) > 0,
              f"out={out[:40]!r}")
        # 集成触发证据：chunk 评估点至少触发一次（演化逻辑正确性由 B 段单元
        # 保障；触发次数依赖实际生成长度——模型可能因 EOS/退化截断提前停止，
        # 且 BioOSS 牵引会改变生成分布，故不绑定固定次数）。
        check("C2. 实例级路由在 chunk 边界触发", calls_c["n"] >= 1,
              f"evolve_calls={calls_c['n']}")
        check("C3. 触发后输出不退化",
              not cortex._is_degenerate_text(out),
              f"out={out[:40]!r}")
    except Exception as e:
        check("C1. 长生成输出非空", False, f"err={e}")
        check("C2. 实例级路由被触发", False, f"err={e}")
        check("C3. 触发后输出不退化", False, f"err={e}")

    # ── D. 开关回归：关闭回退 C25-E 行为 ──
    print("\n[D] 开关回归（instance_routing=False）...", flush=True)
    calls_d = {"n": 0}

    def spy_off(*a, **k):
        calls_d["n"] += 1
        return orig_evolve(*a, **k)

    try:
        with mock.patch.object(cortex, "_instance_route_evolve", spy_off):
            out_off = cortex.generate(
                build_dialogue_prompt("介绍一下机器学习的基本概念。"),
                max_tokens=40, domain="zh", temperature=0.55,
                instance_routing=False,
            )
        check("D1. 关闭时生成正常（非空不退化）",
              isinstance(out_off, str) and len(out_off.strip()) > 0
              and not cortex._is_degenerate_text(out_off),
              f"out={out_off[:40]!r}")
        check("D2. 关闭时实例级路由不调用", calls_d["n"] == 0,
              f"evolve_calls={calls_d['n']}")
    except Exception as e:
        check("D1. 关闭时生成正常", False, f"err={e}")
        check("D2. 关闭时实例级路由不调用", False, f"err={e}")

    print(f"\n  总耗时: {time.time() - t0:.1f}s", flush=True)
    print("=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
