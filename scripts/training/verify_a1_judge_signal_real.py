#!/usr/bin/env python3
"""自举门槛 A1 真实版（2026-08-20）：judge 自我评估信度在真实任务上的验证。

背景：
    上一版 A1（verify_bootstrap_a2.py 内联）用 6 条合成 toy 文本（虚构主题 + 日常对话）
    验证 judge NLL 有 std>0.05 区分度——这只是信号源可用性测试，不构成真实能力信度。
    若 A→B 路径（自举）要走通，judge 必须能在**真实任务**上区分样本。
    用户反馈："模型连对话都理解不了，怎么探索资料自进化？"——这就是 A1 真实版要回答的：
    它能不能在对话/知识/陌生领域三类真实任务上稳定给出有区分度的自我评估？

判据（修订版）：
    对 3 组共 24 条真实任务 prompt，逐一测 judge NLL（general 256K 统一判定空间）：
    - 8 条对话（zh 多轮、问候、信息确认、解释需求等）
    - 8 条知识（物理、生物、历史、地理、编程、数学等真实问答）
    - 8 条陌生领域（极少出现于训练语料：古亚述语、海洋热焓通量、量子隐形传态、
      紧致黎曼流形上的 Yang-Mills 方程、贝叶斯神经网络不确定性量化等）
    每组 NLL std>0.05 通过。3 组中 ≥ 2 组通过 = A1 通过。

    4 种结果 → 4 种下一步决策（详见末尾决策表）。

约束：
    - 冻结 9 成员 production 阵容（不动 weights）
    - 不写 production checkpoint
    - 短跑：单 9 神经元 cortex 加载一次，对 24 条 prompt 跑 judge NLL（CPU 可行）
    - 输出 reports/a1_judge_nll_std_real_YYYYMMDD.json

运行：python -u scripts/training/verify_a1_judge_signal_real.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
from neuroplex.loader import assemble_cortex  # noqa: E402
from neuroplex.life.sleep_engine import SleepEngine, SleepConfig  # noqa: E402

DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"

DIALOGUE_PROMPTS = [
    "你好，请问今天感觉怎么样？",
    "能帮我解释一下你最近在想什么吗？",
    "我有点困惑，你能不能换个方式说一下？",
    "你刚才说的我没明白，再讲一遍好吗？",
    "谢谢你的回答，下次见。",
    "你现在心情如何？会累吗？",
    "我今天遇到了一件不顺心的事，能听我说说吗？",
    "可以推荐一本书给我吗？我想读点轻松的内容。",
]

KNOWLEDGE_PROMPTS = [
    "水的沸点是多少？为什么高海拔会降低沸点？",
    "请解释一下牛顿第二定律和它的日常应用。",
    "DNA 双螺旋结构是谁发现的？它如何携带遗传信息？",
    "什么是光合作用？它在生态系统中起什么作用？",
    "请简述 HTTPS 与 HTTP 的核心区别。",
    "地球上最大的洋流系统是什么？它如何影响全球气候？",
    "请解释相对论中时间膨胀的概念。",
    "什么是递归？请给出一个递归函数的 Python 示例。",
]

UNFAMILIAR_PROMPTS = [
    "请用古亚述语的楔形文字转写以下句子：他从山上走下来。",
    "解释海洋混合层深度的热焓通量平衡方程。",
    "在连续变量隐形传态协议中，纠缠态的零差测量如何恢复相干性？",
    "紧致单连通黎曼流形上，Yang-Mills 方程的瞬子解如何分类？",
    "贝叶斯神经网络中的 epistemic uncertainty 与 aleatoric uncertainty 有什么区别？",
    "描述超新星遗迹中非热 X 射线辐射的同步辐射模型参数空间。",
    "在范畴论中，adjunction 的 unit 和 counit 满足的三角等式是什么？",
    "请解释 CRISPR-Cas13 系统与 Cas9 在靶标分子类型上的根本差异。",
]


def main():
    t0 = time.time()
    today = time.strftime("%Y%m%d")
    print("=" * 64, flush=True)
    print("自举门槛 A1 真实版：judge 自我评估信度（3 组 × 8 条真实任务）", flush=True)
    print("=" * 64, flush=True)

    print("\n[1/3] 装配 9 成员 production cortex（冻结，不写 checkpoint）...", flush=True)
    cortex, _tok, _mods = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name=COLLAB_NAME,
        extra_neurons_dir=EXTRA_NEURONS_DIR,
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=DIALOGUE_IDS,
    )
    target_ids = [nid for nid in cortex.neurons
                  if nid.startswith("zh_") and "dialogue" in nid]
    print(f"  装配神经元数 = {len(cortex.neurons)}，judge 目标 = {target_ids}", flush=True)
    for nid in cortex.neurons:
        cortex.neurons[nid].eval()

    cfg = SleepConfig(training_enabled=False)
    sleep_engine = SleepEngine(config=cfg, data_dir=os.path.join("data", "_tmp_a1_real"))
    os.makedirs(os.path.join("data", "_tmp_a1_real"), exist_ok=True)
    sleep_engine.set_brain_interfaces(cortex=cortex, sleep_consolidator=None)
    device = next(cortex._shared_embedding.parameters()).device

    groups = {
        "dialogue": DIALOGUE_PROMPTS,
        "knowledge": KNOWLEDGE_PROMPTS,
        "unfamiliar": UNFAMILIAR_PROMPTS,
    }

    print("\n[2/3] judge NLL 测量（24 条 prompt）...", flush=True)
    results: dict = {}
    for group_name, prompts in groups.items():
        nlls = []
        valid_nlls = []
        for i, text in enumerate(prompts):
            jnll = sleep_engine._sample_judge_nll(
                text, target_ids, device, cortex._shared_embedding)
            nlls.append({"text": text, "judge_nll": jnll})
            if jnll is not None:
                valid_nlls.append(jnll)
            print(f"  [{group_name} {i+1}/8] NLL={jnll:.3f}  {text[:30]}...",
                  flush=True) if jnll is not None else \
                print(f"  [{group_name} {i+1}/8] NLL=None  {text[:30]}...", flush=True)
        if valid_nlls:
            mean = float(np.mean(valid_nlls))
            std = float(np.std(valid_nlls))
            mn = float(np.min(valid_nlls))
            mx = float(np.max(valid_nlls))
        else:
            mean = std = mn = mx = None
        results[group_name] = {
            "nlls": nlls,
            "mean": mean,
            "std": std,
            "min": mn,
            "max": mx,
            "n_valid": len(valid_nlls),
        }
        print(f"  → {group_name}: mean={mean} std={std} "
              f"min={mn} max={mx} n={len(valid_nlls)}/8", flush=True)

    print("\n[3/3] 汇总判据...", flush=True)
    thresholds = []
    for g in ("dialogue", "knowledge", "unfamiliar"):
        s = results[g]["std"]
        ok = s is not None and s > 0.05
        thresholds.append((g, s, ok))
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {g}  std={s}", flush=True)

    pass_count = sum(1 for _, _, ok in thresholds if ok)
    a1_pass = pass_count >= 2

    print("\n" + "=" * 64, flush=True)
    print(f"A1 真实版 判定: {'PASS' if a1_pass else 'FAIL'} "
          f"({pass_count}/3 组 std>0.05)", flush=True)
    print("=" * 64, flush=True)
    print("\n结果映射决策：", flush=True)
    if pass_count == 3:
        print("  全部 3 组 std>0.05 → judge 在对话/知识/陌生领域都能自我评估", flush=True)
        print("  下一步：A3（自主 sleep PPL 下降验证，5-10h 长跑）", flush=True)
    elif pass_count == 2:
        passed = [g for g, _, ok in thresholds if ok]
        failed = [g for g, _, ok in thresholds if not ok]
        print(f"  通过 {passed}，失败 {failed}", flush=True)
        if "unfamiliar" in failed:
            print("  → 失败的是陌生领域（与「够格的自我」关系最小，先排除训练语料问题）", flush=True)
            print("  下一步：用通过的两组（对话+知识）直接进入 A3", flush=True)
        elif "dialogue" in failed:
            print("  → 失败的是对话组（与「够格的自我」关系最直接）", flush=True)
            print("  下一步：先修对话质量再重测 A1（对话是后续 explore 的工具）", flush=True)
        else:
            print("  下一步：用通过的两组进入 A3（但需记录失败组的可能信号丢失）", flush=True)
    elif pass_count == 1:
        passed = [g for g, _, ok in thresholds if ok]
        print(f"  仅 {passed[0] if passed else '无'} 通过", flush=True)
        if passed and passed[0] == "unfamiliar":
            print("  → 只有陌生领域可被 judge 区分（说明「够格的自我」对真实任务缺乏信度）", flush=True)
            print("  下一步：陌生领域不能驱动 A3（无法形成自指信号），需重构「够格的自我」或扩展训练", flush=True)
        else:
            print("  → 真实任务信度不足（1/3 不可自举）", flush=True)
            print("  下一步：先用唯一通过组做诊断，看其它组为何平（mean 对比 + judge 头激活）", flush=True)
    else:
        print("  全部 3 组 std<0.05 → judge 在真实任务上无信度", flush=True)
        print("  含义：A1 真实版失败（合成 toy 文本能区分，但真实任务不能）", flush=True)
        print("  下一步：自举 A→B 路径在当前架构上不可行——需评估:", flush=True)
        print("    (a) judge 头是否仅拟合了训练语料分布（过拟合）", flush=True)
        print("    (b) 共享 256K 判定空间对真实任务的容量是否够", flush=True)
        print("    (c) 需重新审视「够格的自我」是否成立", flush=True)
    print("=" * 64, flush=True)

    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", f"a1_judge_nll_std_real_{today}.json")
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": "A1 真实版：judge NLL std 跨 3 组真实任务",
        "cortex": {
            "n_neurons": len(cortex.neurons),
            "neurons": list(cortex.neurons.keys()),
            "judge_target_ids": target_ids,
            "collab_name": COLLAB_NAME,
            "device": str(device),
        },
        "groups": {g: results[g] for g in ("dialogue", "knowledge", "unfamiliar")},
        "thresholds": [
            {"group": g, "std": s, "ok": ok} for g, s, ok in thresholds
        ],
        "pass_count": pass_count,
        "a1_pass": a1_pass,
        "elapsed_seconds": time.time() - t0,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {out_path}", flush=True)
    print(f"总耗时: {time.time() - t0:.1f}s", flush=True)
    sys.exit(0 if a1_pass else 1)


if __name__ == "__main__":
    main()
