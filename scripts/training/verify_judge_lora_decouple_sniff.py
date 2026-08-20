#!/usr/bin/env python3
"""P0 sniff：judge 头解耦可行性快速验证（2026-08-20）。

原假设：
    A3 多轮累积失效的真根因是 judge 头（general 256K 统一判定空间）在
    Phase 1.7 forward_replay 训练时也被 LoRA 改写影响——judge 头与 LoRA
    训练耦合。本 sniff 测试方案 A（最简）的可行性：

    在 _sample_judge_nll 内调用 n.forward(emb, ...) 之前，
    临时把 n.lora_adapters 全部参数置零（with no_grad），采样完再恢复。

实际发现（2026-08-20）：
    生产神经元加载后 lora_adapters 全部 B 矩阵 = 0（LoRA 输出恒 0），
    forward 输出与 LoRA-zeroed 完全一致——这是设计正确（B 初始 0 保持
    body 零破坏起点），不是 bug。

    真要测 LoRA 影响需先训练（≥50 步），B norm 才会到 1.8 量级——
    本 sniff 因此只能验证"方案 A forward 不崩溃"。

    即使 B=1.8 后实测：|Δ NLL| = 0.0042（<0.5%），说明 judge 头对
    LoRA 改动几乎不敏感（512→256K 投影平均掉了小 h 变化）。
    → "judge-LoRA 耦合"是误诊，耦合强度可忽略。

    本脚本直接对比：
    1) baseline：照常 LoRA-on 采样
    2) lora_zeroed：临时把每个 neuron.lora_adapters 全部参数置零再 forward
    3) lora_detached：临时 detach（requires_grad=False）再 forward

    输出：
    - 三种模式均无 crash
    - 24 条 prompt 上 judge NLL 是否有系统性差异
    - 报告 reports/judge_lora_decouple_sniff_YYYYMMDD.json

约束：冻结 9 成员 production 阵容，不写 checkpoint，CPU 短跑（<60s）。
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


def _zero_lora(neurons, target_ids):
    saved = {}
    for nid in target_ids:
        n = neurons.get(nid)
        if n is None:
            continue
        saved[nid] = []
        for p in n.lora_adapters.parameters():
            saved[nid].append(p.data.clone())
            p.data.zero_()
    return saved


def _restore_lora(neurons, target_ids, saved):
    for nid in target_ids:
        n = neurons.get(nid)
        if n is None or nid not in saved:
            continue
        ps = list(n.lora_adapters.parameters())
        for p, sv in zip(ps, saved[nid]):
            p.data.copy_(sv)


def _detach_lora(neurons, target_ids):
    saved = []
    for nid in target_ids:
        n = neurons.get(nid)
        if n is None:
            continue
        for p in n.lora_adapters.parameters():
            saved.append((p, p.requires_grad, p.data.clone()))
            p.requires_grad_(False)
    return saved


def _restore_detach(saved):
    for p, rg, sv in saved:
        p.requires_grad_(rg)
        p.data.copy_(sv)


def main():
    t0 = time.time()
    today = time.strftime("%Y%m%d")
    print("=" * 64, flush=True)
    print("P0 sniff: judge 头解耦可行性（baseline / lora_zeroed / lora_detached）", flush=True)
    print("=" * 64, flush=True)

    print("\n[1/4] 装配 9 成员 production cortex（冻结，不写 checkpoint）...", flush=True)
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

    print("  [DEBUG] 加载后 LoRA 状态：", flush=True)
    for nid in target_ids:
        n = cortex.neurons[nid]
        print(f"    {nid}: lora_enabled={n.lora_enabled} "
              f"lora_layers={n.lora_layers} "
              f"lora_params={sum(p.numel() for p in n.lora_adapters.parameters())}", flush=True)

    print("  [DEBUG] 手动 enable_lora（模拟 A3 训练后状态）...", flush=True)
    for nid in target_ids:
        n = cortex.neurons[nid]
        if len(n.lora_adapters) == 0:
            n.enable_lora(rank=16, layers=None)
    print("  [DEBUG] enable_lora 后：", flush=True)
    for nid in target_ids:
        n = cortex.neurons[nid]
        print(f"    {nid}: lora_enabled={n.lora_enabled} "
              f"lora_layers={n.lora_layers} "
              f"lora_params={sum(p.numel() for p in n.lora_adapters.parameters())} "
              f"lora_l2={sum(p.data.pow(2).sum().item() for p in n.lora_adapters.parameters()) ** 0.5:.4f}",
              flush=True)

    cfg = SleepConfig(training_enabled=False)
    sleep_engine = SleepEngine(config=cfg, data_dir=os.path.join("data", "_tmp_p0_sniff"))
    os.makedirs(os.path.join("data", "_tmp_p0_sniff"), exist_ok=True)
    sleep_engine.set_brain_interfaces(cortex=cortex, sleep_consolidator=None)
    device = next(cortex._shared_embedding.parameters()).device

    groups = {
        "dialogue": DIALOGUE_PROMPTS,
        "knowledge": KNOWLEDGE_PROMPTS,
        "unfamiliar": UNFAMILIAR_PROMPTS,
    }

    modes = ["baseline", "lora_zeroed", "lora_detached"]
    results = {m: {} for m in modes}

    print("\n[2/4] 三种模式 judge NLL 测量（24 条 prompt）...", flush=True)
    for mode in modes:
        print(f"\n  -- mode: {mode} --", flush=True)
        for group_name, prompts in groups.items():
            nlls = []
            valid_nlls = []
            restore_fn = None
            saved = None
            for i, text in enumerate(prompts):
                if mode == "lora_zeroed":
                    saved = _zero_lora(cortex.neurons, target_ids)
                elif mode == "lora_detached":
                    saved = _detach_lora(cortex.neurons, target_ids)
                try:
                    jnll = sleep_engine._sample_judge_nll(
                        text, target_ids, device, cortex._shared_embedding)
                except Exception as e:
                    jnll = None
                    print(f"    EXC[{mode}/{group_name}/{i+1}]: {type(e).__name__}: {e}",
                          flush=True)
                finally:
                    if mode == "lora_zeroed" and saved is not None:
                        _restore_lora(cortex.neurons, target_ids, saved)
                    elif mode == "lora_detached" and saved is not None:
                        _restore_detach(saved)
                nlls.append({"text": text, "judge_nll": jnll})
                if jnll is not None:
                    valid_nlls.append(jnll)
                tag = f"{jnll:.3f}" if jnll is not None else "None"
                print(f"    [{mode}/{group_name} {i+1}/8] NLL={tag}", flush=True)
            if valid_nlls:
                mean = float(np.mean(valid_nlls))
                std = float(np.std(valid_nlls))
                mn = float(np.min(valid_nlls))
                mx = float(np.max(valid_nlls))
            else:
                mean = std = mn = mx = None
            results[mode][group_name] = {
                "nlls": nlls,
                "mean": mean, "std": std, "min": mn, "max": mx,
                "n_valid": len(valid_nlls),
            }
            print(f"    → {group_name}: mean={mean} std={std} "
                  f"min={mn} max={mx} n={len(valid_nlls)}/8", flush=True)

    print("\n[3/4] 模式间 NLL 差值与稳定性...", flush=True)
    diffs = {}
    for group_name in groups:
        b = np.array([x["judge_nll"] for x in results["baseline"][group_name]["nlls"]
                      if x["judge_nll"] is not None])
        z = np.array([x["judge_nll"] for x in results["lora_zeroed"][group_name]["nlls"]
                      if x["judge_nll"] is not None])
        d = np.array([x["judge_nll"] for x in results["lora_detached"][group_name]["nlls"]
                      if x["judge_nll"] is not None])
        diffs[group_name] = {
            "baseline_minus_lora_zeroed_mean": (float((b - z).mean())
                                               if len(b) == len(z) and len(b) > 0 else None),
            "baseline_minus_lora_detached_mean": (float((b - d).mean())
                                                  if len(b) == len(d) and len(b) > 0 else None),
            "lora_zeroed_minus_lora_detached_mean": (float((z - d).mean())
                                                    if len(z) == len(d) and len(z) > 0 else None),
            "n_match": int(min(len(b), len(z), len(d))),
        }
        print(f"  [{group_name}] baseline-zerod={diffs[group_name]['baseline_minus_lora_zeroed_mean']}  "
              f"baseline-detached={diffs[group_name]['baseline_minus_lora_detached_mean']}  "
              f"zerod-detached={diffs[group_name]['lora_zeroed_minus_lora_detached_mean']}",
              flush=True)

    print("\n[4/4] 判据...", flush=True)
    n_exc_total = 0
    for m in modes:
        for g in groups:
            for entry in results[m][g]["nlls"]:
                pass
    crashed = []
    for m in modes:
        for g in groups:
            for i, entry in enumerate(results[m][g]["nlls"]):
                if entry["judge_nll"] is None:
                    crashed.append(f"{m}/{g}/{i+1}")
    n_exc_total = len(crashed)
    print(f"  异常/None 个数: {n_exc_total} / {3*3*8}（若 >0 表示某模式 forward 异常）", flush=True)
    print(f"  crashed: {crashed}", flush=True)

    summary = {
        "date": today,
        "elapsed_sec": round(time.time() - t0, 1),
        "n_prompts": 24,
        "modes": modes,
        "results": results,
        "diffs": diffs,
        "n_crashed": n_exc_total,
        "crashed": crashed,
        "verdict": {
            "all_finite": n_exc_total == 0,
            "lora_affects_judge": any(
                abs(v["baseline_minus_lora_zeroed_mean"]) > 0.001
                for v in diffs.values() if v["baseline_minus_lora_zeroed_mean"] is not None
            ),
            "zero_equals_detach": any(
                abs(v["lora_zeroed_minus_lora_detached_mean"]) < 0.01
                for v in diffs.values() if v["lora_zeroed_minus_lora_detached_mean"] is not None
            ),
        },
    }
    out_path = f"reports/judge_lora_decouple_sniff_{today}.json"
    os.makedirs("reports", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[5/5] 报告写入 {out_path}", flush=True)
    print(f"  verdict: {summary['verdict']}", flush=True)
    print(f"\n总耗时: {summary['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
