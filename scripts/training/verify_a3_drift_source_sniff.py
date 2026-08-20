#!/usr/bin/env python3
"""P0 重置：A3 judge NLL 漂移源 sniff（2026-08-20）。

目的：
    区分 R1-R4 候选真根因：
    - R1: 4 个 compact neuron LoRA 累积（已部分证否）
    - R2: 漂移与 LoRA 累积相关性弱（已部分证否）
    - R3: 有状态组件（gamma phase / working memory / neuromodulator）在采样间漂移
    - R4: 24 prompt NLL 测量采样噪声

方法：
    **无 sleep 训练**下，固定 cortex / LoRA / 所有有状态组件（不调用任何 sleep phase），
    对同一 24 prompt 跑 8 轮 `_sample_judge_nll`，看 judge NLL 漂移幅度。

    - 漂移 < 0.01 → R4 成立（噪声是根因），A3 阈值从 0.1 放宽到 0.15
    - 漂移 0.01-0.05 → R3 成立（测量抖动来自有状态组件）
    - 漂移 > 0.05 → R1/R2 仍有效，需要新机制

约束：冻结 9 成员 production weights，不写 checkpoint，CPU 短跑（<60s）。
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


def snapshot_stateful(cortex):
    """快照有状态组件：neuromodulator / gamma oscillator / working memory / maturity。"""
    snap = {}
    nm = getattr(cortex, "neuromodulator", None) or getattr(cortex, "_neuromodulator", None)
    if nm is not None:
        for attr in ("dopamine", "serotonin", "norepinephrine"):
            v = getattr(nm, attr, None)
            if v is not None:
                snap[f"neuromodulator.{attr}"] = float(v) if isinstance(v, float) else (
                    float(v.item()) if hasattr(v, "item") else None)
    go = getattr(cortex, "gamma_oscillator", None) or getattr(cortex, "_gamma_oscillator", None)
    if go is not None:
        if hasattr(go, "phase"):
            snap["gamma.phase"] = float(go.phase) if isinstance(go.phase, float) else (
                float(go.phase.item()) if hasattr(go.phase, "item") else None)
        if hasattr(go, "step"):
            snap["gamma.step"] = int(go.step) if not hasattr(go.step, "item") else int(go.step.item())
    wm = getattr(cortex, "working_memory", None) or getattr(cortex, "_working_memory", None)
    if wm is not None:
        if hasattr(wm, "current"):
            v = wm.current
            snap["wm.current"] = int(v) if not hasattr(v, "item") else int(v.item())
        if hasattr(wm, "max_tokens"):
            v = wm.max_tokens
            snap["wm.max_tokens"] = int(v) if not hasattr(v, "item") else int(v.item())
    mt = getattr(cortex, "maturity", None) or getattr(cortex, "_maturity", None)
    if mt is not None:
        for attr in ("weight", "lr_mult"):
            v = getattr(mt, attr, None)
            if v is not None:
                snap[f"maturity.{attr}"] = float(v) if isinstance(v, float) else (
                    float(v.item()) if hasattr(v, "item") else None)
    return snap


def restore_stateful(cortex, snap):
    """恢复有状态组件。"""
    nm = getattr(cortex, "neuromodulator", None) or getattr(cortex, "_neuromodulator", None)
    if nm is not None:
        for attr in ("dopamine", "serotonin", "norepinephrine"):
            v = snap.get(f"neuromodulator.{attr}")
            if v is not None and hasattr(nm, attr):
                cur = getattr(nm, attr)
                if hasattr(cur, "fill_"):
                    cur.fill_(v)
                else:
                    setattr(nm, attr, v)
    go = getattr(cortex, "gamma_oscillator", None) or getattr(cortex, "_gamma_oscillator", None)
    if go is not None:
        for attr, key in [("phase", "gamma.phase"), ("step", "gamma.step")]:
            v = snap.get(key)
            if v is not None and hasattr(go, attr):
                setattr(go, attr, v)
    wm = getattr(cortex, "working_memory", None) or getattr(cortex, "_working_memory", None)
    if wm is not None:
        for attr, key in [("current", "wm.current")]:
            v = snap.get(key)
            if v is not None and hasattr(wm, attr):
                setattr(wm, attr, v)
    mt = getattr(cortex, "maturity", None) or getattr(cortex, "_maturity", None)
    if mt is not None:
        for attr, key in [("weight", "maturity.weight"), ("lr_mult", "maturity.lr_mult")]:
            v = snap.get(key)
            if v is not None and hasattr(mt, attr):
                setattr(mt, attr, v)


def main():
    t0 = time.time()
    today = time.strftime("%Y%m%d")
    n_rounds = int(os.environ.get("A3_DRIFT_N_ROUNDS", "8"))
    print("=" * 64, flush=True)
    print(f"P0 重置：A3 judge NLL 漂移源 sniff（无 sleep 训练，{n_rounds} 轮）", flush=True)
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

    cfg = SleepConfig(training_enabled=False)
    sleep_engine = SleepEngine(config=cfg, data_dir=os.path.join("data", "_tmp_drift_sniff"))
    os.makedirs(os.path.join("data", "_tmp_drift_sniff"), exist_ok=True)
    sleep_engine.set_brain_interfaces(cortex=cortex, sleep_consolidator=None)
    device = next(cortex._shared_embedding.parameters()).device

    stateful_keys_collected = set()
    initial_snap = snapshot_stateful(cortex)
    stateful_keys_collected.update(initial_snap.keys())
    print(f"  快照有状态组件: {list(initial_snap.keys())}", flush=True)
    for k, v in initial_snap.items():
        print(f"    {k} = {v}", flush=True)

    groups = {
        "dialogue": DIALOGUE_PROMPTS,
        "knowledge": KNOWLEDGE_PROMPTS,
        "unfamiliar": UNFAMILIAR_PROMPTS,
    }
    all_prompts = [p for g in groups.values() for p in g]

    history = []
    print(f"\n[2/4] 跑 {n_rounds} 轮 _sample_judge_nll（不调用 sleep phase，"
          f"只在每轮采样前 restore 有状态组件）...", flush=True)
    for r in range(n_rounds):
        restore_stateful(cortex, initial_snap)
        snap_before = snapshot_stateful(cortex)
        t1 = time.time()
        nlls = []
        valid_nlls = []
        per_neuron_round1 = {}
        for text in all_prompts:
            jnll = sleep_engine._sample_judge_nll(
                text, target_ids, device, cortex._shared_embedding)
            nlls.append({"text": text, "judge_nll": jnll})
            if jnll is not None:
                valid_nlls.append(jnll)
        snap_after = snapshot_stateful(cortex)
        round1 = sleep_engine._sample_judge_nll(
            all_prompts[0], target_ids, device, cortex._shared_embedding)
        dt = time.time() - t1
        if valid_nlls:
            mean = float(np.mean(valid_nlls))
            std = float(np.std(valid_nlls))
            mn = float(np.min(valid_nlls))
            mx = float(np.max(valid_nlls))
        else:
            mean = std = mn = mx = None
        history.append({
            "round": r,
            "judge_nlls": nlls,
            "mean": mean, "std": std, "min": mn, "max": mx,
            "n_valid": len(valid_nlls),
            "duration_sec": round(dt, 2),
            "stateful_before": snap_before,
            "stateful_after": snap_after,
            "stateful_changed": {k: (snap_before.get(k), snap_after.get(k))
                                 for k in snap_before
                                 if snap_before.get(k) != snap_after.get(k)},
        })
        print(f"  [Round {r}] mean={mean} std={std} min={mn} max={mx} "
              f"valid={len(valid_nlls)}/24 dt={dt:.1f}s "
              f"changed={list(history[-1]['stateful_changed'].keys())}", flush=True)

    print("\n[3/4] 分析漂移幅度（max|Δ mean|、单条 prompt 漂移）...", flush=True)
    means = [h["mean"] for h in history if h["mean"] is not None]
    if len(means) >= 2:
        baseline_mean = means[0]
        max_drift = max(abs(m - baseline_mean) for m in means)
        max_drift_any = max(means) - min(means)
        final_drift = abs(means[-1] - baseline_mean)
    else:
        baseline_mean = max_drift = max_drift_any = final_drift = None
    per_prompt_drifts = []
    for i, p in enumerate(all_prompts):
        seq = [h["judge_nlls"][i]["judge_nll"] for h in history
               if h["judge_nlls"][i]["judge_nll"] is not None]
        if len(seq) >= 2:
            per_prompt_drifts.append({
                "text": p,
                "n": len(seq),
                "max_minus_min": float(max(seq) - min(seq)),
                "std": float(np.std(seq)),
            })
    per_prompt_drifts.sort(key=lambda x: x["max_minus_min"], reverse=True)
    print(f"  baseline mean (round 0) = {baseline_mean}", flush=True)
    print(f"  final mean (round {n_rounds - 1}) = {means[-1] if means else None}", flush=True)
    print(f"  max |Δ mean| vs round0 = {max_drift}", flush=True)
    print(f"  max - min across all rounds = {max_drift_any}", flush=True)
    print(f"  Top 5 prompts by drift:")
    for x in per_prompt_drifts[:5]:
        print(f"    {x['max_minus_min']:.4f}  std={x['std']:.4f}  {x['text'][:30]}", flush=True)

    print("\n[4/4] 判据...", flush=True)
    if max_drift is not None:
        if max_drift < 0.01:
            verdict = "R4（采样噪声是根因）"
        elif max_drift < 0.05:
            verdict = "R3（有状态组件在抖动）"
        else:
            verdict = "R1/R2 仍有效（其它机制在动）"
    else:
        verdict = "测量异常"
    print(f"  漂移 < 0.01 → R4 (噪声是根因)", flush=True)
    print(f"  漂移 0.01-0.05 → R3 (有状态组件抖动)", flush=True)
    print(f"  漂移 > 0.05 → R1/R2 (机制问题)", flush=True)
    print(f"  ★ 本次测量 = {max_drift:.4f} → {verdict}", flush=True)

    summary = {
        "date": today,
        "elapsed_sec": round(time.time() - t0, 1),
        "n_rounds": n_rounds,
        "n_prompts": len(all_prompts),
        "history": history,
        "drift": {
            "baseline_mean": baseline_mean,
            "final_mean": means[-1] if means else None,
            "max_abs_drift_vs_round0": max_drift,
            "max_minus_min_all_rounds": max_drift_any,
            "final_abs_drift": final_drift,
        },
        "per_prompt_drift_top": per_prompt_drifts[:5],
        "stateful_initial_snapshot": initial_snap,
        "stateful_changes_across_rounds": [
            h["stateful_changed"] for h in history if h["stateful_changed"]
        ],
        "verdict": verdict,
    }
    out_path = f"reports/a3_drift_source_sniff_{today}.json"
    os.makedirs("reports", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[5/5] 报告写入 {out_path}", flush=True)
    print(f"  verdict: {verdict}", flush=True)
    print(f"\n总耗时: {summary['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
