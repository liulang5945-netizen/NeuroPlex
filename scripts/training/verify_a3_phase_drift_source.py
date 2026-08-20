#!/usr/bin/env python3
"""P0 精确定位：A3 漂移是 Phase 1.6 还是 Phase 1.7 引入的（2026-08-20）。

背景：
    verify_a3_drift_source_sniff.py 已证：8 轮无 sleep 训练下 max|Δ|=0.0000。
    → 漂移 100% 来自 sleep phase 实际执行过程。

    现在精确定位：分别跑 Phase 1.6 (synaptic_consolidation) 和
    Phase 1.7 (forward_replay)，每跑一个 phase 后冻结 cortex 测 24 prompt
    judge NLL，看哪个 phase 让 NLL 漂起来。

方法：
    T0 → 测基线
    T1 → 跑 1 次 Phase 1.6 → 测 NLL（看 LoRA 训练后 NLL 是否变）
    T2 → 跑 1 次 Phase 1.7 → 测 NLL
    T3 → 再跑 1 次 Phase 1.6 → 测 NLL（看是否累积）
    T4 → 再跑 1 次 Phase 1.7 → 测 NLL
    T5 → ... 交替多轮

    关键：
    - Phase 1.6 改 live.lora_adapters（A 直接训）
    - Phase 1.7 改 live.lora_adapters + lora_decay_per_sleep 衰减
    - 应该看 LoRA 训练后 judge NLL 是否真变（之前 P0 sniff 错以为 B=0，
      实际 synaptic_consolidation 会训 B 让它从 0 涨起来）

约束：冻结 9 成员 production weights，CPU 短跑（<2 分钟）。
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
from neuroplex.life.sleep_engine import SleepEngine, SleepConfig, SleepReport  # noqa: E402

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

# 简单 sleep 触发文本
SLEEP_TEXTS = [
    "今天有用户说谢谢，我有点感动。",
    "用户问我在想什么，我不知道怎么回答。",
    "再说一遍我没说清楚的概念给我。",
    "推荐一本书让用户开心。",
]


def lora_l2_norm(neuron):
    s = 0.0
    with torch.no_grad():
        for p in neuron.lora_adapters.parameters():
            s += float(p.data.pow(2).sum().item())
    return s ** 0.5


def main():
    t0 = time.time()
    today = time.strftime("%Y%m%d")
    n_cycles = int(os.environ.get("A3_DRIFT_PHASE_CYCLES", "3"))
    print("=" * 64, flush=True)
    print(f"P0 精确定位：Phase 1.6 vs Phase 1.7 对 judge NLL 漂移的贡献", flush=True)
    print("=" * 64, flush=True)

    print("\n[1/5] 装配 9 成员 production cortex（冻结）...", flush=True)
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
    print(f"  judge 目标 = {target_ids}", flush=True)
    for nid in cortex.neurons:
        cortex.neurons[nid].eval()
    device = next(cortex._shared_embedding.parameters()).device

    cfg = SleepConfig(training_enabled=False, judge_driven_replay=True,
                      lora_decay_per_sleep=0.9)
    sleep_engine = SleepEngine(config=cfg, data_dir=os.path.join("data", "_tmp_phase_sniff"))
    os.makedirs(os.path.join("data", "_tmp_phase_sniff"), exist_ok=True)
    sc_holder = {"sc": None}
    sc_module = None
    try:
        from neuroplex.resonance.neuro_modulation import SleepConsolidator  # noqa
        sc_holder["sc"] = SleepConsolidator(replay_buffer_size=50)
        sleep_engine.set_brain_interfaces(cortex=cortex, sleep_consolidator=sc_holder["sc"])
        print("  [sc] SleepConsolidator 接线成功（replay buffer size=50）", flush=True)
    except Exception as e:
        print(f"  set_brain_interfaces fallback: {e}", flush=True)
        sleep_engine.set_brain_interfaces(cortex=cortex, sleep_consolidator=None)

    groups = {
        "dialogue": DIALOGUE_PROMPTS,
        "knowledge": KNOWLEDGE_PROMPTS,
        "unfamiliar": UNFAMILIAR_PROMPTS,
    }
    all_prompts = [p for g in groups.values() for p in g]

    print("\n[2/5] 注入 24 条 prompt 记忆（喂经验，A3 with decay 一致）...", flush=True)
    injected = 0
    for i, text in enumerate(all_prompts):
        gids = cortex._general_sp.encode(text) or [0]
        ids = torch.tensor([gids], dtype=torch.long, device=cortex.device)
        emb = cortex._shared_embedding(ids)
        res = cortex.think(emb, active_nids=None, fusion_mode="soft",
                           collab_mode="continuous")
        fs = res.get("field_state")
        if fs is None:
            continue
        if fs.dim() == 2:
            fs = fs.mean(dim=0)
        try:
            sleep_engine.record_field_memory(fs, f"init_{i}", text=text)
        except Exception:
            pass
        if sc_holder["sc"] is not None:
            try:
                sc_holder["sc"].record_high_resonance_state(
                    field_state=fs, resonance_score=0.9, step=0,
                    active_nids=target_ids, threshold=0.5, text=text)
            except Exception as e:
                pass
        injected += 1
    r_init = SleepReport(timestamp=time.strftime("%Y%m%d-%H%M%S"),
                         duration_seconds=0.0)
    sleep_engine._sleep_phase_field_consolidation(r_init)
    print(f"  注入 {injected} 条 + 场固化 {r_init.field_memories_consolidated} 条", flush=True)

    history = []

    def measure(label, t_phase_start=None):
        nlls = []
        valid_nlls = []
        per_neuron = {nid: [] for nid in target_ids}
        t1 = time.time()
        for text in all_prompts:
            jnll = sleep_engine._sample_judge_nll(
                text, target_ids, device, cortex._shared_embedding)
            nlls.append({"text": text, "judge_nll": jnll})
            if jnll is not None:
                valid_nlls.append(jnll)
        dt = time.time() - t1
        l2s = {nid: round(lora_l2_norm(cortex.neurons[nid]), 4) for nid in target_ids}
        if valid_nlls:
            mean = float(np.mean(valid_nlls))
            std = float(np.std(valid_nlls))
        else:
            mean = std = None
        entry = {
            "label": label,
            "mean": mean, "std": std, "n_valid": len(valid_nlls),
            "duration_sec": round(dt, 2),
            "lora_l2": l2s,
            "nlls": nlls,
        }
        history.append(entry)
        print(f"  [{label}] mean={mean} std={std} dt={dt:.1f}s lora_l2={l2s}", flush=True)
        return entry

    def run_phase15():
        """跑一次 Phase 1.5 field_consolidation（处理 record_field_memory 写入）。"""
        report = SleepReport(timestamp=time.strftime("%Y%m%d-%H%M%S"),
                             duration_seconds=0.0)
        try:
            sleep_engine._sleep_phase_field_consolidation(report)
        except Exception as e:
            print(f"  Phase 1.5 EXC: {type(e).__name__}: {e}", flush=True)
            return False
        return True

    def run_phase3():
        """跑一次 Phase 3 knowledge_integration（含 downscaling ×0.98 + 通道强化 ×1.1）。"""
        target_neuron_ids = target_ids
        report = SleepReport(timestamp=time.strftime("%Y%m%d-%H%M%S"),
                             duration_seconds=0.0)
        try:
            sleep_engine._sleep_phase_knowledge_integration(report)
        except Exception as e:
            print(f"  Phase 3 EXC: {type(e).__name__}: {e}", flush=True)
            return False
        return True

    def run_phase16():
        """跑一次 Phase 1.6 synaptic_consolidation（训练 LoRA）。

        关键修正：只对 4 个 aug 神经元 enable_lora，跳过 zh_std0_dialogue
        （A3 fast 实际行为：zh_std0_dialogue 是 134M 标准神经元，
         内部 lora 启用条件判为 False，所以 lora_l2 始终 0）。
        """
        target_neuron_ids = target_ids
        shard = SLEEP_TEXTS
        for nid in target_neuron_ids:
            n = cortex.neurons[nid]
            if "std" in nid:
                continue
            if len(n.lora_adapters) == 0:
                n.enable_lora(rank=16, layers=None)
        report = SleepReport(timestamp=time.strftime("%Y%m%d-%H%M%S"),
                             duration_seconds=0.0)
        try:
            sleep_engine._sleep_phase_synaptic_consolidation(report)
        except Exception as e:
            print(f"  Phase 1.6 EXC: {type(e).__name__}: {e}", flush=True)
            return False
        return True

    def run_phase17():
        """跑一次 Phase 1.7 forward_replay。"""
        target_neuron_ids = target_ids
        report = SleepReport(timestamp=time.strftime("%Y%m%d-%H%M%S"),
                             duration_seconds=0.0)
        try:
            sleep_engine._sleep_phase_forward_replay(report)
        except Exception as e:
            print(f"  Phase 1.7 EXC: {type(e).__name__}: {e}", flush=True)
            return False
        return True

    print("\n[2/5] T0 基线...", flush=True)
    measure("T0_baseline")

    print(f"\n[3/5] T1 跑 1 次 Phase 1.5→1.6→1.7（完整 sleep 周期）...", flush=True)
    if run_phase15() and run_phase16() and run_phase17():
        measure("T1_after_full_cycle")

    print(f"\n[5/5] T2-T{1 + n_cycles} 重复完整 sleep 周期 {n_cycles} 次...", flush=True)
    for i in range(n_cycles):
        if run_phase15() and run_phase16() and run_phase17():
            measure(f"T{2 + i}_after_full_cycle_{i+1}")
        if run_phase3():
            measure(f"T{2 + i}_after_phase3_{i+1}")

    print("\n[6/6] 分析漂移来源...", flush=True)
    means = [h["mean"] for h in history if h["mean"] is not None]
    base = means[0] if means else None
    drifts = [(h["label"], abs(h["mean"] - base)) for h in history if h["mean"] is not None]
    print(f"  baseline = {base}", flush=True)
    for label, d in drifts:
        print(f"  {label}: |Δ| = {d:.6f}", flush=True)

    phase16_labels = [h["label"] for h in history if "after_phase16" in h["label"]]
    phase17_labels = [h["label"] for h in history if "after_phase17" in h["label"]]
    phase3_labels = [h["label"] for h in history if "after_phase3" in h["label"]]
    max_drift_phase16 = max(
        (h["mean"] - base for h in history if "after_phase16" in h["label"]),
        default=0.0)
    max_drift_phase17 = max(
        (h["mean"] - base for h in history if "after_phase17" in h["label"]),
        default=0.0)
    max_drift_phase3 = max(
        (h["mean"] - base for h in history if "after_phase3" in h["label"]),
        default=0.0)

    print(f"\n  Phase 1.6 累计 |Δ mean| max = {max_drift_phase16:.6f}", flush=True)
    print(f"  Phase 1.7 累计 |Δ mean| max = {max_drift_phase17:.6f}", flush=True)
    print(f"  Phase 3   累计 |Δ mean| max = {max_drift_phase3:.6f}", flush=True)

    drifts = {
        "Phase 1.6 synaptic_consolidation": abs(max_drift_phase16),
        "Phase 1.7 forward_replay": abs(max_drift_phase17),
        "Phase 3 knowledge_integration": abs(max_drift_phase3),
    }
    biggest = max(drifts, key=drifts.get)
    if drifts[biggest] < 0.001:
        source = "无显著漂移"
    else:
        source = biggest + f" (|Δ mean|={drifts[biggest]:.6f})"
    print(f"  ★ 漂移主来源 = {source}", flush=True)

    summary = {
        "date": today,
        "elapsed_sec": round(time.time() - t0, 1),
        "n_cycles": n_cycles,
        "n_prompts": len(all_prompts),
        "history": history,
        "drift_per_step": drifts,
        "max_drift_phase16": max_drift_phase16,
        "max_drift_phase17": max_drift_phase17,
        "max_drift_phase3": max_drift_phase3,
        "source": source,
    }
    out_path = f"reports/a3_phase_drift_source_{today}.json"
    os.makedirs("reports", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  报告写入 {out_path}", flush=True)
    print(f"\n总耗时: {summary['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
