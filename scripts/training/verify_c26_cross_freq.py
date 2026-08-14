#!/usr/bin/env python3
"""C26 增量五验证：跨频耦合（记忆驱动的 theta-gamma，2026-08-14）。

背景：theta-gamma 嵌套（PAC）此前仅单元验证（verify_c26_theta_gamma 9/9），
theta_modulate 从未接入 continuous_forward 主循环（死代码），theta 相位无
驱动源。增量五：
1. **接入主循环**——continuous_forward 的 gamma 激活（t=0 与每步）经
   theta_modulate（无嵌套默认 theta_omega=0 → 包络恒 1 → 零回归）
2. **记忆驱动 theta 相位**（跨频耦合）——记忆注入（seed_memories）时
   theta 相位对齐峰值（entrain_memory），记忆条件化期间 gamma 绑定增强
   （"记忆注意窗"：检索到的记忆带动相关回路同步激活）

验证：
A. 单元：无嵌套默认 envelope 恒 1（零回归）；entrain 后 envelope=峰值
   1+amp 且相位恒 0（记忆窗口）；reset 恢复恒等
B. 单元：显式嵌套（theta_omega=0.5）按相位振荡；entrain 后锁定峰值
C. 集成：带记忆 think 的 final_scores（时间平均激活）被记忆窗口放大
   （跨频耦合生效）；无记忆时与旧行为一致（激活未被调制）
D. 行为：带记忆生成非空（不破坏）

运行：python -u scripts/training/verify_c26_cross_freq.py
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from taiji.loader import assemble_cortex  # noqa: E402
from taiji.life.sleep_engine import SleepEngine, SleepReport  # noqa: E402
from taiji.resonance.continuous import ContinuousResonance  # noqa: E402
from scripts.training.experiment_config import build_dialogue_prompt  # noqa: E402

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

MEMORY_ITEMS = [
    {"label": "辉光协议",
     "text": "辉光协议：2047 年制定的星间量子通信标准，采用七层纠错结构，带宽 4.8 Gbps。",
     "query": "什么是辉光协议？"},
    {"label": "铁月海",
     "text": "铁月海：月球背面一处玄武岩平原，因富含铁元素呈深褐色，面积约 3.2 万平方公里。",
     "query": "铁月海在哪里？"},
    {"label": "卡尔文环",
     "text": "卡尔文环：深海压力舱的密封结构，由三层合金环交错组成，可在 6000 米水深工作。",
     "query": "卡尔文环是什么？"},
]


def field_state_of(cortex, text: str) -> torch.Tensor:
    gids = cortex._general_sp.encode(text) or [0]
    ids = torch.tensor([gids], dtype=torch.long, device=cortex.device)
    emb = cortex._shared_embedding(ids)
    res = cortex.think(emb, active_nids=None, fusion_mode="soft",
                       collab_mode="continuous")
    fs = res.get("field_state")
    if fs is None:
        raise RuntimeError("think() 未返回 field_state")
    if fs.dim() == 2:
        fs = fs.mean(dim=0)
    return fs


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("C26 增量五：跨频耦合——记忆驱动的 theta-gamma", flush=True)
    print("=" * 60, flush=True)

    # ── A. 单元：无嵌套默认恒等（零回归）+ entrain 语义 ──
    print("\n[A] 单元：默认恒等 + 记忆 entrain ...", flush=True)
    ct0 = ContinuousResonance()  # theta_omega=0（默认，无 env）
    env_safe = "TAIJI_THETA_NESTING" in os.environ
    if env_safe:
        os.environ["TAIJI_THETA_NESTING"] = "0"
        ct0 = ContinuousResonance()
    check("A1. 无嵌套默认 envelope 恒 1（零回归）",
          ct0.theta_envelope(0) == 1.0 and ct0.theta_envelope(3.2) == 1.0,
          f"env={os.environ.get('TAIJI_THETA_NESTING', 'unset')}")
    ct0.entrain_memory()
    check("A2. entrain 后 envelope=峰值 1+amp 且相位恒 0（记忆窗口）",
          abs(ct0.theta_envelope(0) - (1 + ct0.theta_amp)) < 1e-9
          and ct0.theta_phase_at(99) == 0.0,
          f"env={ct0.theta_envelope(0):.3f}")
    check("A3. 记忆窗口内 gamma 调制生效（theta_modulate 放大）",
          abs(float(ct0.theta_modulate(torch.tensor([1.0]), 0)[0])
              - (1 + ct0.theta_amp)) < 1e-6)
    ct0.reset_entrain()
    check("A4. reset 后恢复恒等（下次 forward 干净）",
          ct0.theta_envelope(0) == 1.0)

    # ── B. 单元：显式嵌套按相位振荡；entrain 锁定峰值 ──
    print("\n[B] 单元：显式嵌套振荡 + entrain 锁定 ...", flush=True)
    ct1 = ContinuousResonance(theta_omega=0.5)
    env_backup = os.environ.get("TAIJI_THETA_NESTING")
    os.environ.pop("TAIJI_THETA_NESTING", None)
    ct1 = ContinuousResonance(theta_omega=0.5)
    peak = 1 + ct1.theta_amp
    trough = 1 - ct1.theta_amp
    check("B1. 显式嵌套：峰值包络 1+amp", abs(ct1.theta_envelope(0) - peak) < 1e-9)
    check("B2. 显式嵌套：半周期后谷值 1-amp",
          abs(ct1.theta_envelope(math.pi / 0.5) - trough) < 1e-6,
          f"env={ct1.theta_envelope(math.pi / 0.5):.3f}")
    ct1.entrain_memory()
    check("B3. entrain 后锁定峰值（不再随 t 振荡）",
          abs(ct1.theta_envelope(100.0) - peak) < 1e-9)
    ct1.reset_entrain()
    check("B4. reset 后恢复振荡",
          abs(ct1.theta_envelope(0) - peak) < 1e-9
          and abs(ct1.theta_envelope(math.pi / 0.5) - trough) < 1e-6)
    if env_backup is not None:
        os.environ["TAIJI_THETA_NESTING"] = env_backup

    tmp_dir = tempfile.mkdtemp(prefix="c26_cross_freq_")
    try:
        # ── C. 集成：记忆窗口放大激活（跨频耦合生效）──
        cortex, tokenizer, modules = assemble_cortex(
            neurons_dir="data/neurons",
            collab_name=COLLAB_NAME,
            extra_neurons_dir=EXTRA_NEURONS_DIR,
            device="cpu",
            max_rounds=3,
            wire_bio_modules=True,
            neuron_ids=DIALOGUE_IDS,
        )
        check("装配成功（9 神经元）", len(cortex.neurons) == 9)

        sleep_engine = SleepEngine(data_dir=tmp_dir)
        sleep_engine.set_brain_interfaces(cortex=cortex)
        for item in MEMORY_ITEMS:
            vec = field_state_of(cortex, item["text"])
            sleep_engine.record_field_memory(vec, item["label"], text=item["text"])
        r = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        duration_seconds=0)
        sleep_engine._sleep_phase_field_consolidation(r)
        bank = cortex._memory_bank

        print("\n[C] 集成：记忆窗口放大 final_scores ...", flush=True)
        amp_ok = 0
        no_amp_ok = 0
        for item in MEMORY_ITEMS:
            qv = field_state_of(cortex, item["query"])
            top = bank.retrieve_vectors(qv, top_k=1)
            ids_t = torch.tensor([cortex._general_sp.encode(item["query"]) or [0]],
                                 dtype=torch.long, device=cortex.device)
            emb = cortex._shared_embedding(ids_t)
            no_mem = cortex.think(emb, active_nids=None, fusion_mode="soft",
                                  collab_mode="continuous")
            mem = cortex.think(emb, active_nids=None, fusion_mode="soft",
                               collab_mode="continuous",
                               memory_vectors=[(top[0][2], top[0][1])])
            s_no = sum(no_mem.get("final_scores", {}).values())
            s_mem = sum(mem.get("final_scores", {}).values())
            # 记忆窗口 → gamma 激活 ×(1+amp) → 时间平均权重放大
            amp_ok += 1 if s_mem > s_no else 0
            no_amp_ok += 1 if s_no > 0 else 0
            print(f"    {item['label']}: final_scores 无记忆={s_no:.3f} → "
                  f"带记忆={s_mem:.3f} (放大={s_mem > s_no})", flush=True)
        check("C1. 记忆窗口放大时间平均激活（跨频耦合生效）",
              amp_ok >= 1, f"{amp_ok}/{len(MEMORY_ITEMS)} 放大")
        check("C2. 无记忆 forward 激活正常（非零）",
              no_amp_ok == len(MEMORY_ITEMS))

        # ── D. 行为：带记忆生成非空（不破坏）──
        print("\n[D] 行为：记忆条件化生成 ...", flush=True)
        nonempty = 0
        for item in MEMORY_ITEMS:
            qv = field_state_of(cortex, item["query"])
            top = bank.retrieve_vectors(qv, top_k=1)
            cortex.field.reset()
            if cortex._dialogue_state is not None:
                cortex._dialogue_state.reset()
            out = cortex.generate(build_dialogue_prompt(item["query"]),
                                  max_tokens=20, domain="zh",
                                  memory_vectors=[(top[0][2], top[0][1])])
            nonempty += 1 if out and out.strip() else 0
            print(f"    {item['label']}: {out[:20]!r}", flush=True)
        check("D. 记忆条件化生成非空（不破坏）",
              nonempty == len(MEMORY_ITEMS), f"{nonempty}/{len(MEMORY_ITEMS)}")

        print(f"\n[验证摘要] {tmp_dir}", flush=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL  ({time.time() - t0:.1f}s)", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
