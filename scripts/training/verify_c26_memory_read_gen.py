#!/usr/bin/env python3
"""C26 增量二验证：记忆可读进生成——记忆向量直接条件化生成（2026-08-14）。

背景：C26 第 0 格（场固化）完成时，记忆注入只有**文本标签通道**
（把检索到的记忆标签前置进 prompt）。增量二打通"记忆向量"通道：
检索到的记忆向量（统一场空间快照）写入共振场，round2+ 的场条件化
forward 让记忆通过**已训练的 field_state 注入路径**直接参与 token 生成
（"读"免训练——神经元 forward_train 即用 field_conditioning 训练过该路径）。

设计（上限优先）：写入点选在 round1 判定信号（round1_scores/judge）之后
——判定保持"无记忆的天然反应"（C23 安全边界），记忆只叠加在生成条件化层；
权重 = 检索相似度（近记忆强条件化）。

验证层次：
A. 检索升级：retrieve_vectors 返回记忆向量（top-1 命中 4/4）
B. 安全边界：带记忆的 round1_scores 与不带完全一致（判定信号不被污染）
C. 场拉拽：带记忆的 field_state 更靠近记忆向量（cosine 提升）
D. 条件化生效（硬）：leader 的场条件化 logits 因记忆而改变
E. 生成级（软）：向量通道单独注入改变生成输出（区别于文本通道）
F. 文本通道回归：原标签前置注入仍生效
G. 重启恢复：新实例 load 后检索仍命中

运行：python -u scripts/training/verify_c26_memory_read_gen.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
# N2（REMEDIATION_PLAN R7）：固定 seed 保证可复现
import random  # noqa: E402
import numpy as np  # noqa: E402

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
from taiji.loader import assemble_cortex  # noqa: E402
from taiji.life.sleep_engine import SleepEngine, SleepReport  # noqa: E402
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

# 编造术语样本（模型不可能先验知道 → 对照组"不含"断言可靠）
MEMORY_ITEMS = [
    {
        "label": "辉光协议",
        "text": "辉光协议：2047 年制定的星间量子通信标准，采用七层纠错结构，带宽 4.8 Gbps。",
        "query": "什么是辉光协议？",
        "key": "辉光协议",
    },
    {
        "label": "铁月海",
        "text": "铁月海：月球背面一处玄武岩平原，因富含铁元素呈深褐色，面积约 3.2 万平方公里。",
        "query": "铁月海在哪里？",
        "key": "铁月海",
    },
    {
        "label": "卡尔文环",
        "text": "卡尔文环：深海压力舱的密封结构，由三层合金环交错组成，可在 6000 米水深工作。",
        "query": "卡尔文环是什么？",
        "key": "卡尔文环",
    },
    {
        "label": "频谱蜂鸟",
        "text": "频谱蜂鸟：栖息于安第斯高海拔的鸟类，翼展仅 4 厘米，振翅频率达每秒 80 次。",
        "query": "频谱蜂鸟有什么习性？",
        "key": "频谱蜂鸟",
    },
]


def fresh_generate(cortex, prompt: str, max_tokens: int = 40,
                   memory_vectors: list | None = None) -> str:
    """模拟"新会话"：重置场 + 对话状态后再生成（隔离轮次污染）。"""
    cortex.field.reset()
    if cortex._dialogue_state is not None:
        cortex._dialogue_state.reset()
    # 口径（2026-08-12）：query 包装为对话训练格式（"问：...\n答："），
    # 与 dialogue neuron 的 SFT 训练分布一致，避免裸 prompt 假退化。
    return cortex.generate(build_dialogue_prompt(prompt), max_tokens=max_tokens,
                           domain="zh", memory_vectors=memory_vectors)


def field_state_of(cortex, text: str) -> torch.Tensor:
    """对文本做一次共振前向，取归一化场状态快照（think 返回的 field_state）。

    注意：generate 的生成循环每 token 调 think()，结束即重置场，
    cortex.field 在生成后是空的——场快照必须从 think() 返回值截获。
    """
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


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    return float((a @ b).item() / (a.norm() * b.norm() + 1e-8))


def scores_close(a: dict, b: dict, atol: float = 1e-4) -> bool:
    """round1_scores 容差比较（~1e-7 级运行间浮点噪声，1e-4 容差隔离真实污染）。"""
    if set(a.keys()) != set(b.keys()):
        return False
    return all(abs(a[k] - b[k]) <= atol for k in a)


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("C26 增量二：记忆可读进生成——记忆向量直接条件化生成", flush=True)
    print("=" * 60, flush=True)

    tmp_dir = tempfile.mkdtemp(prefix="c26_mem_read_")
    try:
        cortex, tokenizer, modules = assemble_cortex(
            neurons_dir="data/neurons",
            collab_name=COLLAB_NAME,
            extra_neurons_dir=EXTRA_NEURONS_DIR,
            device="cpu",
            max_rounds=3,
            wire_bio_modules=True,
            neuron_ids=DIALOGUE_IDS,
        )
        nids = list(cortex.neurons.keys())
        print(f"  装配 {len(nids)} 神经元: {nids}", flush=True)
        print(f"  场维度: dim={cortex.field.dim}", flush=True)
        check("装配成功（5 dialogue + 4 general）", len(nids) == 9, f"n={len(nids)}")

        sleep_engine = SleepEngine(data_dir=tmp_dir)
        sleep_engine.set_brain_interfaces(cortex=cortex)

        # ── 准备记忆库：4 条场记忆固化（复用 Phase 1.5 不触发训练）──
        for item in MEMORY_ITEMS:
            vec = field_state_of(cortex, item["text"])
            sleep_engine.record_field_memory(vec, item["label"])
        report = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                             duration_seconds=0)
        sleep_engine._sleep_phase_field_consolidation(report)
        bank = sleep_engine.get_field_memory()
        mem_path = os.path.join(tmp_dir, "field_memory.pt")
        check("记忆库固化 4 条", len(bank) == 4, f"bank={len(bank)}")

        # ── A. 检索升级：retrieve_vectors 返回记忆向量 ──
        print("\n[A] retrieve_vectors：返回 (label, sim, vector) ...", flush=True)
        hit = 0
        retrieved = {}
        for item in MEMORY_ITEMS:
            qv = field_state_of(cortex, item["query"])
            top = bank.retrieve_vectors(qv, top_k=1)
            ok = bool(top) and top[0][0] == item["label"]
            hit += 1 if ok else 0
            retrieved[item["key"]] = top[0] if top else None
            print(f"    {item['key']}: label={top[0][0] if top else '<空>'}, "
                  f"sim={top[0][1] if top else 0.0:.3f}, "
                  f"vec_dim={top[0][2].numel() if top else 0}", flush=True)
        check("A. 检索 top-1 命中 + 返回记忆向量",
              hit == len(MEMORY_ITEMS) and all(
                  r and r[2].numel() == cortex.field.dim
                  for r in retrieved.values()),
              f"{hit}/{len(MEMORY_ITEMS)}, dim 一致")

        # ── B/C/D. 管线级：think 带/不带记忆对比 ──
        print("\n[B-D] 场拉拽 + 判定信号干净 + leader 条件化 logits 改变 ...", flush=True)
        safe_ok = True
        pull_ok = 0
        logit_ok = 0
        for item in MEMORY_ITEMS:
            qv = field_state_of(cortex, item["query"])
            rec = retrieved[item["key"]]
            mem_vec = rec[2].to(cortex.device)
            ids = torch.tensor([cortex._general_sp.encode(item["query"]) or [0]],
                               dtype=torch.long, device=cortex.device)
            emb = cortex._shared_embedding(ids)
            base = cortex.think(emb, active_nids=None, fusion_mode="soft",
                                collab_mode="continuous")
            mem = cortex.think(emb, active_nids=None, fusion_mode="soft",
                               collab_mode="continuous",
                               memory_vectors=[(mem_vec, rec[1])])
            # B. 安全边界：round1_scores 不受记忆影响（写入点在其后）
            # 注意：相同调用间的 round1_scores 本身有 ~1e-7 级浮点噪声
            # （预存在，与记忆无关），故用容差比较隔离真实污染。
            if not scores_close(base.get("round1_scores") or {},
                                mem.get("round1_scores") or {}):
                safe_ok = False
            # C. 场拉拽：带记忆的 field_state 更靠近记忆向量
            c_base = cos(base.get("field_state"), mem_vec)
            c_mem = cos(mem.get("field_state"), mem_vec)
            pull_ok += 1 if c_mem > c_base + 1e-6 else 0
            # D. 条件化生效（硬）：leader 的场条件化 logits 被记忆改变
            r1s = mem.get("round1_scores") or {}
            leader = max(r1s, key=r1s.get) if r1s else "?"
            lb = base.get("round1_logits", {}).get(leader)
            lm = mem.get("round1_logits", {}).get(leader)
            logits_changed = (lb is not None and lm is not None
                              and lb.shape == lm.shape
                              and bool((lb.detach() != lm.detach()).any()))
            if logits_changed:
                logit_ok += 1
            print(f"    {item['key']}: cos(场,记忆) {c_base:.3f}→{c_mem:.3f}, "
                  f"leader={leader}, logits_changed={logits_changed}", flush=True)
        check("B. 安全边界：round1_scores 不被记忆污染", safe_ok,
              "判定信号保持干净（C23）")
        check("C. 场拉拽：带记忆的场更靠近记忆向量", pull_ok == len(MEMORY_ITEMS),
              f"{pull_ok}/{len(MEMORY_ITEMS)}")
        check("D. leader 场条件化 logits 因记忆改变（硬）", logit_ok == len(MEMORY_ITEMS),
              f"{logit_ok}/{len(MEMORY_ITEMS)} 条 leader logits 改变")

        # ── E/F. 生成级：向量通道 vs 文本通道 vs 对照 ──
        print("\n[E-F] 生成：向量通道注入 / 文本通道回归 / 对照组 ...", flush=True)
        vec_changed = 0
        txt_changed = 0
        for item in MEMORY_ITEMS:
            rec = retrieved[item["key"]]
            base = fresh_generate(cortex, item["query"])
            vec_out = fresh_generate(cortex, item["query"],
                                     memory_vectors=[(rec[2], rec[1])])
            txt_out = fresh_generate(
                cortex, f"【记忆】{item['label']}：{item['key']}。\n{item['query']}")
            vec_changed += 1 if vec_out != base else 0
            txt_changed += 1 if txt_out != base else 0
            print(f"    {item['key']}: 对照={base[:28]!r}", flush=True)
            print(f"      向量通道={vec_out[:28]!r} (changed={vec_out != base})",
                  flush=True)
            print(f"      文本通道={txt_out[:28]!r} (changed={txt_out != base})",
                  flush=True)
        check("E. 向量通道单独注入改变生成输出（软）", vec_changed == len(MEMORY_ITEMS),
              f"{vec_changed}/{len(MEMORY_ITEMS)} 与对照不同")
        check("F. 文本通道回归：标签前置注入仍生效", txt_changed == len(MEMORY_ITEMS),
              f"{txt_changed}/{len(MEMORY_ITEMS)} 与对照不同")

        # ── G. 重启恢复 ──
        from taiji.resonance.field_memory import FieldMemoryBank
        bank2 = FieldMemoryBank()
        check("G. 新实例从磁盘恢复场记忆库",
              bank2.load(mem_path) and len(bank2) == 4, f"bank2={len(bank2)}")
        top2 = bank2.retrieve_vectors(field_state_of(cortex, MEMORY_ITEMS[0]["query"]),
                                      top_k=1)
        check("G. 恢复后跨重启检索命中（含向量）",
              bool(top2) and top2[0][0] == MEMORY_ITEMS[0]["label"]
              and top2[0][2].numel() == cortex.field.dim,
              f"top={top2[0][0] if top2 else None}")

        print(f"\n[验证摘要] {tmp_dir}", flush=True)
        print(f"  记忆库: {bank.status()}", flush=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL  ({time.time() - t0:.1f}s)", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
