#!/usr/bin/env python3
"""培养期端到端闭环验证（feed → sleep 训练 → 影子写回 → ckpt 保存，2026-08-11）。

验证 "可挂载客户端进入培养期"（BIO_INSPIRED_ARCHITECTURE_PLAN.md 挂载就绪结论）：
培养期闭环 = FeedEngine 收集样本 → SleepEngine 消化（Phase 2 神经元训练）
→ 影子权重写回 live → cortex_state.pt 经验持久化。

验证点：
1. 装配 9 神经元（5 dialogue + 4 general）+ FeedEngine/SleepEngine 接线
2. feed_text 喂 zh 样本 → get_pending_samples_by_domain 按域分类非空
3. 触发 Phase 2 训练：样本被消费（training_samples_used > 0）、loss 有限、PPL 记录
4. 影子权重写回 live：训练前后 lm_head/embed_adapter/shared_embedding 权重变化
5. ckpt 保存闭环：save_state 自动保存（隔离临时目录，不污染 data/neurons）
   → load_state 恢复成功
6. 训练后推理不崩：generate 非空输出（code 稳定 + zh 不抛异常）

运行：python -u scripts/training/verify_feed_sleep_e2e.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from taiji.loader import assemble_cortex  # noqa: E402
from taiji.life.feed_engine import FeedEngine  # noqa: E402
from taiji.life.sleep_engine import SleepConfig, SleepEngine, SleepReport  # noqa: E402

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

# 培养期目标样本：zh 域结构化中文知识（>50 字符、含标题/列表，质量评估通过）
ZH_SAMPLES = [
    "神经网络基础\n神经网络是一种受生物大脑启发的机器学习模型。\n核心组成：\n- 神经元：加权求和后经过激活函数\n- 层：输入层、隐藏层、输出层\n- 损失函数：衡量预测与真实的差距\n训练通过反向传播更新权重。",
    "梯度下降原理\n梯度下降是训练神经网络的核心优化方法。\n步骤：\n1. 前向传播计算损失\n2. 反向传播求梯度\n3. 沿负梯度方向更新参数\n学习率决定每次更新的步长。",
    "激活函数作用\n激活函数为神经网络引入非线性。\n常见类型：\n- ReLU：输入为正则输出本身\n- sigmoid：输出压缩到 0 到 1\n- tanh：输出压缩到负一到正一\n没有激活函数，多层网络退化为线性。",
    "过拟合与正则化\n过拟合指模型记住训练数据而非学习规律。\n常用正则化方法：\n- L2 正则化：惩罚大权重\n- Dropout：随机丢弃神经元\n- 早停：验证集不改善即停止\n- 数据增强：扩充训练样本。",
    "自然语言处理简介\n自然语言处理让计算机理解人类语言。\n典型任务：\n- 文本分类：判断文章主题\n- 机器翻译：跨语言转换\n- 问答系统：根据问题给出答案\n现代方法普遍使用注意力机制。",
    "Transformer 架构\nTransformer 是当前大语言模型的基础架构。\n核心组件：\n- 自注意力：捕捉词与词的关系\n- 位置编码：记录词的顺序\n- 前馈网络：逐位变换特征\n残差连接帮助深层网络稳定训练。",
    "词嵌入原理\n词嵌入将单词映射为稠密向量。\n特性：\n- 相似词向量距离近\n- 支持向量运算\n- 语义信息蕴含在向量中\n训练方式包括 CBOW 和 Skip-gram。",
    "注意力机制\n注意力机制让模型聚焦重要信息。\n工作方式：\n- 查询与键计算相似度\n- 相似度转为权重\n- 按权重加权求和值向量\n多头注意力并行捕捉不同子空间信息。",
]


def snapshot_learnable(cortex):
    """快照所有可学习参数（lm_head + embed_adapter + shared_embedding）用于对比。"""
    snaps = {}
    for nid, neuron in cortex.neurons.items():
        entry = {}
        if hasattr(neuron, "lm_head") and neuron.lm_head is not None:
            entry["lm_head"] = {k: v.detach().clone()
                                for k, v in neuron.lm_head.state_dict().items()}
        if hasattr(neuron, "embed_adapter") and neuron.embed_adapter is not None:
            entry["embed_adapter"] = {k: v.detach().clone()
                                      for k, v in neuron.embed_adapter.state_dict().items()}
        if entry:
            snaps[nid] = entry
    emb_snap = None
    if cortex._shared_embedding is not None:
        emb_snap = {k: v.detach().clone()
                    for k, v in cortex._shared_embedding.state_dict().items()}
    return snaps, emb_snap


def part_changed(before, after, nid, part) -> bool:
    b = before.get(nid, {}).get(part)
    a = after.get(nid, {}).get(part)
    if b is None or a is None:
        return False
    return any(k in a and not torch.equal(b[k], a[k]) for k in b)


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("培养期端到端闭环验证（feed → sleep → 写回 → ckpt）", flush=True)
    print("=" * 60, flush=True)

    # ── 1. 装配 9 神经元 + 临时数据目录隔离 ──
    tmp_dir = tempfile.mkdtemp(prefix="feed_sleep_e2e_")
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
    print(f"  装配: {nids}", flush=True)
    check("9 神经元装配（5 dialogue + 4 general）", len(nids) == 9, f"n={len(nids)}")
    check("general zh 基座就位（培养目标域）", "zh" in nids)

    # 隔离经验持久化路径（不污染生产 data/neurons/cortex_state.pt）
    cortex.neurons_dir = tmp_dir

    # ── 2. FeedEngine 接线 + 喂 zh 样本 ──
    feed_dir = os.path.join(tmp_dir, "feed_data")
    feed_engine = FeedEngine(data_dir=feed_dir)
    fed = 0
    for i, text in enumerate(ZH_SAMPLES):
        item = feed_engine.feed_text(text, source="e2e", category="knowledge", domain="zh")
        if item is not None and item.status == "digested":
            fed += 1
    print(f"  喂食完成: {fed}/{len(ZH_SAMPLES)} 条 zh 样本被消化", flush=True)
    check("zh 样本被消化（质量评估通过）", fed >= 5, f"{fed} digested")

    by_domain = feed_engine.get_pending_samples_by_domain()
    zh_pending = len(by_domain.get("zh", []))
    print(f"  待消化样本按域: { {d: len(s) for d, s in by_domain.items()} }", flush=True)
    check("按域分类缓冲非空", zh_pending > 0, f"zh samples={zh_pending}")

    # ── 3. SleepEngine 接线 + Phase 2 训练 ──
    sleep_engine = SleepEngine(SleepConfig(training_enabled=True))
    sleep_engine.set_brain_interfaces(
        cortex=cortex,
        feed_engine=feed_engine,
        lifecycle=modules.get("lifecycle"),
        sleep_consolidator=modules.get("sleep_consolidator"),
        stdp_tracker=modules.get("stdp_tracker"),
    )

    before_snaps, before_emb = snapshot_learnable(cortex)
    report = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                         duration_seconds=0)
    print("\n[Phase 2] 睡眠训练（影子权重 COW）...", flush=True)
    try:
        sleep_engine._sleep_phase_model_training(report)
    except Exception as e:
        import traceback
        traceback.print_exc()
        check("Phase 2 训练无异常", False, f"{e}")
        print(f"  训练后推理检查跳过（训练异常），临时目录: {tmp_dir}", flush=True)
        return

    print(f"\n  样本消费: training_samples_used={report.training_samples_used}", flush=True)
    check("样本被消费", report.training_samples_used > 0,
          f"used={report.training_samples_used}")
    check("训练 loss 有限", report.training_loss is not None
          and abs(report.training_loss) < 1e6,
          f"loss={report.training_loss}")
    # 训练-训练互斥锁释放验证（finally 中 finish_training），保证后续睡眠周期可再训练
    from taiji.core.app_state import app_state
    check("训练锁已释放（可进入下一睡眠周期）", not app_state.is_training)

    # ── 4. 影子权重写回 live 验证 ──
    after_snaps, after_emb = snapshot_learnable(cortex)
    zh_changed = part_changed(before_snaps, after_snaps, "zh", "lm_head")
    check("zh 神经元 lm_head 权重已更新（写回 live）", zh_changed)
    emb_changed = before_emb is not None and after_emb is not None and any(
        k in after_emb and not torch.equal(before_emb[k], after_emb[k])
        for k in before_emb
    )
    check("shared_embedding 经验积累生效", emb_changed)

    # ── 5. ckpt 保存闭环 ──
    ckpt_path = os.path.join(tmp_dir, "cortex_state.pt")
    check("训练后自动保存 cortex_state.pt", os.path.exists(ckpt_path),
          f"path={ckpt_path}")
    if os.path.exists(ckpt_path):
        ckpt_size_mb = os.path.getsize(ckpt_path) / 1024 / 1024
        print(f"  ckpt 大小: {ckpt_size_mb:.1f} MB", flush=True)
        check("ckpt 非空", os.path.getsize(ckpt_path) > 1024)
        loaded = cortex.load_state(ckpt_path, strict=False)
        check("load_state 恢复成功", loaded)

    # ── 6. 训练后推理不崩 ──
    print("\n[推理] 训练后 generate ...", flush=True)
    try:
        out_code = cortex.generate(
            "Write a Python function to compute the Fibonacci sequence",
            max_tokens=40)
        print(f"  code 输出: {out_code[:80]!r}", flush=True)
        check("训练后 code 推理非空", bool(out_code and out_code.strip()))
    except Exception as e:
        check("训练后 code 推理不崩", False, f"{e}")

    try:
        out_zh = cortex.generate("请介绍神经网络的基本原理", max_tokens=40)
        print(f"  zh 输出: {out_zh[:80]!r}", flush=True)
        check("训练后 zh 推理不崩", True, f"len={len(out_zh)}")
    except Exception as e:
        check("训练后 zh 推理不崩", False, f"{e}")

    # ── 清理临时目录 ──
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    print("\n" + "=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL  "
          f"({time.time() - t0:.1f}s)", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
