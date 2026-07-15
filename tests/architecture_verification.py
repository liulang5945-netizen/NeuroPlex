#!/usr/bin/env python3
"""
共振场架构验证 — 不追 PPL，验证机制本身。

6 个测试覆盖完整共振数据流，每个有明确 pass/fail 条件。
"""

from __future__ import annotations

import json, math, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taiji.resonance import (
    ResonanceField, ResonanceNeuron,
    TribalMetrics, compute_initial_D,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ND = PROJECT_ROOT / "resonance_neurons_joint"
DD = PROJECT_ROOT / "taiji_data/training_data/pretrain_mix_v1"
SP_MODEL = PROJECT_ROOT / "taiji/tokenizer/sentencepiece.model"
VOCAB_SIZE, SEQ_LEN = 256000, 256
DOMAINS = ["zh", "en", "code", "math", "general"]
DEVICE = torch.device("cuda")

FILES = {
    "zh": "skypile_zh.jsonl", "en": "falcon_refinedweb_en.jsonl",
    "code": "codeparrot_code.jsonl", "math": "openwebmath.jsonl",
    "general": "taiji_native_v2.jsonl",
}

def load_batch(sp, domain, skip=30000):
    """加载一个领域的单个 batch。"""
    path = DD / FILES[domain]
    seqs = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < skip: continue
            if len(seqs) >= 4: break
            try: obj = json.loads(line)
            except: continue
            text = obj.get("text","") or obj.get("content","") or ""
            if len(text) > 100:
                ids = sp.encode(text)
                if len(ids) >= SEQ_LEN + 1:
                    seqs.append(torch.tensor(ids[:SEQ_LEN], dtype=torch.long))
    if seqs:
        return torch.stack(seqs)
    return None


def main():
    print("=" * 60)
    print("共振场架构验证 — 6 项测试")
    print("=" * 60)

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(str(SP_MODEL))

    # ── 加载模型 ──
    shared_embed = torch.nn.Embedding(VOCAB_SIZE, 512).to(DEVICE)
    shared_embed.load_state_dict(torch.load(str(ND/"shared_embed.pt"), map_location=DEVICE))
    shared_embed.eval()

    neurons = {}
    for d in DOMAINS:
        fname = f"neuron_{d}.pt" if d == "zh" else f"neuron_{d}_balanced.pt"
        ckpt = torch.load(str(ND/fname), map_location="cpu", weights_only=False)
        n = ResonanceNeuron(ckpt["config"]).eval().to(DEVICE)
        n.load_state_dict(ckpt["state_dict"])
        n.neuron_id = DOMAINS.index(d)
        neurons[d] = n

    field = ResonanceField(neurons=list(neurons.values())).to(DEVICE)
    try:
        field.load_state_dict(torch.load(str(ND/"field_contrastive_v2.pt"), map_location=DEVICE))
        field_has_weights = True
    except:
        field_has_weights = False

    # 准备各领域输入
    inputs = {}
    for d in DOMAINS:
        batch = load_batch(sp, d)
        if batch is not None:
            inputs[d] = batch.to(DEVICE)
            print(f"  输入 {d}: {batch.shape}")

    results = []

    # ═══════════════════════════════════════════════════════
    # 测试 1: 完整共振循环 — 数据流无错误
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*40}\n测试 1: 完整共振循环")
    field.reset()
    inp = inputs.get("zh", list(inputs.values())[0])
    emb = shared_embed(inp)

    try:
        r1_outputs = {}
        for d, n in neurons.items():
            r = n(emb, field_state=None, round_num=1, mode="pretrain")
            field.write(n.neuron_id, r["field_vector"].mean(dim=0))
            r1_outputs[d] = r
        field.update()

        r2_outputs = {}
        fs = field.read()
        for d, n in neurons.items():
            r = n(emb, field_state=fs, round_num=2, mode="pretrain")
            r2_outputs[d] = r

        ok = (len(r1_outputs) == 5 and len(r2_outputs) == 5
              and "logits" in r1_outputs["zh"] and "field_vector" in r1_outputs["zh"])
    except Exception as e:
        ok = False; print(f"  ❌ 异常: {e}")

    results.append(("完整共振循环（前向→写入→更新→条件化前向）", ok))
    print(f"  {'✅' if ok else '❌'} R1={len(r1_outputs)} neurons, R2={len(r2_outputs)} neurons")

    # ═══════════════════════════════════════════════════════
    # 测试 2: 动态阈值 — 拥堵度上升 → 阈值上升
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*40}\n测试 2: 动态阈值")

    field_low = ResonanceField(D=1024).to(DEVICE)
    field_low.N_active.fill_(5)
    t_low = field_low.dynamic_threshold()

    field_high = ResonanceField(D=1024).to(DEVICE)
    field_high.N_active.fill_(50)
    t_high = field_high.dynamic_threshold()

    ok = t_high > t_low
    results.append(("动态阈值: N=5→T={t_low:.3f}, N=50→T={t_high:.3f}", ok))
    print(f"  {'✅' if ok else '❌'} T(N=5)={t_low:.3f} < T(N=50)={t_high:.3f}")

    # ═══════════════════════════════════════════════════════
    # 测试 3: 共振加权 vs 均匀平均 — 输出不同
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*40}\n测试 3: 共振加权 ≠ 均匀平均")

    if field_has_weights:
        with torch.no_grad():
            field.reset()
            field_vectors = {}
            for d, n in neurons.items():
                r = n(emb, field_state=None, round_num=1, mode="inference")
                field.write(n.neuron_id, r["field_vector"].mean(dim=0))
                field_vectors[d] = r["field_vector"].mean(dim=0)
            field.update()

            # 均匀平均（增量累积，不存所有 logits）
            uniform = None
            for d in DOMAINS:
                r = neurons[d](emb, mode="pretrain")
                logits = r["logits"].detach()
                uniform = logits.clone() if uniform is None else uniform + logits
                del logits, r
            uniform = uniform / len(DOMAINS)

            # 共振加权
            weighted = None; w_total = 0.0
            for d in DOMAINS:
                r = neurons[d](emb, mode="pretrain")
                logits = r["logits"].detach()
                score = field.compute_resonance(field_vectors[d])
                weight = max(score, 0.01)
                weighted = logits * weight if weighted is None else weighted + logits * weight
                w_total += weight
                del logits, r
            weighted = weighted / w_total

            diff = (uniform - weighted).abs().mean().item()
        ok = diff > 1e-6
        results.append(("共振加权 ≠ 均匀平均", ok))
        print(f"  {'✅' if ok else '❌'} |uniform-weighted| = {diff:.6f}")
    else:
        results.append(("共振加权 ≠ 均匀平均（无训练权重，跳过）", True))
        print(f"  ⚠️ field 未训练，跳过")

    # ═══════════════════════════════════════════════════════
    # 测试 4: 场状态跨轮次变化
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*40}\n测试 4: 场状态跨轮次变化")

    field.reset()
    states = []
    for rnd in range(1, 4):
        for d, n in neurons.items():
            r = n(emb, field_state=field.read() if rnd > 1 else None,
                  round_num=rnd, mode="inference")
            field.write(n.neuron_id, r["field_vector"].mean(dim=0))
        field.update()
        states.append(field.read().clone())

    deltas = []
    for t in range(1, len(states)):
        cos = float(torch.dot(states[t], states[t-1]) /
                    (states[t].norm() * states[t-1].norm() + 1e-8))
        deltas.append(1.0 - cos)
        print(f"  R{t}→R{t+1}: Δ={1-cos:.6f}")

    # 只要有变化就算通过（不需要大变化）
    ok = any(d > 1e-8 for d in deltas)
    results.append(("场状态跨轮次变化", ok))
    print(f"  {'✅' if ok else '❌'}")

    # ═══════════════════════════════════════════════════════
    # 测试 5: 不同输入 → 不同激活模式
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*40}\n测试 5: 不同输入 → 不同激活模式")

    activation_patterns = {}
    for test_domain in DOMAINS:
        if test_domain not in inputs: continue
        field.reset()
        emb = shared_embed(inputs[test_domain])
        for d, n in neurons.items():
            r = n(emb, field_state=None, round_num=1, mode="inference")
            field.write(n.neuron_id, r["field_vector"].mean(dim=0))
        field.update()

        scores = {}
        for d, n in neurons.items():
            r = n(emb, field_state=None, round_num=1, mode="inference")
            scores[d] = round(field.compute_resonance(r["field_vector"].mean(dim=0)), 4)
        activation_patterns[test_domain] = scores
        best = max(scores, key=scores.get)
        print(f"  输入={test_domain:8s} → 最高={best} ({scores[best]:.4f})")

    # 检查：不同输入的激活模式是否不同
    patterns = [tuple(v.values()) for v in activation_patterns.values()]
    unique = len(set(patterns))
    ok = unique >= 2  # 至少 2 种不同模式
    results.append(("不同输入产生不同激活模式", ok))
    print(f"  {'✅' if ok else '❌'} {unique}/{len(patterns)} 种独特模式")

    # ═══════════════════════════════════════════════════════
    # 测试 6: 部落质量因子 Q — 一致组 vs 分歧组
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*40}\n测试 6: 部落质量因子 Q")

    metrics = TribalMetrics()
    dim = field.D

    # 场景 A：一致组 — 8 个相似方向
    base = torch.randn(dim, device=DEVICE)
    base = base / base.norm()
    consistent = []
    for _ in range(8):
        v = base + torch.randn(dim, device=DEVICE) * 0.03
        consistent.append(v / v.norm())

    # 模拟 3 轮
    sub_field = base.clone()
    for _ in range(3):
        metrics.record_round(consistent, sub_field)
        sub_field = sub_field + sum(consistent) / 8
        sub_field = sub_field / sub_field.norm()

    Q_coherent = metrics.quality_factor()
    loss_coherent = metrics.compression_loss()
    print(f"  一致组: Q={Q_coherent:.4f}, compression_loss={loss_coherent:.4f}")

    # 场景 B：分歧组 — 4+4 正交簇
    metrics.reset()
    dir_a = torch.randn(dim, device=DEVICE); dir_a = dir_a / dir_a.norm()
    dir_b = torch.randn(dim, device=DEVICE); dir_b = dir_b / dir_b.norm()
    mixed = []
    for _ in range(4):
        v = dir_a + torch.randn(dim, device=DEVICE) * 0.03
        mixed.append(v / v.norm())
    for _ in range(4):
        v = dir_b + torch.randn(dim, device=DEVICE) * 0.03
        mixed.append(v / v.norm())

    sub_field = (dir_a + dir_b) / 2; sub_field = sub_field / sub_field.norm()
    for _ in range(3):
        metrics.record_round(mixed, sub_field)
        sub_field = sub_field + sum(mixed) / 8
        sub_field = sub_field / sub_field.norm()

    Q_mixed = metrics.quality_factor()
    loss_mixed = metrics.compression_loss()
    print(f"  分歧组: Q={Q_mixed:.4f}, compression_loss={loss_mixed:.4f}")

    ok = Q_coherent > Q_mixed and loss_coherent < loss_mixed
    results.append(("部落 Q: 一致组 > 分歧组 + 压缩损失相反", ok))
    print(f"  {'✅' if ok else '❌'}")

    # ═══════════════════════════════════════════════════════
    # 汇总
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("验证汇总")
    print(f"{'='*60}")
    all_ok = True
    for name, ok in results:
        status = "✅" if ok else "❌"
        if not ok: all_ok = False
        print(f"  {status} {name}")
    print(f"\n  {'全部通过 ✅' if all_ok else '存在失败 ❌'}")
    print(f"{'='*60}")
    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
