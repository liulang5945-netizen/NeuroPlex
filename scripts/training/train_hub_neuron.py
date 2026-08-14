#!/usr/bin/env python3
"""Hub Neuron 训练（缺口 L 阶段 2，2026-08-14）。

参考人脑联合皮层：hub 是 high-capacity、全词表（general 256K）、可生成的
跨域联合 neuron（草案决策 1B + 2B）——expert 规格（hidden=1024, 14 层,
field_dim=4096）+ general lm_head 256K。输入 = general 256K 空间（shared
embedding），目标 = general 空间 next-token（同空间 SFT，无需跨词表对齐）。

数据（跨域混合，hub 学会"跨域共享子空间"）：
1. 域 SFT 混合：zh/code/en/math/general sft_pt（每域采样）
2. 跨域平行语料：data/cross_domain_pairs.jsonl（zh 指令 ↔ code 实现）——
   本身即"中文指令 → 代码"跨域 SFT 对，直接训练 hub 跨域生成能力

训练：SFT answer masking（response 位置算 loss），general tokenizer 编码，
shared_embedding 冻结（保留全局 token 语义），lr 5e-4（domain SFT 成功配方）。

工程保障（用户规则：训练前检查 checkpoint 能正确保存）：
- 每 checkpoint 保存后立即回读重算 answer-masked val PPL（verify_checkpoint）
- --smoke 模式：极小步数快速验证链路（装配/保存/回读/收敛）

产物：data/hub_neuron/neuron_hub.pt（config + state_dict + lm_head）

Usage:
    python -u scripts/training/train_hub_neuron.py --smoke
    python -u scripts/training/train_hub_neuron.py --epochs 1 --max-steps 5000 --device cuda --out-name neuron_hub_formal
    python -u scripts/training/train_hub_neuron.py --resume --epochs 1 --max-steps 5000 --out-name neuron_hub_formal
    # 日志落盘（N3 规范）:
    #   python -u scripts/training/train_hub_neuron.py --epochs 1 2>&1 | Tee-Object -FilePath logs\train_hub_$(Get-Date -Format yyyyMMdd_HHmmss).log
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from taiji.resonance import ResonanceNeuron
from taiji.resonance.config import get_default_neuron_config
from scripts.training.utils import load_general_tokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "hub_neuron")
SFT_DIR = os.path.join(PROJECT_ROOT, "data", "sft")
PAIRS_PATH = os.path.join(PROJECT_ROOT, "data", "cross_domain_pairs.jsonl")

SEQ_LEN = 192
BATCH_SIZE = 4
LR = 5e-4
WEIGHT_DECAY = 0.1
EMBED_DIM = 512
GENERAL_VOCAB = 256000
# 跨域混合采样配额（域 SFT 全量太大，CPU 预算按比例采样）
DOMAIN_QUOTA = {"zh": 8000, "code": 8000, "en": 6000, "math": 6000, "general": 3000}
PAIRS_QUOTA = 3000  # 平行语料全量 1629 对 < 配额


# ======================== 数据 ========================


def load_sft(domain: str) -> List[dict]:
    path = os.path.join(SFT_DIR, f"{domain}_sft.pt")
    data = torch.load(path, map_location="cpu", weights_only=False)
    return list(data)


def load_pairs() -> List[dict]:
    """平行语料 → SFT 样本：zh 指令 → code 响应（天然跨域 QA 对）。"""
    out = []
    if not os.path.exists(PAIRS_PATH):
        return out
    with open(PAIRS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            out.append({"instruction": p["zh"], "input": "", "response": p["code"]})
    return out


def build_mixed_samples(args) -> List[dict]:
    """跨域混合样本：域 SFT 采样 + 平行语料（zh 指令→code 响应）。"""
    samples: List[dict] = []
    for domain, quota in DOMAIN_QUOTA.items():
        try:
            d = load_sft(domain)
            random.shuffle(d)
            samples.extend(d[:quota])
            print(f"  [{domain}] {len(d)} 条 → 采样 {min(quota, len(d))}", flush=True)
        except Exception as e:
            print(f"  [{domain}] 跳过: {e}", flush=True)
    pairs = load_pairs()
    if pairs:
        random.shuffle(pairs)
        samples.extend(pairs[:PAIRS_QUOTA])
        print(f"  [pairs] {len(pairs)} 条平行语料 → {min(PAIRS_QUOTA, len(pairs))} 条跨域样本", flush=True)
    print(f"  混合样本总数: {len(samples)}", flush=True)
    return samples


def build_sample(text: str, prompt: str, general_sp) -> Tuple[List[int], int]:
    """general 同空间样本：g_ids + answer 起始位置（prompt+"\n" 前缀匹配）。"""
    g_ids = general_sp.encode(text)
    p_ids = general_sp.encode(prompt + "\n")
    k = 0
    n = min(len(p_ids), len(g_ids))
    while k < n and int(g_ids[k]) == int(p_ids[k]):
        k += 1
    return g_ids, k


def build_batch(samples, general_sp, shared_emb) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """批量样本：emb + targets（非 answer 为 -100）+ attn mask。"""
    try:
        eos = general_sp.eos_id()
    except Exception:
        eos = 1
    rows = []
    for s in samples:
        instruction = str(s.get("instruction", "")).strip()
        inp = str(s.get("input", "")).strip()
        response = str(s.get("response", "")).strip()
        if not instruction or not response:
            continue
        prompt = instruction + (("\n" + inp) if inp else "")
        text = prompt + "\n" + response
        g_ids, ans_start = build_sample(text, prompt, general_sp)
        g_ids = g_ids[: SEQ_LEN - 1] + [eos]
        rows.append((g_ids, ans_start))
    if not rows:
        return None, None, None
    max_len = max(len(r[0]) for r in rows)
    B = len(rows)
    padded_ids = torch.zeros(B, max_len, dtype=torch.long)
    targets = torch.full((B, max_len), -100, dtype=torch.long)
    attn = torch.zeros(B, max_len, dtype=torch.bool)
    for b, (g_ids, ans_start) in enumerate(rows):
        L = len(g_ids)
        padded_ids[b, :L] = torch.tensor(g_ids, dtype=torch.long)
        targets[b, ans_start:L] = torch.tensor(g_ids[ans_start:L], dtype=torch.long)
        attn[b, :L] = True
    emb = shared_emb(padded_ids)
    return emb, targets, attn


# ======================== 模型 ========================


def build_hub_neuron(device: str) -> Tuple[ResonanceNeuron, torch.nn.Embedding]:
    """hub = EXPERT 规格 + general 256K lm_head（从零训练，无基座）。"""
    cfg = get_default_neuron_config("expert")
    cfg.vocab_size = GENERAL_VOCAB
    cfg.neuron_id = "hub"
    # 平行语料含代码/中文混合——hub 应同时具备 general 判定与生成
    cfg.unified_field_dim = None
    neuron = ResonanceNeuron(cfg).to(device)
    # 共享 embedding 冻结（保留 general token 语义）
    emb = torch.nn.Embedding(GENERAL_VOCAB, EMBED_DIM)
    emb_path = os.path.join(PROJECT_ROOT, "data", "shared_embedding.pt")
    if os.path.exists(emb_path):
        w = torch.load(emb_path, map_location=device, weights_only=True)
        if isinstance(w, dict):
            w = w["weight"]
        if w.shape == (GENERAL_VOCAB, EMBED_DIM):
            emb.weight.data.copy_(w)
    emb.to(device)
    emb.weight.requires_grad = False
    n_params = sum(p.numel() for p in neuron.parameters())
    print(f"  hub neuron: {n_params / 1e6:.0f}M 参数（EXPRT hidden=1024, lm_head 256K）", flush=True)
    return neuron, emb


# ======================== 保存 + 回读验证（用户规则） ========================


def verify_checkpoint(eval_samples: List[dict], general_sp, emb, n_check: int = 4) -> float:
    """保存后立即回读：重算 answer-masked val PPL，防坏 checkpoint。"""
    ckpt = torch.load(os.path.join(OUT_DIR, "neuron_hub.pt"), map_location="cpu", weights_only=False)
    cfg = ckpt["neuron_config"]
    cfg.unified_field_dim = None
    neuron = ResonanceNeuron(cfg)
    neuron.load_state_dict(ckpt["state_dict"], strict=False)
    neuron.eval()
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for _ in range(n_check):
            batch = random.sample(eval_samples, min(BATCH_SIZE, len(eval_samples)))
            e, y, m = build_batch(batch, general_sp, emb)
            if e is None:
                continue
            r = neuron.forward(e, return_logits=True)
            lg = r["logits"][:, :-1, :].contiguous()
            tgt = y[:, 1:].clone().contiguous()
            am = m[:, 1:].contiguous()
            tgt[~am] = -100
            nt = am.sum().item()
            l = F.cross_entropy(lg.view(-1, lg.size(-1)), tgt.view(-1),
                                ignore_index=-100, reduction="sum")
            total_loss += l.item()
            total_tok += max(nt, 1)
    avg = total_loss / max(total_tok, 1)
    ppl = math.exp(min(avg, 20))
    print(f"  [verify] hub 回读 answer PPL={ppl:.1f}", flush=True)
    return avg


def save_checkpoint(neuron, shared_emb, step: int, ppl: Optional[float], loss_history: list,
                    out_name: str = "neuron_hub"):
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt = {
        "neuron_config": neuron.config,
        "state_dict": neuron.state_dict(),
        "domain": "general",
        "neuron_id": "hub",
        "step": step,
        "result": {"best_ppl": ppl, "best_step": step, "steps": step},
        "loss_history": loss_history,
        "hub_neuron": True,  # 标记：hub 联合皮层 neuron
        "saved_at": datetime.now().isoformat(),
    }
    torch.save(ckpt, os.path.join(OUT_DIR, f"{out_name}.pt"))
    torch.save(shared_emb.weight.data.clone(), os.path.join(OUT_DIR, "shared_embedding.pt"))
    # N3：训练历史 JSON 落盘 logs/（与 verify 日志规范一致）
    try:
        os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
        hist_path = os.path.join(PROJECT_ROOT, "logs", f"{out_name}_history.json")
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump({"steps": loss_history, "last_step": step, "best_ppl": ppl}, f, ensure_ascii=False)
    except Exception:
        pass


# ======================== 训练 ========================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="极小步数验证链路")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-every", type=int, default=500,
                        help="ckpt 保存间隔（步）；正式训练默认 500，smoke 强制 5")
    parser.add_argument("--out-name", default=None,
                        help="产物文件名（不含 .pt）；默认：正式=neuron_hub / smoke=neuron_hub_smoke（防覆盖正式产物）")
    parser.add_argument("--resume", action="store_true",
                        help="从现有 ckpt 恢复 state_dict+loss_history 续训（步数预算重新计算）")
    args = parser.parse_args()
    if args.out_name is None:
        args.out_name = "neuron_hub_smoke" if args.smoke else "neuron_hub"

    t0 = time.time()
    print("=" * 60, flush=True)
    print("Hub Neuron 训练（缺口 L 阶段 2：expert 规格 + general 256K）", flush=True)
    print("=" * 60, flush=True)

    random.seed(0)
    torch.manual_seed(0)
    device = args.device
    general_sp = load_general_tokenizer()

    samples = build_mixed_samples(args)
    if not samples:
        print("无训练样本，退出", flush=True)
        sys.exit(1)
    random.shuffle(samples)
    # 训练/验证划分
    n_val = max(8, len(samples) // 20)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    print(f"  训练 {len(train_samples)} / 验证 {len(val_samples)}", flush=True)

    neuron, shared_emb = build_hub_neuron(device)
    optimizer = torch.optim.AdamW(
        [p for p in neuron.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_samples) // BATCH_SIZE)

    loss_history = []
    if args.resume:
        resume_path = os.path.join(OUT_DIR, f"{args.out_name}.pt")
        if not os.path.exists(resume_path):
            print(f"--resume 但产物不存在: {resume_path}，改为从零训练", flush=True)
        else:
            old = torch.load(resume_path, map_location="cpu", weights_only=False)
            neuron.load_state_dict(old.get("state_dict", {}), strict=False)
            loss_history = list(old.get("loss_history", []))
            print(f"[resume] 恢复 {len(loss_history)} 步历史（权重已加载，预算重计）", flush=True)

    max_steps = args.max_steps if not args.smoke else 2
    total_steps = min(max_steps, steps_per_epoch * args.epochs)
    save_every = 5 if args.smoke else args.save_every
    print(f"  预算: {total_steps} 步（--smoke={args.smoke}），每 {save_every} 步保存到 {args.out_name}.pt", flush=True)

    neuron.train()
    loss_history = []
    step = 0
    best_ppl = None
    for epoch in range(args.epochs):
        random.shuffle(train_samples)
        for i in range(0, len(train_samples), BATCH_SIZE):
            if step >= total_steps:
                break
            batch = train_samples[i:i + BATCH_SIZE]
            e, y, m = build_batch(batch, general_sp, shared_emb)
            if e is None:
                continue
            optimizer.zero_grad()
            r = neuron.forward(e, return_logits=True)
            lg = r["logits"][:, :-1, :].contiguous()
            tgt = y[:, 1:].clone().contiguous()
            am = m[:, 1:].contiguous()
            tgt[~am] = -100
            nt = am.sum().item()
            loss = F.cross_entropy(lg.view(-1, lg.size(-1)), tgt.view(-1),
                                   ignore_index=-100, reduction="mean")
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.item()))
            step += 1
            if step % save_every == 0 or step == total_steps:
                print(f"  step {step}/{total_steps} loss={float(loss.item()):.3f}"
                      f" ({time.time() - t0:.0f}s)", flush=True)
                # 用户规则：周期保存 + 回读验证（先保存再回读，防坏 checkpoint）
                save_checkpoint(neuron, shared_emb, step, None, loss_history, out_name=args.out_name)
                ppl = verify_checkpoint(val_samples, general_sp, shared_emb)
                best_ppl = ppl if best_ppl is None else min(best_ppl, ppl)
                if args.smoke:
                    break
        if step >= total_steps:
            break

    # 最终保存 + 回读（先保存再回读）
    save_checkpoint(neuron, shared_emb, step, best_ppl, loss_history, out_name=args.out_name)
    ppl = verify_checkpoint(val_samples, general_sp, shared_emb)
    print(f"\n  完成: {step} 步, best val PPL={best_ppl:.1f}", flush=True)
    print(f"  产物: {os.path.join(OUT_DIR, args.out_name + '.pt')}", flush=True)
    print(f"  总耗时: {time.time() - t0:.0f}s", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
