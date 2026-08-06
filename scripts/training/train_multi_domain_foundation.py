"""多域基座训练：为跨域协作层提供可靠的域神经元 + 共享 embedding。

背景（2026-08-06 诊断）：
- verify_v3 多域神经元不可用：train_neurons_from_scratch 用域编码输入 +
  随机初始化的共享 embedding（训练时未保存）→ 评估时任何管线 PPL 均≈随机。
- 对话管线（finetune_cross_spec，PPL 2.2 可复现）证明正确配方是：
  输入 general_sp 编码 → 共享 embedding → neuron；标签 = 域 tokenizer。
- 冒烟验证：对话 embedding 基座 + code 数据，code neuron 500 步 loss 9.4→2.7。

本脚本按对话配方重训各域 neuron，并联合训练共享 embedding（对话基座 warm-start），
输出与 train_cross_domain_collab.py 兼容的目录格式：
  {save_dir}/neuron_{domain}.pt   （neuron_config + state_dict + shared_embedding_state + result）
  {save_dir}/shared_embedding.pt  （Tensor 256000×512）

Usage:
    python -u scripts/training/train_multi_domain_foundation.py \
        --domains code,math,zh,en --steps-per-domain 600 --save-dir data/foundation_v1
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from taiji.resonance.config import get_domain_neuron_config
from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.utils import load_general_tokenizer
from scripts.training.train_cross_domain_collab import load_tokenizer_for_vocab
from scripts.training.experiment_config import SFT_ANSWER_MARKER

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
DIALOGUE_EMB_SRC = "data/neurons/neuron_zh_std0_dialogue.pt"
SEQ_LEN = 96
SEED = 0


def load_dialogue_embedding() -> torch.nn.Embedding:
    """warm-start：从对话 neuron ckpt 的 shared_embedding_state 加载基座 embedding。"""
    emb = torch.nn.Embedding(256000, 512)
    ckpt = torch.load(DIALOGUE_EMB_SRC, map_location="cpu", weights_only=False)
    if "shared_embedding_state" in ckpt and ckpt["shared_embedding_state"] is not None:
        emb.weight.data.copy_(ckpt["shared_embedding_state"]["weight"])
        print(f"  [embedding] warm-start from {DIALOGUE_EMB_SRC}", flush=True)
    else:
        print("  [embedding] ⚠️ 对话基座无 shared_embedding_state，随机初始化", flush=True)
    return emb


def load_domain_texts(domain: str, max_texts: int) -> List[str]:
    path = f"data/sft/{domain}_sft.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    texts = [d["full"] for d in data]
    if max_texts > 0:
        texts = texts[:max_texts]
    print(f"  [{domain}] {len(texts)} 条 SFT 文本", flush=True)
    return texts


def save_foundation(save_dir: str, domain: str, step: int, neuron: ResonanceNeuron,
                    shared_emb: torch.nn.Embedding, loss_history: list):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "neuron_config": neuron.config,
        "state_dict": neuron.state_dict(),
        "shared_embedding_state": {"weight": shared_emb.weight.data.clone()},
        "domain": domain,
        "step": step,
        "result": {"best_step": step, "steps": step},
        "saved_at": datetime.now().isoformat(),
    }
    torch.save(ckpt, os.path.join(save_dir, f"neuron_{domain}.pt"))
    torch.save(shared_emb.weight.data.clone(), os.path.join(save_dir, "shared_embedding.pt"))
    return os.path.join(save_dir, f"neuron_{domain}.pt")


def save_best_foundation(save_dir: str, domain: str, step: int, neuron: ResonanceNeuron,
                         shared_emb: torch.nn.Embedding, ppl: float):
    """保存每域最优权重（早停选择）+ 该步 embedding 快照。

    评估/协作层用 shared_embedding.pt（最终），此处额外保存每域最优 embedding
    快照（shared_embedding_best_{domain}.pt）用于配对校验。
    """
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "neuron_config": neuron.config,
        "state_dict": neuron.state_dict(),
        "shared_embedding_state": {"weight": shared_emb.weight.data.clone()},
        "domain": domain,
        "step": step,
        "result": {"best_ppl": ppl, "best_step": step, "steps": step},
        "saved_at": datetime.now().isoformat(),
    }
    torch.save(ckpt, os.path.join(save_dir, f"neuron_{domain}.pt"))
    torch.save(shared_emb.weight.data.clone(),
               os.path.join(save_dir, f"shared_embedding_best_{domain}.pt"))
    return os.path.join(save_dir, f"neuron_{domain}.pt")


def verify_checkpoint(save_dir: str, domain: str, sp, general_sp,
                      shared_emb: torch.nn.Embedding, texts: List[str],
                      n_check: int = 8, embed_path: Optional[str] = None) -> float:
    """保存后立即回读，验证 neuron + embedding 能复现低 PPL（防坏 checkpoint）。

    embed_path 为 None 时用共享 embedding（collab/eval 口径）；否则用该步 embedding
    快照（配对校验：best 权重 ↔ best 步 embedding）。
    """
    ckpt = torch.load(os.path.join(save_dir, f"neuron_{domain}.pt"),
                      map_location="cpu", weights_only=False)
    cfg = ckpt["neuron_config"]
    cfg.unified_field_dim = None
    n = ResonanceNeuron(cfg)
    n.load_state_dict(ckpt["state_dict"], strict=False)
    emb = torch.nn.Embedding(256000, 512)
    src = embed_path if embed_path else os.path.join(save_dir, "shared_embedding.pt")
    emb.weight.data.copy_(torch.load(src, map_location="cpu", weights_only=False))
    n.eval()
    answer_marker = SFT_ANSWER_MARKER if domain == "zh" else None
    marker_mode = "last" if answer_marker else "first"
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for t in random.sample(texts, min(n_check, len(texts))):
            out = batch_align_and_embed([t], sp, general_sp, emb,
                                        max_seq_len=SEQ_LEN, answer_marker=answer_marker,
                                        answer_marker_mode=marker_mode)
            x, y, m = out[0], out[1], out[2]
            sft_mask = out[3] if len(out) > 3 else None
            r = n.forward(x, return_logits=True)
            sl, st = r["logits"][:, :-1, :].contiguous(), y[:, 1:].clone().contiguous()
            sm = m[:, 1:].contiguous()
            st = st.clone()
            if sft_mask is not None:
                ss = sft_mask[:, 1:].contiguous()
                st[~(sm & ss)] = -100
                nt = (sm & ss).sum().item()
            else:
                st[~sm] = -100
                nt = sm.sum().item()
            l = F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1),
                                ignore_index=-100, reduction="sum")
            total_loss += l.item()
            total_tok += max(nt, 1)
    avg = total_loss / max(total_tok, 1)
    print(f"  [verify] {domain} 回读 PPL={math.exp(min(avg, 20)):.1f}", flush=True)
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="code,math,zh,en")
    parser.add_argument("--steps-per-domain", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--lr", type=float, default=3e-4, help="neuron 学习率")
    parser.add_argument("--embed-lr", type=float, default=1e-4, help="shared embedding 学习率")
    parser.add_argument("--max-texts", type=int, default=3000)
    parser.add_argument("--save-dir", default="data/foundation_v1")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    assert len(domains) >= 2
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"foundation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    print("=" * 60, flush=True)
    print(f"多域基座训练（对话配方：general 编码输入 + 域标签）", flush=True)
    print(f"domains={domains} steps/domain={args.steps_per_domain} "
          f"batch={args.batch_size} lr={args.lr} embed_lr={args.embed_lr}", flush=True)
    print(f"log: {log_path}", flush=True)

    general_sp = load_general_tokenizer()
    domain_sps = {d: load_tokenizer_for_vocab(d, get_domain_neuron_config(d, spec="compact").vocab_size)
                  for d in domains}
    texts = {d: load_domain_texts(d, args.max_texts) for d in domains}

    shared_emb = load_dialogue_embedding()
    neurons = {}
    for d in domains:
        cfg = get_domain_neuron_config(d, spec="compact")
        cfg.unified_field_dim = None
        neurons[d] = ResonanceNeuron(cfg)

    # 优化器：neuron 主体 lr，embedding 独立低 lr（联合训练共享感知层）
    opt_neurons = {d: torch.optim.AdamW(neurons[d].parameters(), lr=args.lr, weight_decay=0.01)
                   for d in domains}
    opt_emb = torch.optim.AdamW(shared_emb.parameters(), lr=args.embed_lr, weight_decay=0.0)

    total_steps = args.steps_per_domain * len(domains)
    step = 0
    loss_history: list = []
    best_ppl: Dict[str, float] = {d: float("inf") for d in domains}
    best_step: Dict[str, int] = {d: 0 for d in domains}
    t0 = time.time()

    for dom_idx in range(args.steps_per_domain):
        for d in domains:  # 域轮转：每步一个域
            step += 1
            batch = random.sample(texts[d], args.batch_size)
            answer_marker = SFT_ANSWER_MARKER if d == "zh" else None
            marker_mode = "last" if answer_marker else "first"
            out = batch_align_and_embed(batch, domain_sps[d], general_sp, shared_emb,
                                        max_seq_len=args.seq_len, answer_marker=answer_marker,
                                        answer_marker_mode=marker_mode)
            x, y, m = out[0], out[1], out[2]
            sft_mask = out[3] if len(out) > 3 else None

            r = neurons[d].forward(x, return_logits=True)
            sl, st = r["logits"][:, :-1, :].contiguous(), y[:, 1:].clone().contiguous()
            sm = m[:, 1:].contiguous()
            st = st.clone()
            if sft_mask is not None:
                ss = sft_mask[:, 1:].contiguous()
                st[~(sm & ss)] = -100
                nt = max((sm & ss).sum().item(), 1)
            else:
                st[~sm] = -100
                nt = max(sm.sum().item(), 1)
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1),
                                   ignore_index=-100, reduction="sum") / nt

            opt_neurons[d].zero_grad()
            opt_emb.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(neurons[d].parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(shared_emb.parameters(), 1.0)
            opt_neurons[d].step()
            opt_emb.step()

            ppl = math.exp(min(loss.item(), 20))
            loss_history.append({"step": step, "domain": d, "loss": loss.item(), "ppl": ppl})
            if ppl < best_ppl[d]:
                best_ppl[d] = ppl
                best_step[d] = step
                # 早停选择：保存该域最优权重 + 该步 embedding 快照
                save_best_foundation(args.save_dir, d, step, neurons[d], shared_emb, ppl)

            if step % 50 == 0:
                elapsed = time.time() - t0
                print(f"  step {step}/{total_steps} [{d}] loss={loss.item():.3f} "
                      f"PPL={ppl:.1f} best[{d}]={best_ppl[d]:.1f}@{best_step[d]} ({elapsed:.0f}s)", flush=True)

            # 周期保存最终 embedding（训练全程共享，供 eval 使用）
            if step % 400 == 0:
                torch.save(shared_emb.weight.data.clone(),
                           os.path.join(args.save_dir, "shared_embedding.pt"))

    # 最终：保存最终 embedding 为规范版本（collab/eval 用），并全域回读验证
    torch.save(shared_emb.weight.data.clone(), os.path.join(args.save_dir, "shared_embedding.pt"))
    print("\n[最终] 回读验证（best 权重 + 最终 embedding，collab/eval 口径）：", flush=True)
    for d in domains:
        verify_checkpoint(args.save_dir, d, domain_sps[d], general_sp, shared_emb, texts[d])
    print("[配对校验] best 权重 + 各自 best 步 embedding 快照：", flush=True)
    for d in domains:
        p = os.path.join(args.save_dir, f"shared_embedding_best_{d}.pt")
        if os.path.exists(p):
            verify_checkpoint(args.save_dir, d, domain_sps[d], general_sp, shared_emb,
                              texts[d], embed_path=p)

    hist_path = os.path.join(LOG_DIR, "foundation_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        import json
        json.dump(loss_history, f, ensure_ascii=False)
    print(f"\n[完成] best PPL: {best_ppl}", flush=True)
    print(f"  checkpoint: {args.save_dir}/", flush=True)
    print(f"  历史: {hist_path} ({len(loss_history)} 条)", flush=True)


if __name__ == "__main__":
    main()
