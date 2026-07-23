"""诊断生成管线：隔离"模型argmax质量" vs "反馈回路保真度"。

三个测试：
  T1. Teacher-forcing argmax：喂训练文本的 general tokens，看每位置 argmax 是否=domain target
      → 若 argmax 准确率高，模型本身没问题，问题在反馈回路
      → 若 argmax 准确率低，模型 PPL 好但 argmax 差（分布太平），需改进模型/解码
  T2. Token 往返保真度：domain_id → decode → re-encode(general) → 是否能还原
      → 检查反馈回路是否丢失信息
  T3. 自回归生成追踪：打印每步的 argmax token、其概率、解码文本
      → 看模型从哪一步开始"跑偏"

Usage:
    python -u scripts/training/diagnose_generation.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from taiji.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble,
    get_domain_neuron_config,
)
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.train_neuron import (
    load_domain_texts, load_domain_tokenizer, load_general_tokenizer,
    OUTPUT_DIR,
)

DOMAIN = "zh"


def load_joint_ensemble(n_neurons, domain, device):
    cfg = get_domain_neuron_config(domain)
    neurons = {}
    for i in range(n_neurons):
        nid = f"{domain}_j{i}"
        path = os.path.join(OUTPUT_DIR, f"neuron_{nid}.pt")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        neuron = ResonanceNeuron(cfg).to(device)
        neuron.load_state_dict(ckpt["state_dict"], strict=False)
        neuron.eval()
        neurons[nid] = neuron
    emb_path = os.path.join(OUTPUT_DIR, "shared_embedding_joint.pt")
    shared_embedding = torch.nn.Embedding(256000, 512)
    shared_embedding.load_state_dict(torch.load(emb_path, map_location=device))
    shared_embedding.to(device).eval()
    field = ResonanceField(dim=cfg.field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=1)
    return neurons, shared_embedding, ensemble, cfg


def test_teacher_forcing(neurons, shared_embedding, ensemble, domain_sp, general_sp, device, n_texts=20):
    """T1: Teacher-forcing argmax 准确率。"""
    print("\n" + "=" * 70, flush=True)
    print("[T1] Teacher-forcing argmax 准确率（协作）", flush=True)
    print("=" * 70, flush=True)
    texts = load_domain_texts(DOMAIN, max_texts=n_texts)
    texts = texts[-n_texts:]

    total = 0
    correct = 0
    top5_correct = 0
    with torch.no_grad():
        for text in texts:
            shared_emb, targets, mask = batch_align_and_embed(
                [text], domain_sp, general_sp, shared_embedding,
            )
            shared_emb = shared_emb.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            result = ensemble.forward_train(shared_emb, temperature=1.0)
            logits = result["fused_logits"]  # [1, L, V_domain]
            # next-token: position t predicts t+1
            shift_logits = logits[:, :-1, :]  # [1, L-1, V]
            shift_targets = targets[:, 1:]     # [1, L-1]
            shift_mask = mask[:, 1:]           # [1, L-1]
            valid = shift_mask & (shift_targets != -100)
            preds = shift_logits.argmax(dim=-1)  # [1, L-1]
            correct += (preds[valid] == shift_targets[valid]).sum().item()
            total += valid.sum().item()
            # top-5
            top5 = shift_logits.topk(5, dim=-1).indices  # [1, L-1, 5]
            top5_correct += sum(
                shift_targets[0, j].item() in top5[0, j].tolist()
                for j in range(shift_targets.size(1))
                if valid[0, j]
            )

    acc = correct / max(total, 1) * 100
    top5_acc = top5_correct / max(total, 1) * 100
    print(f"  样本数: {len(texts)}, 有效 token: {total}", flush=True)
    print(f"  argmax top-1 准确率: {acc:.1f}%", flush=True)
    print(f"  argmax top-5 准确率: {top5_acc:.1f}%", flush=True)

    # 展示一个样本的 argmax vs target
    text = texts[0]
    shared_emb, targets, mask = batch_align_and_embed(
        [text], domain_sp, general_sp, shared_embedding,
    )
    with torch.no_grad():
        result = ensemble.forward_train(shared_emb.to(device), temperature=1.0)
        logits = result["fused_logits"]
        shift_logits = logits[:, :-1, :]
        shift_targets = targets[:, 1:].to(device)
        preds = shift_logits.argmax(dim=-1)[0]
        tgts = shift_targets[0]
    print(f"\n  样本前 15 位置 (argmax → target):", flush=True)
    for j in range(min(15, len(tgts))):
        if tgts[j].item() == -100:
            continue
        pred_piece = domain_sp.IdToPiece(preds[j].item()) if preds[j].item() < domain_sp.GetPieceSize() else "?"
        tgt_piece = domain_sp.IdToPiece(tgts[j].item()) if tgts[j].item() < domain_sp.GetPieceSize() else "?"
        # 概率
        probs = F.softmax(shift_logits[0, j], dim=-1)
        pred_prob = probs[preds[j]].item()
        print(f"    pos{j}: argmax={preds[j].item()}({pred_piece}) p={pred_prob:.3f}  "
              f"target={tgts[j].item()}({tgt_piece})", flush=True)

    return acc


def test_token_roundtrip(domain_sp, general_sp, n=200):
    """T2: domain_id → decode → re-encode(general) 往返保真度。"""
    print("\n" + "=" * 70, flush=True)
    print("[T2] Token 往返保真度（domain → decode → general re-encode）", flush=True)
    print("=" * 70, flush=True)
    # 检查 domain token 0..n 往返后能否还原
    exact = 0
    multi = 0  # 一个 domain token 变多个 general token
    empty = 0
    sample_mismatches = []
    for did in range(min(n, domain_sp.GetPieceSize())):
        piece = domain_sp.IdToPiece(did)
        text = piece.replace("▁", " ")
        gen_ids = general_sp.EncodeAsIds(text)
        # 反向：general ids → decode → domain encode
        if not gen_ids:
            empty += 1
            continue
        gen_text = general_sp.DecodeIds(gen_ids)
        back_ids = domain_sp.EncodeAsIds(gen_text)
        if len(gen_ids) > 1:
            multi += 1
        if back_ids == [did]:
            exact += 1
        elif len(sample_mismatches) < 5:
            sample_mismatches.append((did, piece, text, gen_ids, gen_text, back_ids))

    print(f"  检查 {min(n, domain_sp.GetPieceSize())} 个 domain token:", flush=True)
    print(f"  往返精确还原: {exact} ({exact/n*100:.1f}%)", flush=True)
    print(f"  一对多(domain→多general): {multi} ({multi/n*100:.1f}%)", flush=True)
    print(f"  空: {empty}", flush=True)
    if sample_mismatches:
        print(f"  不匹配样本:", flush=True)
        for did, piece, text, gen_ids, gen_text, back_ids in sample_mismatches:
            print(f"    did={did} piece={piece!r} text={text!r} → gen_ids={gen_ids} "
                  f"gen_text={gen_text!r} back_ids={back_ids}", flush=True)
    return exact / n


def test_autoregressive_trace(neurons, shared_embedding, ensemble, domain_sp, general_sp, device, prompt="你好，请介绍一下自己"):
    """T3: 自回归生成逐步追踪。"""
    print("\n" + "=" * 70, flush=True)
    print(f"[T3] 自回归生成追踪 (prompt: {prompt})", flush=True)
    print("=" * 70, flush=True)
    general_ids = general_sp.EncodeAsIds(prompt)
    ids = torch.tensor([general_ids], dtype=torch.long, device=device)
    print(f"  初始 general_ids: {general_ids}", flush=True)
    print(f"  初始解码: {general_sp.DecodeIds(general_ids)!r}", flush=True)
    print(f"\n  逐步生成:", flush=True)

    generated_domain = []
    with torch.no_grad():
        for step in range(30):
            shared_emb = shared_embedding(ids)
            result = ensemble.forward_train(shared_emb, temperature=1.0)
            logits = result["fused_logits"][:, -1, :]  # [1, V]
            probs = F.softmax(logits[0], dim=-1)
            next_token = logits.argmax(dim=-1).item()
            pred_prob = probs[next_token].item()
            piece = domain_sp.IdToPiece(next_token) if next_token < domain_sp.GetPieceSize() else "?"
            text_piece = piece.replace("▁", " ")
            gen_ids_new = general_sp.EncodeAsIds(text_piece)
            print(f"    step{step}: domain_tok={next_token}({piece!r}) p={pred_prob:.3f} "
                  f"→ gen_ids={gen_ids_new} text={text_piece!r}", flush=True)
            generated_domain.append(next_token)
            if gen_ids_new:
                ids = torch.cat([ids, torch.tensor([gen_ids_new], dtype=torch.long, device=device)], dim=1)
            else:
                print(f"      ⚠️ 空 gen_ids，停止", flush=True)
                break
    print(f"\n  最终生成(domain decode): {domain_sp.DecodeIds(generated_domain)!r}", flush=True)


def main():
    device = "cpu"
    print("加载联合训练 ensemble...", flush=True)
    neurons, shared_embedding, ensemble, cfg = load_joint_ensemble(5, "zh", device)
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()

    acc = test_teacher_forcing(neurons, shared_embedding, ensemble, domain_sp, general_sp, device)
    rt = test_token_roundtrip(domain_sp, general_sp)

    print("\n" + "-" * 70, flush=True)
    print("诊断结论:", flush=True)
    print(f"  T1 argmax top-1 准确率: {acc:.1f}%", flush=True)
    print(f"  T2 往返保真度: {rt*100:.1f}%", flush=True)
    if acc < 30:
        print(f"  → argmax 准确率低：模型 PPL 好但分布太平，问题在模型/解码策略", flush=True)
    else:
        print(f"  → argmax 准确率尚可：问题在反馈回路", flush=True)
    if rt < 0.9:
        print(f"  → 往返保真度低：反馈回路丢失信息", flush=True)

    test_autoregressive_trace(neurons, shared_embedding, ensemble, domain_sp, general_sp, device)


if __name__ == "__main__":
    main()
