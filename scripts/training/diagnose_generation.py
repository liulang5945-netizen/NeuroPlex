"""诊断 PPL 低但生成乱码的根因。

检查：
1. domain tokenizer encode/decode 是否正确
2. batch_align_and_embed 的映射方式和生成函数是否一致
3. 模型 forward 输入输出是否正确
4. domain vocab 的 byte_fallback 情况
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

from taiji.resonance import ResonanceNeuron, get_domain_neuron_config
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.train_neuron import (
    load_domain_tokenizer, load_general_tokenizer,
    load_or_create_shared_embedding,
    OUTPUT_DIR,
)


def main():
    print("=" * 60)
    print("诊断 PPL 低但生成乱码的根因")
    print("=" * 60)

    # 加载
    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    shared_embedding = load_or_create_shared_embedding("cpu")

    print(f"\n[1] Tokenizer 信息:")
    print(f"  domain vocab_size: {domain_sp.get_piece_size()}")
    print(f"  general vocab_size: {general_sp.get_piece_size()}")

    # 测试 encode/decode
    test_texts = ["小猫", "从前有一只小熊", "大森林里住着小动物", "妈妈说宝宝乖"]
    print(f"\n[2] domain tokenizer encode/decode 测试:")
    for text in test_texts:
        ids = domain_sp.encode(text)
        decoded = domain_sp.decode(ids)
        pieces = [domain_sp.decode([i]) for i in ids]
        print(f"  '{text}'")
        print(f"    encode: {ids}")
        print(f"    pieces: {pieces}")
        print(f"    decode: '{decoded}'")
        # 检查 byte_fallback
        byte_tokens = [p for p in pieces if p.startswith('<') and p.endswith('>')]
        if byte_tokens:
            print(f"    ⚠️ byte_fallback tokens: {byte_tokens}")

    # 测试 domain → general 映射
    print(f"\n[3] domain → general token 映射测试:")
    for text in test_texts:
        domain_ids = domain_sp.encode(text)
        general_ids = []
        for tid in domain_ids:
            piece = domain_sp.decode([tid])
            gids = general_sp.encode(piece)
            if gids:
                general_ids.append(gids[0])
                # 检查映射是否 1:1
                if len(gids) > 1:
                    print(f"  ⚠️ '{piece}' → {gids} (多对一，取第一个)")
        # 反向解码
        back_pieces = [general_sp.decode([gid]) for gid in general_ids]
        back_text = "".join(back_pieces)
        print(f"  '{text}' → domain_ids={domain_ids}")
        print(f"    → general_ids={general_ids}")
        print(f"    → general decode: '{back_text}'")
        if back_text != text:
            print(f"    ⚠️ 往返不一致!")

    # 对比 batch_align_and_embed 和手动映射
    print(f"\n[4] batch_align_and_embed vs 手动映射对比:")
    text = "小猫在花园里玩耍"
    # batch_align_and_embed
    shared_emb_bat, targets_bat, mask_bat = batch_align_and_embed([text], domain_sp, general_sp, shared_embedding)
    print(f"  text: '{text}'")
    print(f"  batch_align_and_embed:")
    print(f"    targets (domain ids): {targets_bat[0].tolist()}")
    print(f"    shared_emb shape: {shared_emb_bat.shape}")
    print(f"    mask sum: {mask_bat.sum().item()}")

    # 手动映射
    domain_ids = domain_sp.encode(text)
    general_ids = []
    for tid in domain_ids:
        piece = domain_sp.decode([tid])
        gids = general_sp.encode(piece)
        if gids:
            general_ids.append(gids[0])
    manual_emb = shared_embedding(torch.tensor([general_ids]))
    print(f"  手动映射:")
    print(f"    domain_ids: {domain_ids}")
    print(f"    general_ids: {general_ids}")
    print(f"    manual_emb shape: {manual_emb.shape}")

    # 对比 embedding 值
    if shared_emb_bat.shape == manual_emb.shape:
        diff = (shared_emb_bat - manual_emb).abs().max().item()
        print(f"  embedding 差异: {diff:.6f}")
        if diff < 1e-6:
            print(f"    ✅ 映射一致")
        else:
            print(f"    ⚠️ 映射不一致！差异 {diff}")
    else:
        print(f"  ⚠️ shape 不一致: batch={shared_emb_bat.shape} vs manual={manual_emb.shape}")

    # 加载模型测试
    print(f"\n[5] 模型 forward 测试:")
    ckpt_path = os.path.join(OUTPUT_DIR, "neuron_zh_par1.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = get_domain_neuron_config("zh", spec="compact")
        neuron = ResonanceNeuron(cfg)
        neuron.load_state_dict(ckpt['state_dict'])
        neuron.eval()

        # 用 batch_align_and_embed 的输入
        result = neuron.forward(shared_emb_bat, return_logits=True)
        logits = result["logits"]
        print(f"  logits shape: {logits.shape}")
        print(f"  targets shape: {targets_bat.shape}")

        # 检查第一个预测
        first_pred = logits[0, -1, :].argmax().item()
        first_top5 = logits[0, -1, :].topk(5)
        print(f"  最后一个位置预测:")
        print(f"    argmax: {first_pred} → '{domain_sp.decode([first_pred])}'")
        print(f"    top-5: {first_top5.indices.tolist()} → {[domain_sp.decode([i]) for i in first_top5.indices.tolist()]}")
        print(f"    top-5 probs: {[f'{p:.3f}' for p in first_top5.values.softmax(dim=-1).tolist()]}")

        # 检查 teacher-forcing 准确率
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = targets_bat[:, 1:].contiguous()
        shift_mask = mask_bat[:, 1:].contiguous()
        preds = shift_logits.argmax(dim=-1)
        valid = shift_mask.bool()
        correct = (preds[valid] == shift_targets[valid]).sum().item()
        total = valid.sum().item()
        print(f"\n  teacher-forcing 准确率: {correct}/{total} = {correct/total*100:.1f}%")

        # 检查预测的 token 分布
        print(f"\n  预测 token 分布（前20个位置）:")
        for i in range(min(20, shift_logits.size(1))):
            if valid[0, i]:
                pred = preds[0, i].item()
                target = shift_targets[0, i].item()
                pred_piece = domain_sp.decode([pred])
                target_piece = domain_sp.decode([target])
                mark = "✓" if pred == target else "✗"
                print(f"    pos {i}: pred={pred}('{pred_piece}') target={target}('{target_piece}') {mark}")
    else:
        print(f"  ❌ checkpoint 不存在: {ckpt_path}")

    print(f"\n{'='*60}")
    print("诊断完成")


if __name__ == "__main__":
    main()
