"""临时诊断：dialogue neuron 在 zh 词表空间（home embedding + zh 50K 输出头）生成中文
是否正常（对比 C19 的 general decode 错位）。验证后清理。
"""
import os
import sys

os.environ.setdefault("TAIJI_TEST_MODE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from scripts.training.utils import load_general_tokenizer, create_shared_embedding
from taiji.resonance.neuron import ResonanceNeuron

DIALOGUE_ID = "zh_aug0_dialogue"
PROMPTS = ["你好，请介绍一下你自己", "什么是人工智能？", "今天天气怎么样？"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nid", default="zh_aug0_dialogue")
    ap.add_argument("--inject-lora", default="", help="C16 collab ckpt，注入其 lora_state 对比")
    args = ap.parse_args()
    DIALOGUE_ID = args.nid

    # 加载 dialogue neuron（zh 域空间：zh 表 embedding + zh 50K lm_head）
    ck = torch.load(f"data/neurons/neuron_{DIALOGUE_ID}.pt", map_location="cpu", weights_only=False)
    cfg = ck["neuron_config"]
    cfg.unified_field_dim = None
    n = ResonanceNeuron(cfg)
    n.load_state_dict(ck["state_dict"], strict=False)
    n.eval()
    print(f"lm_head vocab: {n.lm_head.out_features}")

    # 可选：注入 C16 LoRA（对比 body 是否被 C16 训练扭曲）
    if args.inject_lora:
        c16 = torch.load(args.inject_lora, map_location="cpu", weights_only=False)
        ls = c16.get("lora_state", {})
        if DIALOGUE_ID in ls:
            n.enable_lora(16, layers=None)
            n.lora_adapters.load_state_dict(ls[DIALOGUE_ID])
            print(f"  已注入 C16 LoRA（{DIALOGUE_ID}）")
        else:
            print(f"  ⚠️ C16 ckpt 无 {DIALOGUE_ID} 的 lora_state")

    # home embedding（zh 表 50000）
    emb = create_shared_embedding("cpu")
    ses = ck.get("shared_embedding_state", {})
    w = ses["weight"] if isinstance(ses, dict) else ses
    emb.weight.data.copy_(w)
    print(f"home embedding vocab: {emb.num_embeddings}")

    general_sp = load_general_tokenizer()
    import sentencepiece as spm
    zh_sp = spm.SentencePieceProcessor()
    zh_sp.Load("taiji/domains/zh/sp_zh.model")
    print(f"zh tokenizer vocab: {zh_sp.GetPieceSize()}")

    # 生成（v3 口径）：general ids 输入（general 表）→ zh 50K logits →
    # domain token → 文本 → general ids 回填（保持自回归输入 general 空间）→ zh decode
    for prompt in PROMPTS:
        cur = general_sp.encode(prompt)
        gen_ids = []
        for step in range(30):
            ids_t = torch.tensor([cur], dtype=torch.long)
            r = n.forward(emb(ids_t), round_num=1, return_logits=True)
            logits = r["logits"][0, -1]  # zh 50K 空间
            probs = torch.softmax(logits / 0.9, dim=-1)
            topv, _ = torch.topk(probs, 50)
            probs[probs < topv[-1].item()] = 0.0
            probs = probs / probs.sum()
            nxt = int(torch.multinomial(probs, 1).item())
            gen_ids.append(nxt)
            piece = zh_sp.id_to_piece(nxt)
            if piece in ("</s>", "<s>"):
                break
            # domain token → 文本 → general ids 回填（v3 修复：保持输入 general 空间）
            piece_text = zh_sp.decode([nxt])
            new_general = general_sp.encode(piece_text)
            if new_general:
                cur.extend(new_general)
        out = zh_sp.DecodeIds(gen_ids)
        print(f"\n[{prompt[:12]}]\n  → {out}")


if __name__ == "__main__":
    main()
