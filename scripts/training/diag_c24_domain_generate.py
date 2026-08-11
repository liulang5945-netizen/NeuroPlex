"""C24 诊断：单独验证 4 个域头 SFT neuron 的域生成能力（不经过 ensemble/head）。

验证目标：
1. 每个域头 neuron 在域空间能否连贯生成（C24 训练成果本身）
2. 生成 decode 是否走对词表空间（code 12K / math 10K / zh 50K / en 16K）
"""
import os
import sys

os.environ.setdefault("TAIJI_TEST_MODE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from taiji.resonance import ResonanceNeuron
from taiji.resonance.translator import build_position_alignment
from scripts.training.utils import load_general_tokenizer
from scripts.training.train_cross_domain_collab import load_tokenizer_for_vocab

DEVICE = "cpu"
DOMAIN_VOCAB = {"code": 12000, "math": 10000, "zh": 50000, "en": 16000}
OUT_DIR = "data/foundation_v1_sft"

PROMPTS = {
    "code": "Write a Python function to compute the Fibonacci sequence",
    "math": "If a train travels at 60 mph for 3 hours, how many miles does it travel?",
    "zh": "写一个 Python 函数计算斐波那契数列",
    "en": "What is the capital of France?",
}


def load_neuron(domain: str, out_dir: str = OUT_DIR):
    ckpt = torch.load(os.path.join(out_dir, f"neuron_{domain}.pt"), map_location=DEVICE, weights_only=False)
    cfg = ckpt["neuron_config"]
    cfg.unified_field_dim = None
    neuron = ResonanceNeuron(cfg).to(DEVICE)
    neuron.load_state_dict(ckpt["state_dict"], strict=False)
    neuron.eval()
    emb = torch.nn.Embedding(256000, 512)
    emb.weight.data.copy_(torch.load(os.path.join(out_dir, "shared_embedding.pt"), map_location=DEVICE))
    return neuron, emb


def generate(neuron, emb, domain_sp, general_sp, prompt, max_tokens=40):
    # C24 训练样本 = prompt + "\n" + response（answer 起点在 prompt+'\n' 之后，
    # ckpt 标记 c24_domain_sft=True）。生成输入必须补 "\n"，否则模型未见该模式。
    general_ids = general_sp.encode(prompt + "\n")
    ids = torch.tensor([general_ids], dtype=torch.long, device=DEVICE)
    eos_id = domain_sp.eos_id() if hasattr(domain_sp, "eos_id") else 3
    out_ids = []
    with torch.no_grad():
        for _ in range(max_tokens):
            x = emb(ids)
            r = neuron.forward(x, return_logits=True)
            logits = r["logits"][:, -1, :].float()
            logits[0, eos_id] += 0.5
            topk_v, topk_i = torch.topk(logits, min(15, logits.size(-1)))
            probs = F.softmax(topk_v, dim=-1)
            nxt = topk_i[0, torch.multinomial(probs, 1).item()].item()
            out_ids.append(nxt)
            if nxt == eos_id:
                break
            # domain token → 文本 → general ids 回填（自回归输入保持 general 空间）
            piece = domain_sp.decode([nxt])
            new_g = general_sp.encode(piece)
            if not new_g:
                new_g = [general_sp.pad_id()]
            ids = torch.cat([ids, torch.tensor([new_g], dtype=torch.long, device=DEVICE)], dim=1)
    return domain_sp.DecodeIds(out_ids)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=OUT_DIR, help="neuron 目录：foundation_v1（SFT 前）或 foundation_v1_sft（SFT 后）")
    args = ap.parse_args()
    general_sp = load_general_tokenizer()
    for dom, prompt in PROMPTS.items():
        domain_sp = load_tokenizer_for_vocab(dom, DOMAIN_VOCAB[dom])
        neuron, emb = load_neuron(dom, args.dir)
        text = generate(neuron, emb, domain_sp, general_sp, prompt)
        print(f"\n── [{dom}] {prompt}\n  → {text}")


if __name__ == "__main__":
    main()
