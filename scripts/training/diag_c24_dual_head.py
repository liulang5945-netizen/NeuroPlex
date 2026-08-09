"""C24 双头可行性验证：foundation_v1_general body + 域头 组合（临时脚本）。

假设：双头方案 = body 用 foundation_v1_general（保留 general 判定对角 4/4）
+ 域头（生成）。验证：
1. 直接组合（域头复制自 foundation_v1_sft）的域空间 PPL——body/域头是否匹配
2. general 判定对角是否保留
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from taiji.resonance import ResonanceNeuron
from scripts.training.utils import load_general_tokenizer
from scripts.training.train_cross_domain_collab import load_tokenizer_for_vocab

PROMPTS = [
    ("code", "Write a Python function to compute the Fibonacci sequence"),
    ("math", "If a train travels at 60 mph for 3 hours, how many miles does it travel?"),
    ("zh", "写一个 Python 函数计算斐波那契数列"),
    ("en", "What is the capital of France?"),
]

GENERAL_VOCAB = {"code": 12000, "math": 10000, "zh": 50000, "en": 16000}


def load_neuron(path, general_head=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["neuron_config"]
    cfg.unified_field_dim = None
    n = ResonanceNeuron(cfg)
    sd = {k: v for k, v in ck["state_dict"].items() if not k.startswith("lm_head")}
    n.load_state_dict(sd, strict=False)
    if general_head is not None:
        n.lm_head = general_head
    return n, cfg


def main():
    general = load_general_tokenizer()
    # general 256K 头（foundation_v1_general 共享头）
    gs = torch.load("data/foundation_v1_general/shared_lm_head.pt", map_location="cpu", weights_only=False)
    general_head = torch.nn.Linear(512, 256000, bias=False)
    general_head.weight.data.copy_(gs["weight"])
    shared_emb = torch.nn.Embedding(256000, 512)
    shared_emb.weight.data.copy_(torch.load("data/foundation_v1_general/shared_embedding.pt",
                                            map_location="cpu", weights_only=False))

    print("=== 组合 1：foundation_v1_general body + foundation_v1_sft 域头 ===")
    for d in ["code", "math", "zh", "en"]:
        gen_neuron, _ = load_neuron(f"data/foundation_v1_general/neuron_{d}.pt")
        sft_ck = torch.load(f"data/foundation_v1_sft/neuron_{d}.pt", map_location="cpu", weights_only=False)
        # 用 sft 的域头替换 general body 的 lm_head
        dom_head = torch.nn.Linear(512, GENERAL_VOCAB[d], bias=False)
        dom_head.weight.data.copy_(sft_ck["state_dict"]["lm_head.weight"])
        gen_neuron.lm_head = dom_head
        # 域空间 PPL：用该域 SFT 数据测
        dsp = load_tokenizer_for_vocab(d, GENERAL_VOCAB[d])
        sft = torch.load(f"data/sft/{d}_sft.pt", map_location="cpu", weights_only=False)
        total_loss, total_tok = 0.0, 0
        gen_neuron.eval()
        import random
        random.seed(0)
        with torch.no_grad():
            for _ in range(4):
                for s in random.sample(sft, 2):
                    text = s["full"]
                    from taiji.resonance.translator import build_position_alignment
                    g_ids, d_tgt = build_position_alignment(text, dsp, general)
                    if len(g_ids) < 2:
                        continue
                    g_ids = torch.tensor(g_ids, dtype=torch.long).unsqueeze(0)
                    d_tgt_t = torch.tensor(d_tgt, dtype=torch.long)
                    e = shared_emb(g_ids)
                    r = gen_neuron.forward(e, return_logits=True)
                    lg = r["logits"][:, :-1, :].contiguous()
                    t = d_tgt_t[1:].unsqueeze(0).clone()
                    t[t < 0] = -100
                    valid = (t >= 0)
                    if valid.sum().item() == 0:
                        continue
                    l = F.cross_entropy(lg.view(-1, lg.size(-1)), t.view(-1), ignore_index=-100, reduction="sum")
                    total_loss += l.item()
                    total_tok += int(valid.sum().item())
        import math
        ppl = math.exp(min(total_loss / max(total_tok, 1), 20))
        print(f"  [{d}] 域空间 answer PPL = {ppl:.1f}")


if __name__ == "__main__":
    main()
