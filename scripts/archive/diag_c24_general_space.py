"""C24 诊断：general 256K 空间 NLL 排序有效性（临时脚本）。

背景：C20 当年判定 5/5 用 foundation_v1_general（general 256K 共享头）。
C24 换域头后 native NLL 不可比（en 膨胀）。验证：
1. foundation_v1_general 完整 neuron（body+general 256K 头）对 5 类回合的 NLL 排序
   ——general 空间信号是否仍然有效（code 回合 code 低？）
2. foundation_v1_sft body + foundation_v1_general general 头（混合）的 NLL 排序
   ——SFT 微调 body 是否破坏 general 空间能力（双头可行性）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from taiji.resonance import ResonanceNeuron

PROMPTS = [
    ("code", "Write a Python function to compute the Fibonacci sequence"),
    ("math", "If a train travels at 60 mph for 3 hours, how many miles does it travel?"),
    ("zh", "写一个 Python 函数计算斐波那契数列"),
    ("en", "What is the capital of France?"),
]

BASE_GENERAL = "data/foundation_v1_general"
BASE_SFT = "data/foundation_v1_sft"
EMBED_DIM = 512


def load_neuron(base_dir, domain, general_head=None):
    ck = torch.load(os.path.join(base_dir, f"neuron_{domain}.pt"), map_location="cpu", weights_only=False)
    cfg = ck["neuron_config"]
    cfg.unified_field_dim = None
    n = ResonanceNeuron(cfg)
    # lm_head 兼容：general 基座 ckpt 存 256K 共享头，先删掉再按需注入
    sd = {k: v for k, v in ck["state_dict"].items() if not k.startswith("lm_head")}
    n.load_state_dict(sd, strict=False)
    if general_head is not None:
        n.lm_head = general_head
    return n, cfg


def load_emb(base_dir):
    emb = torch.nn.Embedding(256000, EMBED_DIM)
    emb.weight.data.copy_(torch.load(os.path.join(base_dir, "shared_embedding.pt"), map_location="cpu", weights_only=False))
    return emb


def nll(neuron, emb, general_sp, text):
    ids = torch.tensor([general_sp.encode(text)], dtype=torch.long)
    e = emb(ids)
    with torch.no_grad():
        r = neuron.forward(e, return_logits=True)
        lg = r["logits"][:, :-1, :].contiguous()  # [1, L-1, V]
        tgt = ids[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(lg.view(-1, lg.size(-1)), tgt.view(-1), reduction="mean")
    return float(loss)


def main():
    import sentencepiece as spm
    general = spm.SentencePieceProcessor()
    general.Load("taiji/domains/general/sp_general.model")

    # 方案 1：foundation_v1_general 完整（body + general 256K 头）
    print("=== 1. foundation_v1_general（body + general 256K 头）NLL ===")
    gen_state = torch.load(os.path.join(BASE_GENERAL, "neuron_code.pt"), map_location="cpu", weights_only=False)
    gen_sd = gen_state["state_dict"]
    # general 256K 共享头直接来自 ckpt lm_head.weight（256000×512）
    general_head = torch.nn.Linear(512, 256000, bias=False)
    general_head.weight.data.copy_(gen_sd["lm_head.weight"])
    neurons_g = {d: load_neuron(BASE_GENERAL, d, general_head=general_head) for d in ["code", "math", "zh", "en"]}
    emb_g = load_emb(BASE_GENERAL)
    print(f"{'回合':<8} " + " ".join(f"{d:>10}" for d in ["code", "math", "zh", "en"]))
    for tag, text in PROMPTS:
        row = [f"{tag:<8}"]
        for dom in ["code", "math", "zh", "en"]:
            n, _ = neurons_g[dom]
            row.append(f"{nll(n, emb_g, general, text):>10.2f}")
        print(" ".join(row))

    # 方案 3：foundation_v1（SFT 前基座）body + general 256K 头——双头重训可行性
    print("\n=== 3. foundation_v1（SFT 前基座）body + general 256K 头 NLL ===")
    neurons_b = {d: load_neuron("data/foundation_v1", d, general_head=general_head) for d in ["code", "math", "zh", "en"]}
    emb_b = load_emb(BASE_GENERAL)
    print(f"{'回合':<8} " + " ".join(f"{d:>10}" for d in ["code", "math", "zh", "en"]))
    for tag, text in PROMPTS:
        row = [f"{tag:<8}"]
        for dom in ["code", "math", "zh", "en"]:
            n, _ = neurons_b[dom]
            row.append(f"{nll(n, emb_b, general, text):>10.2f}")
        print(" ".join(row))
    print("\n=== 2. foundation_v1_sft body + general 256K 头（混合）NLL ===")
    neurons_s = {d: load_neuron(BASE_SFT, d, general_head=general_head) for d in ["code", "math", "zh", "en"]}
    emb_s = load_emb(BASE_GENERAL)  # general 空间共享 embedding
    print(f"{'回合':<8} " + " ".join(f"{d:>10}" for d in ["code", "math", "zh", "en"]))
    for tag, text in PROMPTS:
        row = [f"{tag:<8}"]
        for dom in ["code", "math", "zh", "en"]:
            n, _ = neurons_s[dom]
            row.append(f"{nll(n, emb_s, general, text):>10.2f}")
        print(" ".join(row))


if __name__ == "__main__":
    main()
