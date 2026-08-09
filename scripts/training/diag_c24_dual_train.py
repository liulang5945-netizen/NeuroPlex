"""C24 双头冒烟训练：从 foundation_v1_general 出发（body 保留 general 判定），
加域头（生成），双 loss 训练（域 SFT + general 保留）。

验证目标：
1. 域头 answer PPL 能否收敛（生成能力）
2. 训练后 general 256K 空间 NLL 对角是否保留（判定能力）
"""
import os
import sys
import random
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from taiji.resonance import ResonanceNeuron
from taiji.resonance.translator import build_position_alignment
from scripts.training.utils import load_general_tokenizer
from scripts.training.train_cross_domain_collab import load_tokenizer_for_vocab

GENERAL_VOCAB = {"code": 12000, "math": 10000, "zh": 50000, "en": 16000}
SEQ_LEN = 192
LR = 5e-4
GENERAL_LOSS_WEIGHT = 0.5  # general 保留 loss 权重


def load_general_base(domain, device="cpu"):
    """加载 foundation_v1_general（body + general 256K 共享头）。"""
    ck = torch.load(f"data/foundation_v1_general/neuron_{domain}.pt", map_location=device, weights_only=False)
    cfg = ck["neuron_config"]
    cfg.unified_field_dim = None
    neuron = ResonanceNeuron(cfg).to(device)
    sd = {k: v for k, v in ck["state_dict"].items() if not k.startswith("lm_head")}
    neuron.load_state_dict(sd, strict=False)
    # general 256K 共享头
    gs = torch.load("data/foundation_v1_general/shared_lm_head.pt", map_location=device, weights_only=False)
    general_head = torch.nn.Linear(512, 256000, bias=False).to(device)
    general_head.weight.data.copy_(gs["weight"])
    neuron.lm_head = general_head  # 判定头（general 256K）
    # 域头（新训练，生成）
    dom_head = torch.nn.Linear(512, GENERAL_VOCAB[domain], bias=False).to(device)
    torch.nn.init.normal_(dom_head.weight, std=512 ** -0.5)
    neuron.domain_lm_head = dom_head
    return neuron, cfg


def build_batch(samples, domain_sp, general_sp, shared_emb, device):
    rows = []
    for text, prompt in samples:
        g_ids, d_targets, ans_start = _build_sample(text, prompt, domain_sp, general_sp)
        g_ids = torch.cat([g_ids, torch.tensor([general_sp.eos_id()])])
        d_targets = torch.cat([d_targets, torch.tensor([domain_sp.eos_id()])])
        if len(g_ids) > SEQ_LEN:
            g_ids = torch.cat([g_ids[:SEQ_LEN - 1], torch.tensor([general_sp.eos_id()])])
            d_targets = torch.cat([d_targets[:SEQ_LEN - 1], torch.tensor([domain_sp.eos_id()])])
        rows.append((g_ids, d_targets, ans_start))
    max_len = max(len(r[0]) for r in rows)
    B = len(rows)
    ids = torch.zeros(B, max_len, dtype=torch.long)
    d_tgt = torch.full((B, max_len), -100, dtype=torch.long)
    g_tgt = torch.full((B, max_len), -100, dtype=torch.long)
    sft = torch.zeros(B, max_len, dtype=torch.bool)
    for b, (g, d, a) in enumerate(rows):
        L = len(g)
        ids[b, :L] = g
        d_tgt[b, :L] = d
        g_tgt[b, :L] = g  # general next-token 目标 = 输入本身（shift 在 loss 内）
        sft[b, a:L] = True
    emb = shared_emb(ids.to(device))
    return emb, d_tgt.to(device), g_tgt.to(device), sft.to(device)


def _build_sample(text, prompt, domain_sp, general_sp):
    g_ids, d_targets = build_position_alignment(text, domain_sp, general_sp)
    p_ids = general_sp.encode(prompt + "\n")
    k = 0
    n = min(len(p_ids), len(g_ids))
    while k < n and int(g_ids[k]) == int(p_ids[k]):
        k += 1
    return g_ids, d_targets, k


def main():
    random.seed(0)
    torch.manual_seed(0)
    device = "cpu"
    domain = "code"
    general = load_general_tokenizer()
    dsp = load_tokenizer_for_vocab(domain, GENERAL_VOCAB[domain])
    shared_emb = torch.nn.Embedding(256000, 512)
    shared_emb.weight.data.copy_(torch.load("data/foundation_v1_general/shared_embedding.pt",
                                            map_location="cpu", weights_only=False))
    for p in shared_emb.parameters():
        p.requires_grad = False

    sft = torch.load(f"data/sft/{domain}_sft.pt", map_location="cpu", weights_only=False)
    random.shuffle(sft)
    train, ev = sft[:30], sft[30:40]
    train_pairs = [(s["full"], s["prompt"]) for s in train]
    ev_pairs = [(s["full"], s["prompt"]) for s in ev]

    neuron, cfg = load_general_base(domain, device)
    shared_emb.to(device)
    print(f"[code] body 参数 {sum(p.numel() for p in neuron.parameters())/1e6:.1f}M")

    # 冻结 shared_embedding（C24 配方），全部 body + 双头可训
    opt = torch.optim.AdamW(
        [p for p in neuron.parameters()] + [p for p in neuron.domain_lm_head.parameters()],
        lr=LR, weight_decay=0.1)

    print("\n=== 训练前：general 256K 判定对角（4 回合 NLL）===")
    PROMPTS = [
        ("code", "Write a Python function to compute the Fibonacci sequence"),
        ("math", "If a train travels at 60 mph for 3 hours, how many miles does it travel?"),
        ("zh", "写一个 Python 函数计算斐波那契数列"),
        ("en", "What is the capital of France?"),
    ]
    neuron.eval()
    with torch.no_grad():
        for tag, text in PROMPTS:
            ids = torch.tensor([general.encode(text)], dtype=torch.long).to(device)
            e = shared_emb(ids)
            r = neuron.forward(e, return_logits=True)
            lg = r["logits"][:, :-1, :].contiguous()
            t = ids[:, 1:].contiguous()
            loss = F.cross_entropy(lg.view(-1, lg.size(-1)), t.view(-1), reduction="mean")
            print(f"  [{tag:<8}] general NLL = {loss.item():.2f}")

    neuron.train()
    steps = 60
    print("\n=== 训练 ===")
    for step in range(1, steps + 1):
        batch = random.sample(train_pairs, 4)
        emb, d_tgt, g_tgt, sft_mask = build_batch(batch, dsp, general, shared_emb, device)
        r = neuron.forward(emb, return_logits=True)
        # 域头 SFT loss（answer masked）
        dl = r["logits"][:, :-1, :].contiguous()
        dt = d_tgt[:, 1:].clone().contiguous()
        am = sft_mask[:, 1:].contiguous()
        dt[~am] = -100
        nt = max(int(am.sum().item()), 1)
        loss_dom = F.cross_entropy(dl.view(-1, dl.size(-1)), dt.view(-1), ignore_index=-100, reduction="sum") / nt
        # general 保留 loss（next-token，全序列）
        gl = neuron.lm_head(emb)  # 复用输入 embedding → general logits
        gt = g_tgt[:, 1:].clone().contiguous()
        loss_gen = F.cross_entropy(gl[:, :-1, :].contiguous().view(-1, gl.size(-1)),
                                   gt.view(-1), ignore_index=-100)
        loss = loss_dom + GENERAL_LOSS_WEIGHT * loss_gen
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(neuron.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            print(f"  [step {step}] dom_loss={loss_dom.item():.3f} gen_loss={loss_gen.item():.3f} "
                  f"dom_PPL={math.exp(min(loss_dom.item(), 20)):.1f}")

    # 训练后验证
    print("\n=== 训练后：域头 answer PPL（eval 集）===")
    neuron.eval()
    with torch.no_grad():
        e, dt, gt, sm = build_batch(ev_pairs, dsp, general, shared_emb, device)
        r = neuron.forward(e, return_logits=True)
        lg = r["logits"][:, :-1, :].contiguous()
        t = dt[:, 1:].clone().contiguous()
        am = sm[:, 1:].contiguous()
        t[~am] = -100
        nt = int(am.sum().item())
        l = F.cross_entropy(lg.view(-1, lg.size(-1)), t.view(-1), ignore_index=-100, reduction="sum")
        print(f"  code 域 answer PPL = {math.exp(min(l.item()/max(nt,1), 20)):.1f}")

    print("\n=== 训练后：general 256K 判定对角 ===")
    with torch.no_grad():
        for tag, text in PROMPTS:
            ids = torch.tensor([general.encode(text)], dtype=torch.long).to(device)
            e = shared_emb(ids)
            r = neuron.forward(e, return_logits=True)
            lg = r["logits"][:, :-1, :].contiguous()
            t = ids[:, 1:].contiguous()
            loss = F.cross_entropy(lg.view(-1, lg.size(-1)), t.view(-1), reduction="mean")
            print(f"  [{tag:<8}] general NLL = {loss.item():.2f}")


if __name__ == "__main__":
    main()
